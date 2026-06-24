import os
import cv2
import glob
import json
import random
import numpy as np
import pandas as pd
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from multiprocessing import freeze_support

#1. Semillas para reproducibilidad

pl.seed_everything(42, workers=True)

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(42)

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

#2. Configuración. Introducir las arquitecturas, encoders y rutas de los pesos de los modelos entrenados. Cambiar el valor de la clase objetivo (TARGET_VALUE) según la clase que se quiera segmentar. Cambiar el umbral de decisión (THRESHOLD) según se necesite.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.5 #Umbral de decisión para convertir probabilidades en predicciones binarias (0/1). Cambiar según se necesite.

MODEL_CONFIGS = [
    {"arch": "", "encoder": "", "weights": ""},
    {"arch": "",    "encoder": "", "weights": ""},
    {"arch": "",     "encoder": "", "weights": ""},
]
#Introducir aquí la ruta del Excel que contiene la división en splits de los videos, así como las rutas de las imágenes y máscaras (de todo, luego a través de la división en splits con el Excel se seleccionan las imágenes que se necesiten)
EXCEL_PATH   = ""
IMG_ROOT     = ""
MASK_ROOT    = ""
TARGET_VALUE = 53   # Valor de la clase objetivo (anastomotic_line) en la máscara. Cambiar según la clase que se quiera segmentar.

OUTPUT_JSON  = "ensemble_results.json"  # Nombre del archivo JSON de salida con los resultados del ensemble.

#3. Aumentaciones y preprocesado. Cambiar según se necesite un tamaño u otro. Este está pensado para que coincida con el tamaño de entrada de los modelos entrenados (512x512).

preprocessing = A.Compose([ToTensorV2()])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    ])

# Dataset binario para clases. Devuelve la máscara binaria de la clase objetivo (target_value) y el frame correspondiente.

class BinaryDataset(Dataset):
    def __init__(self, images, masks, target_value, augmentation=None, preprocessing=None):
        self.images        = images
        self.masks         = masks
        self.target_value  = target_value
        self.augmentation  = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img  = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask == self.target_value).astype(np.float32)

        if self.augmentation:
            aug  = self.augmentation(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if self.preprocessing:
            prep = self.preprocessing(image=img, mask=mask)
            img, mask = prep["image"], prep["mask"]

        # shape mask: (1, H, W) float32  — igual que en entrenamiento
        return img, mask.unsqueeze(0), self.images[i]

# 5. Carga de datos desde el Excel. 

def get_files_by_split(split_value):
    """split_value: 1=validación, 2=test"""
    df = pd.read_excel(EXCEL_PATH)
    col_split = [c for c in df.columns if "Train" in c][0]
    print(f"Columna de split detectada: {col_split}")

    imgs, msks = [], []
    for _, row in df.iterrows():
        if row[col_split] != split_value:
            continue
        video   = row["Video"]
        img_dir = os.path.join(IMG_ROOT, video)
        msk_dir = os.path.join(MASK_ROOT, video)
        imgs += sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        msks += sorted(glob.glob(os.path.join(msk_dir, "*.png")))

    label = "validación" if split_value == 1 else "test"
    print(f"Vídeos de {label}: {df[df[col_split] == split_value]['Video'].tolist()}")
    print(f"Total imágenes   : {len(imgs)}")
    return imgs, msks

#6. Modelo binario. Se utiliza la misma arquitectura y encoder que en el entrenamiento, pero sin pesos preentrenados. Se cargan los pesos entrenados posteriormente.
class BinaryModel(torch.nn.Module):
    def __init__(self, arch, encoder_name):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(params["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)   # logits (B, 1, H, W)


def load_model(config):
    model = BinaryModel(arch=config["arch"], encoder_name=config["encoder"])

    state = torch.load(config["weights"], map_location=DEVICE)

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"Cargado: {config['weights']}")
    return model

# 7. Votación del ensemble por mayoría. Devuelve la máscara binaria final (0/1) según la mayoría de modelos que voten positivo.

def majority_voting(binary_preds):
    stack = torch.stack(binary_preds, dim=0).float()  # (N, H, W)
    votes = stack.sum(dim=0)                           # (H, W)
    return (votes >= (len(binary_preds) // 2+1)).long()      # mayoría estricta

# 8. MAIN. Realiza la inferencia del ensemble sobre los splits de validación y test, y guarda los resultados en un JSON. Modificar en función de las métricas que se quiera obtener. Actualmente guarda IoU, F1, precisión y TP/TN/FP/FN por imagen, así como métricas globales micro y macro.

def run_inference(models, imgs, msks, label, output_json):
    """Ejecuta el ensemble sobre un conjunto y guarda el JSON."""
    print(f"\n── Preparando dataset {label} ──")
    dataset = BinaryDataset(
        images=imgs,
        masks=msks,
        target_value=TARGET_VALUE,
        augmentation=get_validation_augmentation(),
        preprocessing=preprocessing,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    print(f"\n── Ejecutando ensemble ({label}) ──")
    per_image_results = []
    total_tp = total_tn = total_fp = total_fn = 0

    with torch.no_grad():
        for images, masks, paths in loader:
            images   = images.float().to(DEVICE)
            masks_2d = masks.squeeze(1).to(DEVICE)

            probs_per_model = []
            for model in models:
                    logits = model(images)
                    probs  = torch.sigmoid(logits.squeeze(1))
                    probs_per_model.append(probs)

            for b in range(images.shape[0]):
                avg_probs  = torch.stack([probs_per_model[m][b] for m in range(len(models))], dim=0).mean(dim=0)
                final_pred = (avg_probs >= THRESHOLD).long()
                true_mask  = masks_2d[b].long()

                pred_b = final_pred.bool()
                true_b = true_mask.bool()

                tp = (pred_b  &  true_b).sum().item()
                tn = (~pred_b & ~true_b).sum().item()
                fp = (pred_b  & ~true_b).sum().item()
                fn = (~pred_b &  true_b).sum().item()

                total_tp += tp;  total_tn += tn
                total_fp += fp;  total_fn += fn

                iou_img       = tp / (tp + fp + fn + 1e-7) if (tp + fp + fn) > 0 else np.nan
                f1_img        = 2*tp / (2*tp + fp + fn + 1e-7)
                precision_img = tp / (tp + fp + 1e-7)

                per_image_results.append({
                    "image_path": paths[b],
                    "iou":       float(iou_img) if true_b.sum().item() > 0 else None,
                    "f1":        float(f1_img)  if true_b.sum().item() > 0 else None,
                    "precision": float(precision_img),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                })

    # Métricas globales micro
    micro_iou       = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    micro_f1        = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7)
    micro_precision = total_tp / (total_tp + total_fp + 1e-7)

    # Macro IoU
    iou_per_image = [r["iou"] for r in per_image_results if r["iou"] is not None]
    macro_iou = float(np.mean(iou_per_image)) if iou_per_image else float("nan")

    print(f"\n── Resultados Ensemble [{label}] ─────────────────")
    print(f"  Micro IoU      : {micro_iou:.4f}")
    print(f"  Macro IoU      : {macro_iou:.4f}")
    print(f"  F1             : {micro_f1:.4f}")
    print(f"  Precisión      : {micro_precision:.4f}")
    print(f"  Imágenes       : {len(per_image_results)}")
    print(f"  TP={total_tp}  TN={total_tn}  FP={total_fp}  FN={total_fn}")
    print("─────────────────────────────────────────────────\n")

    output = {
        "summary": {
            "split":             label,
            "micro_iou":         float(micro_iou),
            "macro_iou":         float(macro_iou),
            "micro_f1":          float(micro_f1),
            "micro_precision":   float(micro_precision),
            "total_tp": total_tp, "total_tn": total_tn,
            "total_fp": total_fp, "total_fn": total_fn,
            "n_images":  len(per_image_results),
            "threshold": THRESHOLD,
            "models":    [cfg["weights"] for cfg in MODEL_CONFIGS],
        },
        "per_image": per_image_results,
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Resultados guardados en: {output_json}")


def main():
    # 1. Cargar modelos una sola vez
    print("\n── Cargando modelos ──")
    models = [load_model(cfg) for cfg in MODEL_CONFIGS]

     #2. Validación (split=1)
    val_imgs, val_msks = get_files_by_split(1)
    run_inference(models, val_imgs, val_msks,
                  label="validación",
                  output_json="ensemble_val_anastomotic_line.json")

    # 3. Test (split=2)
    test_imgs, test_msks = get_files_by_split(2)
    run_inference(models, test_imgs, test_msks,
                  label="test",
                  output_json="ensemble_test_anastomotic_line.json")


if __name__ == "__main__":
    freeze_support()
    main()