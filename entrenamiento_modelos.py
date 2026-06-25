import os
from unittest import result
from cuda.benchmarks.test_pointer_attributes import idx
import cv2
import glob
import json
import random
import numpy as np
import pandas as pd
import albumentations as A
from pytest import param
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt  
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

#Semillas para reproducibilidad

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

#2. Aumentaciones y preprocesado
def get_training_augmentation():
    return A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    ])

preprocessing = A.Compose([ToTensorV2()])

# 3. DATASET BINARIO PARA CLASES

class BinaryDataset(Dataset):
    def __init__(self, images, masks, target_value, augmentation=None, preprocessing=None):
        self.images = images
        self.masks = masks
        self.target_value = target_value
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask == self.target_value).astype(np.float32)

        if self.augmentation:
            aug = self.augmentation(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if self.preprocessing:
            prep = self.preprocessing(image=img, mask=mask)
            img, mask = prep["image"], prep["mask"]

        return img, mask.unsqueeze(0), self.images[i]


class BinarySegModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights=None,checkpoint_path=None):
        super().__init__()

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.encoder.load_state_dict(checkpoint, strict=False)
            print(f"Encoder inicializado desde checkpoint: {checkpoint_path}")
            print("Claves del checkpoint:", list(checkpoint.keys())[:5])
            print("Claves del encoder:", list(self.model.encoder.state_dict().keys())[:5])
        

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(params["std"]).view(1,3,1,1))

        self.loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

        # Acumuladores TRAIN
        self.train_tp = 0
        self.train_tn = 0
        self.train_fp = 0
        self.train_fn = 0

        # Acumuladores VAL
        self.val_tp = 0
        self.val_tn = 0
        self.val_fp = 0
        self.val_fn = 0

        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

    def shared_step(self, batch):
        img, mask, _ = batch
        logits = self.forward(img)
        loss = self.loss_fn(logits, mask)
        pred = (torch.sigmoid(logits) > 0.5).float()
        return loss, pred, mask

    # ---------------- TRAINING ----------------
    def training_step(self, batch, batch_idx):
        loss, pred, mask = self.shared_step(batch)

        pred = pred.view(-1)
        mask = mask.view(-1)

        self.train_tp += ((pred == 1) & (mask == 1)).sum()
        self.train_tn += ((pred == 0) & (mask == 0)).sum()
        self.train_fp += ((pred == 1) & (mask == 0)).sum()
        self.train_fn += ((pred == 0) & (mask == 1)).sum()

        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):

        self.train_losses.append(self.trainer.callback_metrics["train_loss"].item())

        tp = self.train_tp.item()
        tn = self.train_tn.item()
        fp = self.train_fp.item()
        fn = self.train_fn.item()

        iou = tp / (tp + fp + fn + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)

        metrics = {
            "iou": float(iou),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

        with open(f"train_metrics_{self.current_class}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        self.train_tp = self.train_tn = self.train_fp = self.train_fn = 0

    # ---------------- VALIDATION ----------------
    def validation_step(self, batch, batch_idx):
        loss, pred, mask = self.shared_step(batch)

        pred = pred.view(-1)
        mask = mask.view(-1)

        self.val_tp += ((pred == 1) & (mask == 1)).sum()
        self.val_tn += ((pred == 0) & (mask == 0)).sum()
        self.val_fp += ((pred == 1) & (mask == 0)).sum()
        self.val_fn += ((pred == 0) & (mask == 1)).sum()

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):

        
        self.val_losses.append(self.trainer.callback_metrics["val_loss"].item())

        tp = self.val_tp.item()
        tn = self.val_tn.item()
        fp = self.val_fp.item()
        fn = self.val_fn.item()

        iou = tp / (tp + fp + fn + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)

        metrics = {
            "iou": float(iou),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }

        with open(f"val_metrics_{self.current_class}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        self.val_tp = self.val_tn = self.val_fp = self.val_fn = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": sch}
#Introducir aquí la ruta del Excel que contiene la división en splits de los videos
df = pd.read_excel("")

col_split = [c for c in df.columns if "Train" in c][0]
print("Columna de split detectada:", col_split)

split_dict = {row["Video"]: row[col_split] for _, row in df.iterrows()}
#Introducir el path de las imágenes y máscaras (de todo, luego a traves de la división en splits con el Excel se seleccionan las imágenes que se necesiten)
IMG_ROOT  = ""
MASK_ROOT = ""

def get_video_paths(video):
    img_dir = os.path.join(IMG_ROOT, video)
    msk_dir = os.path.join(MASK_ROOT, video)
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    masks  = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
    return images, masks

train_imgs, train_msks = [], []
val_imgs, val_msks     = [], []

for video, split in split_dict.items():
    imgs, msks = get_video_paths(video)

    if split == 0:
        train_imgs += imgs
        train_msks += msks

    elif split == 1:
        val_imgs += imgs
        val_msks += msks

CLASS_VALUES = {
    "background": 0,
    "Untitled": 20,
    "anastomotic_line": 53,
    "ileal_body": 86,
    "colonic_blind_loop": 119,
    "ileal_inlet": 151,
    "neo-TI": 184,
    "ileal_blind_loop": 217,
    "Colon_proximal_to_anastomosis": 250
}

def entrenar_clase(nombre_clase, valor_clase, arch, encoder, encoder_weights=None, checkpoint_path=None):

   
    print(f" ENTRENANDO CLASE: {nombre_clase}")
    

    print(f"Train imgs: {len(train_imgs)}  |  Val imgs: {len(val_imgs)}")

    train_ds = BinaryDataset(train_imgs, train_msks, valor_clase,
                             augmentation=get_training_augmentation(),
                             preprocessing=preprocessing)

    val_ds = BinaryDataset(val_imgs, val_msks, valor_clase,
                           augmentation=get_validation_augmentation(),
                           preprocessing=preprocessing)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    model = BinarySegModel(arch, encoder, encoder_weights=encoder_weights, checkpoint_path=checkpoint_path)
    model.current_class = nombre_clase

    trainer = pl.Trainer(
        max_epochs=20,
        log_every_n_steps=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, train_loader, val_loader)
    #Graficos de perdidas de train y validación

    # TRAIN LOSS
    plt.figure()
    plt.plot(model.train_losses, label="Train Loss")
    plt.title(f"Train Loss - {nombre_clase}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"loss_train_{nombre_clase}.png")
    plt.close()

    # VAL LOSS
    plt.figure()
    plt.plot(model.val_losses, label="Val Loss", color="orange")
    plt.title(f"Validation Loss - {nombre_clase}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"loss_val_{nombre_clase}.png")
    plt.close()

    print(f"Guardados loss_train_{nombre_clase}.png y loss_val_{nombre_clase}.png")

    # Guardar modelo
    torch.save(model.state_dict(), f"FPN_encoder_freezedmenoslayer3y4_{encoder}_modelo_binario_{nombre_clase}.pth")
    print(f"\nModelo guardado: FPN_encoder_freezedmenoslayer3y4_{encoder}_modelo_binario_{nombre_clase}.pth\n")

#ENTRENAR
if __name__ == "__main__":
    entrenar_clase("anastomotic_line", CLASS_VALUES["anastomotic_line"],arch="",encoder="",encoder_weights="", checkpoint_path=None) 
    #Si se quiere entrenar con pesos que estén en SMP, se pasa el nombre del encoder y los pesos (ejemplo. "ImageNet" o "imagenet"). Si se quiere entrenar con pesos de un checkpoint propio, se pasa la ruta del checkpoint en checkpoint_path y encoder_weights=None. Si no se quieren pesos, se pasa encoder_weights=None y checkpoint_path=None.