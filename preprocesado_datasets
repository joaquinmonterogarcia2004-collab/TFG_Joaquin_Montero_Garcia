
from glob import glob
from importlib.resources import path
from xml.etree.ElementPath import find
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from collections import Counter
import shutil
import glob
from PIL import Image

dict_classes = {
    "0": "background",
    "20": "Untitled",
    "53": "anastomotic_line",
    "86": "ileal_body",
    "119": "colonic_blind_loop",
    "151": "ileal_inlet",
    "184": "neo-TI",
    "217": "ileal_blind_loop",
    "250": "Colon_proximal_to_anastomosis"
}

colores = {
    0:  (0, 0, 0),        # fondo
    20: (0, 0, 255),      # rojo
    53: (0, 255, 0),      # verde
    86: (255, 0, 0),      # azul
    119: (0, 255, 255),   # amarillo
    151: (255, 0, 255),   # magenta
    184: (255, 255, 0),   # cyan
    217: (128, 128, 128), # gris
    250: (64, 64, 64)     # gris oscuro
}

def colorear_mascara(mask,colores):
    mask_color=np.zeros((*mask.shape,3),dtype=np.uint8)
    for valor, color in colores.items():
        mask_color[mask==valor]=color
    return mask_color

def superponer_mascara(frame_path, mask_path, alpha=0.5):
    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    if frame is None:
        print("Frame no cargado:", frame_path)
        return None
    if mask is None:
        print("Imagen no cargada:", mask_path)
        return None

    mask_color=colorear_mascara(mask,colores)
    superpuesta = cv2.addWeighted(mask_color, alpha, frame, 1 - alpha, 0)
    return superpuesta

def procesar_carpeta(frames_dir, masks_dir, salida_dir):
    masks_dict = {}
    for root, _, files in os.walk(masks_dir):
        for f in files:
            base = os.path.splitext(f)[0]
            masks_dict[base] = os.path.join(root, f)
    for root, _, files in os.walk(frames_dir):
        for f in files:
            base = os.path.splitext(f)[0]
            frame_path = os.path.join(root, f)

            if base not in masks_dict:
                continue

            mask_path = masks_dict[base]
            superpuesta = superponer_mascara(frame_path, mask_path)

            if superpuesta is not None:
                subruta = os.path.relpath(root, frames_dir)
                carpeta_salida = os.path.join(salida_dir, subruta)
                os.makedirs(carpeta_salida, exist_ok=True)

                out_path = os.path.join(carpeta_salida, f"{base}_overlay.png")
                cv2.imwrite(out_path, superpuesta)
                print("Guardado:", out_path)
#EJEMPLO DE USO: procesar_carpeta("Ruta_carpeta_frames", "Ruta_carpeta_masks", "Ruta_carpeta_salida_overlays")

def procesar_frames_y_masks(carpeta_frames, carpeta_masks, carpeta_salida):
    imagenes_sin_mascara = []
    for root, dirs, files in os.walk(carpeta_frames):
        for nombre in files:

            # Ruta completa del frame
            frame_path = os.path.join(root, nombre)

            # Nombre base sin extensión
            base = os.path.splitext(nombre)[0]

            # Nombre esperado de la máscara
            mask_name = base + "_mask.png"
            mask_path = os.path.join(carpeta_masks, mask_name)

            # Crear carpeta equivalente en salida
            relative = os.path.relpath(root, carpeta_frames)
            salida_dir = os.path.join(carpeta_salida, relative)
            os.makedirs(salida_dir, exist_ok=True)

            if os.path.exists(mask_path):
                salida = superponer_mascara(frame_path, mask_path)
                if salida is not None:
                    cv2.imwrite(os.path.join(salida_dir, nombre), salida)
            else:
                print("No hay máscara para:", nombre)
                imagenes_sin_mascara.append(nombre)


def clases_en_mascaras(carpeta_masks):
    clases=[]
    for root, dirs, files in os.walk(carpeta_masks):
        for nombre in files:
            mask_path = os.path.join(root,nombre)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                clases_presentes=np.unique(mask)
                nombres=[dict_classes.get(str(clase)) for clase in clases_presentes]
                clases.append(nombres)
        return clases 
#Introduciendo la ruta de la carpeta de máscaras,se obtiene una lista de listas, donde cada sublista contiene los nombres de las clases presentes en cada máscara. Útil para comprobar resultados

def contador_clases_en_mascaras(carpeta_masks):
    contador_clases={}
    for root, dirs, files in os.walk(carpeta_masks):
        for nombre in files:
            mask_path = os.path.join(root,nombre)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                clases_presentes=np.unique(mask)
                for clase in clases_presentes:
                    nombre_clase=dict_classes.get(str(clase))
                    if nombre_clase is None:
                        continue

                    if nombre_clase in contador_clases:
                        contador_clases[nombre_clase] += 1
                    else:
                        contador_clases[nombre_clase] = 1
    return contador_clases
#Introduciendo la ruta de la carpeta de máscaras, se obtiene un diccionario con el conteo de cada clase presente en todas las máscaras. Útil para obtener estadísticas del dataset.


def contar_clases_por_video(carpeta_principal, dict_classes):
    resultado = {}
    for nombre_video in os.listdir(carpeta_principal):
        ruta_video = os.path.join(carpeta_principal, nombre_video)

        if not os.path.isdir(ruta_video):
            continue  

        contador = Counter()

      
        for archivo in os.listdir(ruta_video):
            if not archivo.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                continue

            mask_path = os.path.join(ruta_video, archivo)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # Obteniene clases presentes en la máscara
            clases = np.unique(mask)

            for c in clases:
                nombre_clase = dict_classes.get(str(c), None)
                if nombre_clase is None:
                    continue  # ignorar valores no etiquetados
                contador[nombre_clase] += 1

        resultado[nombre_video] = dict(contador)

    return resultado
#Esta función realiza lo mismo que la anterior, pero organiza los resultados por vídeo en lugar de dar un conteo global. Devuelve un diccionario donde cada clave es el nombre del vídeo y el valor es otro diccionario con el conteo de clases presentes en las máscaras de ese vídeo.

def imagenes_sin_mascara(carpeta_frames, carpeta_masks):
    imagenes_sin_mascara = []
    masks_set = set()
    for root, dirs, files in os.walk(carpeta_masks):
        for nombre in files:
            base = os.path.splitext(nombre)[0]
            masks_set.add(base)

    for root, dirs, files in os.walk(carpeta_frames):
        for nombre in files:
            base = os.path.splitext(nombre)[0]
            if base not in masks_set:
                imagenes_sin_mascara.append(os.path.join(root, nombre))
    return imagenes_sin_mascara

#Introduciendo las rutas de las carpetas de frames y máscaras, se crea un set con los nombres base de las máscaras y se compara con los nombres base de los frames (por lo que es necesario que se llamen igual y solo cambie el formato), devolviendo una lista con las imágenes que no tienen máscara. 
def borrar_frames_sin_mascara(carpeta_frames, carpeta_masks):
    for root, dirs, files in os.walk(carpeta_frames):
        for nombre in files:
            frame_path = os.path.join(root, nombre)
            base = os.path.splitext(nombre)[0]
            mask_name = base + "_mask.png"
            mask_path = os.path.join(carpeta_masks, mask_name)

            if not os.path.exists(mask_path):
                os.remove(frame_path)
                print(f"Eliminado: {frame_path}")
#Introduciendo las rutas de las carpetas de frames y máscaras, se recorre la carpeta de frames y se elimina cualquier frame que no tenga su correspondiente máscara. Se asume que las máscaras tienen el mismo nombre base que los frames, pero con el sufijo "_mask" y extensión ".png", por lo que es necesario adaptar el código a otra estructura de datos.

def obtener_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ignorar negro y blanco extremo
    mask = (gray > 10) & (gray < 230)

    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        return 0, img.shape[0], 0, img.shape[1]

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return y_min, y_max, x_min, x_max

def recortar_img_y_mask(img, mask, bbox):
    y_min, y_max, x_min, x_max = bbox
    img_rec  = img[y_min:y_max+1,  x_min:x_max+1]
    mask_rec = mask[y_min:y_max+1, x_min:x_max+1]
    return img_rec, mask_rec

def recortar_marco_negro(frame_path, mask_path):
    img  = cv2.imread(frame_path)
    anot = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("No se pudo cargar la imagen:", frame_path)
        return None, None
    if anot is None:
        print("No se pudo cargar la máscara:", mask_path)
        return None, None

    bbox = obtener_bbox(img)
    if bbox is None:
        return img, anot  # imagen completamente negra

    return recortar_img_y_mask(img, anot, bbox)
#EJEMPLO DE USO: recortar_marco_negro("ruta/a/frames", "ruta/a/mascaras"), paraque se lleve a cabo el recorte de las imágenes gracias a la obtención de la bounding_box en base a los niveles de gris que se indiquen en la función obtener_bbox

def separar_por_video(frames_dir, masks_dir, out_frames_dir, out_masks_dir):
    
    os.makedirs(out_frames_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)


    for fname in os.listdir(frames_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if "_" not in fname:
            print(f" Archivo sin '_': {fname}")
            continue

        
        video = fname.rsplit("_", 1)[0]

        src = os.path.join(frames_dir, fname)
        dst_dir = os.path.join(out_frames_dir, video)
        dst = os.path.join(dst_dir, fname)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)

        print(f"[FRAME] {fname} → {video}/")

    for fname in os.listdir(masks_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        if "_" not in fname:
            print(f" Archivo sin '_': {fname}")
            continue
        video = fname.rsplit("_", 1)[0]

        src = os.path.join(masks_dir, fname)
        dst_dir = os.path.join(out_masks_dir, video)
        dst = os.path.join(dst_dir, fname)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)

        print(f"[MASK]  {fname} → {video}/")
#EJEMPLO DE USO: separar_por_video("ruta/a/frames", "ruta/a/mascaras", "ruta/a/salida_frames", "ruta/a/salida_mascaras"), para que se lleve a cabo la separación de los frames y máscaras en carpetas por vídeo, asumiendo que el nombre del archivo contiene un guion bajo "_" que separa el nombre del vídeo del resto del nombre del archivo.

    
















