import cv2
import os
import csv
import time
from rapidfuzz.distance import Levenshtein
import easyocr
import re
import numpy as np

# === CONFIGURACIÓN ===
DATASET_FOLDER = "dataset_placas"
GT_FILE = os.path.join(DATASET_FOLDER, "ground_truth.txt")
REPORT_FILE = "reporte_final_easyocr.txt"

CHAR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

print("Cargando EasyOCR (CPU)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def recortar_cabecera(img):
    """
    Recorta la parte superior donde dice ECUADOR.
    Normalmente ocupa entre 20% y 35% de la altura total.
    """
    h = img.shape[0]
    corte = int(h * 0.28)   # puedes ajustar 0.28–0.38 según las imágenes
    return img[corte:h, :]
# ======================
# PREPROCESAMIENTO
# ======================
def preprocesar(img):
    """
    Preprocesa la imagen para mejorar el OCR.
    """
    recortar_cabecera(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Suavizado ligero para reducir ruido
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Umbral adaptativo
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 61, 10)
    return thresh


# ======================
# FUNCIÓN OCR
# ======================
def ocr_easy(img):
    # 1. Recortar la franja superior con “ECUADOR”
    img_rec = recortar_cabecera(img)

    # 2. Preprocesar
    img_proc = preprocesar(img_rec)

    # 3. EasyOCR
    resultados = reader.readtext(img_proc, detail=1, paragraph=False)

    if len(resultados) == 0:
        return ""

    textos = [r[1] for r in resultados]
    texto = "".join(textos).upper()

    # Limpiar caracteres no permitidos
    texto = re.sub(f"[^{CHAR_ALLOWLIST}]", "", texto)

    return texto


# ======================
# GENERAR REPORTE
# ======================
def calcular_y_guardar_reporte(resultados, tiempo_total):
    total_imgs = len(resultados)

    tp = sum(1 for r in resultados if r["pred"] != "")
    fn = total_imgs - tp

    exactas = sum(1 for r in resultados if r["pred"] == r["real"])
    total_dist = sum(r["levenshtein"] for r in resultados)
    total_chars = sum(len(r["real"]) for r in resultados)

    precision = 1.0
    recall = tp / total_imgs
    f1 = 2 * precision * recall / (precision + recall) if recall > 0 else 0

    par = exactas / total_imgs
    cer = total_dist / total_chars
    car = 1 - cer
    wer = (total_imgs - exactas) / total_imgs
    avg_lev = total_dist / total_imgs

    fps = total_imgs / tiempo_total
    ms = (tiempo_total / total_imgs) * 1000

    lines = [
        "=" * 50,
        "REPORTE DE EVALUACIÓN: OPENCV + EASYOCR (CPU)",
        "=" * 50,
        f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Imágenes: {total_imgs}",
        "-" * 50,
        "1. MÉTRICAS DE DETECCIÓN",
        f"   Precision (P):    {precision:.4f}",
        f"   Recall (R):       {recall:.4f}",
        f"   F1-Score:         {f1:.4f}",
        "-" * 50,
        "2. MÉTRICAS DE OCR (RECONOCIMIENTO)",
        f"   PAR (Plate Accuracy):     {par*100:.2f}%",
        f"   CAR (Character Accuracy): {car*100:.2f}%",
        f"   CER (Character Error):    {cer*100:.2f}%",
        f"   WER (Word Error Rate):    {wer*100:.2f}%",
        f"   Levenshtein Promedio:     {avg_lev:.2f}",
        "-" * 50,
        "3. RENDIMIENTO",
        f"   Tiempo Total:     {tiempo_total:.2f} seg",
        f"   FPS:              {fps:.2f}",
        f"   Latencia:         {ms:.2f} ms/img",
        "=" * 50,
    ]

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[INFO] Reporte guardado en: {REPORT_FILE}")


# ======================
# MAIN
# ======================
def main():
    if not os.path.exists(GT_FILE):
        print(f"No se encuentra {GT_FILE}")
        return

    print("--- Iniciando Evaluación ---")

    # Leer ground truth
    datos = []
    with open(GT_FILE, "r") as f:
        reader_csv = csv.reader(f)
        next(reader_csv)
        for fila in reader_csv:
            datos.append(fila)

    resultados = []
    start = time.time()

    for i, (nombre_archivo, texto_real) in enumerate(datos):
        ruta = os.path.join(DATASET_FOLDER, nombre_archivo)
        
        img = cv2.imread(ruta)
        if img is None:
            print(f"[ERROR] No se puede leer {ruta}")
            continue

        texto_pred = ocr_easy(img)
        dist = Levenshtein.distance(texto_real, texto_pred)

        resultados.append({
            "real": texto_real,
            "pred": texto_pred,
            "levenshtein": dist
        })

        print(f"Procesando {i}/{len(datos)}...", end="\r")
        print(f"Real: {texto_real} | Pred: {texto_pred} | Dist: {dist}      ", end="\r")

    end = time.time()
    calcular_y_guardar_reporte(resultados, end - start)


if __name__ == "__main__":
    main()
