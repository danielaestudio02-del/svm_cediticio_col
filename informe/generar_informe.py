"""
Genera el PDF del informe de la Segunda entrega.

1. Reúne las figuras que usa el informe en `informe/figuras/`: las que los
   scripts y notebooks ya guardan en `docs/`, y las del notebook 04, que se
   extraen de sus salidas (buscando el título de la figura en el código). A las
   que traen un título general de matplotlib se les recorta, porque el informe
   las numera y describe en su propio pie de figura.
2. Convierte `informe/informe.html` a PDF con Microsoft Edge (o Chrome) en
   modo sin ventana.

Uso (desde la raíz del repositorio, después de ejecutar los notebooks):
    python informe/generar_informe.py
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DIR_INFORME = RAIZ / "informe"
DIR_FIG = DIR_INFORME / "figuras"
HTML = DIR_INFORME / "informe.html"
PDF = DIR_INFORME / "Informe_Segunda_Entrega.pdf"

# Figuras ya guardadas en docs/ por los scripts y notebooks
COPIAR = {
    "fig1_distribuciones.png": RAIZ / "docs" / "validacion" / "fig_distribuciones.png",
    "fig2_relaciones_riesgo.png": RAIZ / "docs" / "validacion" / "fig_relaciones_riesgo.png",
    "fig5_metricas_por_fold.png": RAIZ / "docs" / "resultados" / "05_fig1_metricas_por_fold.png",
    "fig6_matriz_mejor.png": RAIZ / "docs" / "resultados" / "05_fig3_matriz_mejor.png",
    "fig7_importancia.png": RAIZ / "docs" / "resultados" / "05_fig5_importancia_mejor.png",
}
# Figuras del notebook 04: (archivo de salida, texto que identifica la celda)
EXTRAER_NB04 = {
    "fig3_kernels.png": "Figura 2. ROC-AUC por kernel",
    "fig4_curvas_validacion.png": "Figura 3. Curvas de validación",
}
# Figuras con título de matplotlib que se recorta: {archivo: líneas de texto del título}
RECORTAR_TITULO = {"fig1_distribuciones.png": 1, "fig3_kernels.png": 1, "fig4_curvas_validacion.png": 1,
                   "fig5_metricas_por_fold.png": 1, "fig6_matriz_mejor.png": 1, "fig7_importancia.png": 2}
NAVEGADORES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def recortar_titulo(ruta, lineas):
    """Quita las primeras `lineas` franjas de texto (el título) de la imagen."""
    img = Image.open(ruta).convert("RGB")
    con_tinta = (np.asarray(img.convert("L")) < 200).any(axis=1)
    fila = 0
    for _ in range(lineas):
        fila += np.argmax(con_tinta[fila:])     # inicio de la línea de texto
        while con_tinta[fila]:                  # fin de la línea de texto
            fila += 1
    inicio_figura = fila + np.argmax(con_tinta[fila:])
    img.crop((0, max(inicio_figura - 3, 0), img.width, img.height)).save(ruta)


def reunir_figuras():
    DIR_FIG.mkdir(exist_ok=True)
    for destino, origen in COPIAR.items():
        shutil.copyfile(origen, DIR_FIG / destino)
    nb = json.loads((RAIZ / "notebooks" / "04_SVM_reproduccion_metodo.ipynb").read_text(encoding="utf-8"))
    for destino, marca in EXTRAER_NB04.items():
        celdas = [c for c in nb["cells"] if c["cell_type"] == "code" and marca in "".join(c["source"])]
        imagenes = [o["data"]["image/png"] for c in celdas for o in c.get("outputs", [])
                    if "image/png" in o.get("data", {})]
        if len(imagenes) != 1:
            sys.exit(f"No se encontró una única figura para '{marca}' (¿se ejecutó el notebook 04?)")
        (DIR_FIG / destino).write_bytes(base64.b64decode(imagenes[0]))
    for nombre, lineas in RECORTAR_TITULO.items():
        recortar_titulo(DIR_FIG / nombre, lineas)
    print(f"Figuras reunidas en {DIR_FIG.relative_to(RAIZ)}")


def generar_pdf():
    navegador = next((n for n in NAVEGADORES if n.exists()), None)
    if navegador is None:
        sys.exit("No se encontró Edge ni Chrome; abra informe.html en un navegador e imprima a PDF.")
    subprocess.run([str(navegador), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=120)
    print(f"PDF generado: {PDF.relative_to(RAIZ)}")


if __name__ == "__main__":
    reunir_figuras()
    generar_pdf()
