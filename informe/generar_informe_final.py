"""
Genera el PDF del informe de la Entrega final.

1. Reúne en `informe/figuras/` las figuras que guardan el notebook 06
   (`docs/resultados/06_*.png`) y las capturas de la aplicación
   (`docs/despliegue/`).
2. Convierte `informe/informe_final.html` a PDF con Edge o Chrome sin ventana.

Uso (desde la raíz del repositorio, después de ejecutar el notebook 06):
    python informe/generar_informe_final.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_INFORME = RAIZ / "informe"
DIR_FIG = DIR_INFORME / "figuras"
HTML = DIR_INFORME / "informe_final.html"
PDF = RAIZ / "entregas" / "03_entrega_final" / "Informe_Entrega_Final.pdf"

COPIAR = {
    "f1_seleccion.png": RAIZ / "docs" / "resultados" / "06_fig1_seleccion.png",
    "f2_roc_costo.png": RAIZ / "docs" / "resultados" / "06_fig2_roc_costo.png",
    "f3_calibracion.png": RAIZ / "docs" / "resultados" / "06_fig3_calibracion.png",
    "f4_shap_global.png": RAIZ / "docs" / "resultados" / "06_fig4_shap_global.png",
    "f5_dependencia.png": RAIZ / "docs" / "resultados" / "06_fig5_dependencia_parcial.png",
    "f6_casos.png": RAIZ / "docs" / "resultados" / "06_fig6_casos_shap_lime.png",
    "f7_app_individual.png": RAIZ / "docs" / "despliegue" / "app_solicitud.png",
    "f8_app_lote.png": RAIZ / "docs" / "despliegue" / "app_lote.png",
}
NAVEGADORES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
]


def reunir_figuras():
    DIR_FIG.mkdir(exist_ok=True)
    for destino, origen in COPIAR.items():
        if not origen.exists():
            sys.exit(f"Falta {origen.relative_to(RAIZ)} (¿se ejecutó el notebook 06?)")
        shutil.copyfile(origen, DIR_FIG / destino)
    print(f"Figuras reunidas en {DIR_FIG.relative_to(RAIZ)}")


def generar_pdf():
    navegador = next((n for n in NAVEGADORES if n.exists()), None)
    if navegador is None:
        sys.exit("No se encontró Edge ni Chrome; abra informe_final.html en un navegador e imprima a PDF.")
    subprocess.run([str(navegador), "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=120)
    print(f"PDF generado: {PDF.relative_to(RAIZ)}")


if __name__ == "__main__":
    reunir_figuras()
    generar_pdf()
