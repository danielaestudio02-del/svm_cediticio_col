"""
Despliegue del modelo final (Entrega final, 8.3) con la interfaz web.

Streamlit publica, a pantalla completa, la versión web de la aplicación
(web/index.html, diseño de shadcn/ui): las mismas tres pestañas (evaluar una
solicitud, evaluar un lote en CSV y acerca del modelo), con la decisión, la
probabilidad calibrada, la explicación SHAP, el mapa de decisión interactivo y
la verificación de coherencia de los datos.

La página lleva dentro el modelo final exportado por web/construir_web.py, que
antes de escribirla comprueba que la reimplementación reproduce
`decision_function` del pipeline en las 1.300 solicitudes de prueba (diferencia
máxima < 1e-9). Los cálculos se hacen en el navegador de quien usa la
aplicación.

La versión con la interfaz nativa de Streamlit, que ejecuta el modelo en
Python, está en app/app_clasica.py.

Ejecución (desde la raíz del repositorio):
    python web/construir_web.py     # solo si cambia el modelo final
    streamlit run app/app.py
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

RUTA_WEB = Path(__file__).resolve().parents[1] / "web" / "index.html"

st.set_page_config(page_title="Riesgo crediticio · SVM", page_icon="💳", layout="wide",
                   initial_sidebar_state="collapsed")

# La interfaz web ocupa toda la ventana: se ocultan el encabezado y los márgenes de
# Streamlit, y el marco de la página toma el alto de la ventana (la página web se
# desplaza por dentro, así no hay dos barras de desplazamiento).
st.markdown("""
<style>
  header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], footer { display: none !important; }
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #171717 !important; overflow: hidden !important;
  }
  [data-testid="stMainBlockContainer"], .block-container {
    padding: 0 !important; max-width: 100% !important;
  }
  [data-testid="stVerticalBlock"] { gap: 0 !important; }
  [data-testid="stElementContainer"]:has(> .stMarkdown style) { display: none !important; }
  iframe { display: block; width: 100% !important; height: 100vh !important; height: 100dvh !important; border: 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def pagina_web():
    return RUTA_WEB.read_text(encoding="utf-8")


if RUTA_WEB.exists():
    components.html(pagina_web(), height=900, scrolling=True)
else:
    st.error("Falta web/index.html. Genérelo con `python web/construir_web.py` "
             "(o use la versión nativa: `streamlit run app/app_clasica.py`).")
