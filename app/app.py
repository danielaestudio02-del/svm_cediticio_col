"""
Despliegue básico del modelo final (Entrega final, 8.3).

Aplicación de Streamlit que recibe los datos de una solicitud de crédito (o un
lote en CSV), verifica que sean coherentes con los datos de entrenamiento,
aplica el `Pipeline` completo del modelo final (imputación, escalamiento,
one-hot y SVM RBF) y devuelve:
  - la decisión con el umbral de costo mínimo 5:1 elegido en entrenamiento,
  - la probabilidad calibrada de "malo" (Platt),
  - la explicación SHAP de la predicción, a nivel de las 11 variables.

Ejecución (desde la raíz del repositorio):
    python src/modelo_final.py      # solo si models/modelo_final.joblib no existe
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from interpretacion import (COLUMNAS, NOMBRES, ModeloFinal, crear_explicador,  # noqa: E402
                            filas_con_error, tabla_explicacion, verificar_coherencia)

RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
RUTA_EJEMPLO = RAIZ / "app" / "ejemplo_solicitudes.csv"
NO_INFORMADO = "(no informado)"

ETIQUETAS = {
    "genero": {"F": "Femenino", "M": "Masculino"},
    "tipo_ocupacion": {"alta_calificacion": "Alta calificación", "formal_calificado": "Formal calificado",
                       "formal_no_calificado": "Formal no calificado", "informal": "Informal"},
    "tipo_vivienda": {"propia": "Propia", "familiar": "Familiar", "arrendada": "Arrendada"},
    "nivel_ahorro": {"muy_alto": "Muy alto", "alto": "Alto", "medio": "Medio", "bajo": "Bajo"},
    "saldo_categoria": {"alto": "Alto", "medio": "Medio", "bajo": "Bajo", "sin_cuenta": "Sin cuenta"},
    "proposito_credito": {"educacion": "Educación", "vehiculo": "Vehículo", "remodelacion": "Remodelación",
                          "electrodomesticos": "Electrodomésticos", "otros": "Otros",
                          "libre_inversion": "Libre inversión", "negocio": "Negocio"},
    "estado_pago_previo": {"al_dia": "Al día", "sin_creditos_previos": "Sin créditos previos",
                           "mora_leve": "Mora leve", "mora_severa": "Mora severa"},
}

st.set_page_config(page_title="Riesgo crediticio · SVM", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------------
# Carga (una sola vez por sesión del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar():
    if not RUTA_MODELO.exists():
        st.error("No se encontró `models/modelo_final.joblib`. Ejecute primero `python src/modelo_final.py`.")
        st.stop()
    artefacto = joblib.load(RUTA_MODELO)
    modelo = ModeloFinal(artefacto)
    explicar = crear_explicador(modelo, artefacto["fondo_shap"].iloc[:50])
    return artefacto, modelo, explicar


artefacto, modelo, explicar = cargar()
perfil = artefacto["perfil"]
P_UMBRAL = modelo.umbral_probabilidad


def cop(v):
    return f"$ {v:,.0f}".replace(",", ".")


def barra_riesgo(p):
    """Barra horizontal con la probabilidad y el umbral de decisión."""
    color = "#C0392B" if p >= P_UMBRAL else "#1E8449"
    ancho = min(p / 0.8, 1) * 100
    marca = P_UMBRAL / 0.8 * 100
    st.markdown(f"""
<div style="position:relative;height:22px;background:rgba(128,128,128,.18);border-radius:11px;margin:6px 0 4px 0">
  <div style="width:{ancho:.1f}%;height:100%;background:{color};border-radius:11px"></div>
  <div style="position:absolute;left:{marca:.1f}%;top:-4px;height:30px;border-left:2px dashed currentColor"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:.8rem;opacity:.75">
  <span>0%</span><span>umbral de decisión: {P_UMBRAL:.1%}</span><span>80%</span>
</div>""", unsafe_allow_html=True)


def grafico_shap(tabla):
    import matplotlib.pyplot as plt

    t = tabla.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    etiquetas = [f"{v}: {x}" for v, x in zip(t["Variable"], t["Valor"])]
    ax.barh(etiquetas, t["Contribución"], color=["#C0392B" if v > 0 else "#2471A3" for v in t["Contribución"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribución al puntaje de riesgo (SHAP)   ← reduce · aumenta →")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(alpha=0.3, axis="x")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def mostrar_hallazgos(h):
    for _, f in h.iterrows():
        texto = f"**{NOMBRES.get(f['variable'], f['variable'])}**: {f['detalle']}"
        if f["filas"] > 1:
            texto += f" ({f['filas']} filas)"
        {"error": st.error, "advertencia": st.warning, "info": st.info}[f["nivel"]](texto)


def resultado_individual(X):
    hall = verificar_coherencia(X, perfil, deriva=False)
    if (hall["nivel"] == "error").any():
        st.subheader("Los datos no son coherentes")
        mostrar_hallazgos(hall[hall["nivel"] == "error"])
        st.caption("Corrija los datos: el modelo no puntúa solicitudes con errores de captura.")
        return
    p = float(modelo.probabilidad(X)[0])
    rechazar = p >= P_UMBRAL

    col1, col2 = st.columns([1, 1.4], gap="large")
    with col1:
        st.subheader("Decisión")
        if rechazar:
            st.error("### Rechazar o revisar\nRiesgo por encima del umbral de costo mínimo.")
        else:
            st.success("### Aprobar\nRiesgo por debajo del umbral de costo mínimo.")
        st.metric("Probabilidad estimada de incumplimiento", f"{p:.1%}",
                  f"{(p - P_UMBRAL) * 100:+.1f} p.p. frente al umbral", delta_color="inverse")
        barra_riesgo(p)
        st.caption(
            f"Rechazar a un buen cliente cuesta 1 y aprobar a uno malo cuesta 5, así que el modelo "
            f"rechaza desde un riesgo de {P_UMBRAL:.1%}, muy por debajo del 50%. "
            f"Tasa media de incumplimiento en entrenamiento: 22.5%.")
        if not hall.empty:
            with st.expander(f"Observaciones sobre los datos ({len(hall)})"):
                mostrar_hallazgos(hall)

    with col2:
        st.subheader("¿Por qué?")
        with st.spinner("Calculando la explicación (SHAP)…"):
            exp = explicar(X)
        tabla = tabla_explicacion(exp[0], X.iloc[0])
        st.pyplot(grafico_shap(tabla), width="stretch")
        sube = tabla[tabla["Contribución"] > 0].head(3)
        baja = tabla[tabla["Contribución"] < 0].head(3)
        frases = []
        if len(sube):
            frases.append("**Aumentan el riesgo:** " + "; ".join(
                f"{v.lower()} = {x}" for v, x in zip(sube["Variable"], sube["Valor"])) + ".")
        if len(baja):
            frases.append("**Lo reducen:** " + "; ".join(
                f"{v.lower()} = {x}" for v, x in zip(baja["Variable"], baja["Valor"])) + ".")
        st.markdown("  \n".join(frases))
        st.caption("Cada barra es lo que esa variable suma o resta al puntaje del SVM frente a un "
                   "solicitante promedio de entrenamiento (valores SHAP). Las barras suman la diferencia "
                   "entre el puntaje de esta solicitud y el promedio.")


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("Evaluación de riesgo crediticio")
st.caption("SVM con kernel RBF · reproducción y adaptación de Xu y Cheng al contexto colombiano · "
           "**prototipo académico entrenado con datos simulados**: no usar para decisiones reales.")

with st.sidebar:
    st.header("Modelo")
    hp = artefacto["hiperparametros"]
    m = artefacto["metricas_test"]
    n_sv = f"{artefacto['n_vectores_soporte']:,}".replace(",", ".")
    st.markdown(f"""
**{artefacto['nombre']}** · `C = {hp['C']:g}` · `γ = {hp['gamma']:g}`
{n_sv} vectores de soporte

**Desempeño en prueba** (1.300 clientes)
- ROC-AUC: **{m['ROC-AUC']:.3f}**
- Recall de "malo": **{m['Recall (malo)']:.3f}**
- Costo medio 5:1: **{m['Costo medio (5:1)']:.3f}**

**Umbral de decisión:** P(malo) ≥ {P_UMBRAL:.1%}

Entrenado el {artefacto['fecha_entrenamiento']} con scikit-learn {artefacto['versiones']['scikit-learn']}.
""")

pestanas = st.tabs(["Evaluar una solicitud", "Evaluar un lote (CSV)", "Acerca del modelo"])

# --- 1. Solicitud individual ---
with pestanas[0]:
    cats = perfil["categorias"]

    def selector(col, ayuda=None, permitir_faltante=False, indice=0):
        opciones = cats[col] + ([NO_INFORMADO] if permitir_faltante else [])
        v = st.selectbox(NOMBRES[col], opciones, index=indice, help=ayuda,
                         format_func=lambda x: ETIQUETAS.get(col, {}).get(x, x))
        return np.nan if v == NO_INFORMADO else v

    st.markdown("Ingrese los datos de la solicitud. Los campos marcados como *(no informado)* se "
                "imputan con la moda de entrenamiento, igual que en el pipeline.")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Solicitante**")
        edad = st.number_input("Edad", min_value=0, max_value=100, value=35, step=1)
        genero = selector("genero", "No interviene en el riesgo de la simulación; se incluye para "
                                    "verificar que el modelo no lo use.")
        ocupacion = selector("tipo_ocupacion", permitir_faltante=True, indice=3)
        vivienda = selector("tipo_vivienda", indice=2)
    with c2:
        st.markdown("**Situación financiera**")
        saldo_cat = selector("saldo_categoria", indice=2)
        if saldo_cat == "sin_cuenta":
            saldo = 0.0
            st.number_input("Saldo en cuenta (COP)", value=0, disabled=True,
                            help="Sin cuenta: el saldo es cero por definición.")
        else:
            rango = perfil["saldo_por_categoria"].get(saldo_cat, {"min": 0, "max": 1e7})
            sugerido = int(round((rango["min"] * rango["max"]) ** 0.5, -3)) if rango["min"] > 0 else 500_000
            saldo = float(st.number_input("Saldo en cuenta (COP)", min_value=0, value=sugerido, step=50_000,
                                          help=f"En entrenamiento, la categoría va de {cop(rango['min'])} "
                                               f"a {cop(rango['max'])}."))
        ahorro = selector("nivel_ahorro", permitir_faltante=True, indice=3)
        pago = selector("estado_pago_previo", "La variable más influyente del modelo.")
    with c3:
        st.markdown("**Crédito solicitado**")
        monto = float(st.number_input("Monto del crédito (COP)", min_value=0, value=4_000_000, step=250_000))
        plazo = float(st.select_slider("Plazo (meses)", options=list(range(3, 73, 3)), value=18))
        proposito = selector("proposito_credito", indice=5)

    X = pd.DataFrame([{
        "edad": float(edad), "genero": genero, "tipo_ocupacion": ocupacion, "tipo_vivienda": vivienda,
        "nivel_ahorro": ahorro, "saldo_categoria": saldo_cat, "saldo_cuenta_cop": saldo,
        "monto_credito_cop": monto, "plazo_meses": plazo, "proposito_credito": proposito,
        "estado_pago_previo": pago,
    }])[COLUMNAS]

    if st.button("Evaluar solicitud", type="primary"):
        st.divider()
        resultado_individual(X)

# --- 2. Lote ---
with pestanas[1]:
    st.markdown(f"Suba un CSV con las 11 columnas del modelo ({', '.join(f'`{c}`' for c in COLUMNAS)}). "
                "Antes de puntuar se verifica el esquema, las categorías, los rangos, las reglas de negocio "
                "y la deriva frente a entrenamiento (PSI).")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    usar_ejemplo = st.checkbox("Usar el archivo de ejemplo (200 solicitudes de prueba y 4 con errores de captura)",
                               value=archivo is None)
    lote = None
    if archivo is not None:
        lote = pd.read_csv(archivo)
    elif usar_ejemplo and RUTA_EJEMPLO.exists():
        lote = pd.read_csv(RUTA_EJEMPLO)

    if lote is not None:
        hall = verificar_coherencia(lote, perfil)
        n_err = int(hall.loc[hall["nivel"] == "error", "filas"].sum()) if not hall.empty else 0
        st.subheader("Verificación de coherencia con entrenamiento")
        if hall.empty:
            st.success("Sin hallazgos: el lote es coherente con los datos de entrenamiento.")
        else:
            cuenta = hall["nivel"].value_counts()
            etiquetas = {"error": "errores", "advertencia": "advertencias", "info": "observaciones informativas"}
            st.markdown(" · ".join(f"**{cuenta.get(n, 0)}** {etiquetas[n]}" for n in etiquetas))
            st.dataframe(hall.assign(variable=hall["variable"].map(lambda v: NOMBRES.get(v, v))),
                         hide_index=True, width="stretch")

        if any(c not in lote.columns for c in COLUMNAS):
            st.stop()
        # Filas con errores: no se puntúan
        filas_error = filas_con_error(lote, perfil)
        X_lote = lote[COLUMNAS].copy()
        for c in ["edad", "saldo_cuenta_cop", "monto_credito_cop", "plazo_meses"]:
            X_lote[c] = pd.to_numeric(X_lote[c], errors="coerce")
        validas = X_lote[~filas_error]

        if len(validas):
            prob = modelo.probabilidad(validas)
            res = lote.loc[validas.index].copy()
            res["prob_malo"] = prob.round(4)
            res["decision"] = np.where(prob >= P_UMBRAL, "rechazar / revisar", "aprobar")
            st.subheader("Resultados")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Solicitudes puntuadas", f"{len(res)}")
            k2.metric("Excluidas por errores", f"{int(filas_error.sum())}")
            k3.metric("Aprobadas", f"{(res['decision'] == 'aprobar').mean():.1%}")
            k4.metric("P(malo) media", f"{prob.mean():.1%}")
            if "riesgo" in res.columns:
                real = (res["riesgo"] == "malo").astype(int).to_numpy()
                pred = (prob >= P_UMBRAL).astype(int)
                st.caption(f"El archivo trae la etiqueta real: recall de 'malo' = {pred[real == 1].mean():.3f}, "
                           f"costo medio 5:1 = {(5 * ((real == 1) & (pred == 0)).sum() + ((real == 0) & (pred == 1)).sum()) / len(real):.3f}.")
            st.dataframe(res, hide_index=True, width="stretch",
                         column_config={"prob_malo": st.column_config.ProgressColumn(
                             "P(malo)", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8"),
                               "resultados_riesgo.csv", "text/csv")

            st.subheader("Explicar una solicitud del lote")
            pos = st.number_input("Fila (posición en la tabla de resultados)", min_value=1,
                                  max_value=len(res), value=1, step=1)
            if st.button("Explicar"):
                resultado_individual(validas.iloc[[int(pos) - 1]].reset_index(drop=True))

# --- 3. Acerca del modelo ---
with pestanas[2]:
    st.markdown(f"""
#### Qué hace
Clasifica solicitudes de crédito de consumo en riesgo **bueno** o **malo** con una máquina de vectores de
soporte (kernel gaussiano), el método de Xu y Cheng (*Credit Scoring Models Enhancement Using Support Vector
Machines*), adaptado a una base simulada del contexto colombiano.

#### Flujo de cada predicción
1. **Verificación** de los datos frente al perfil de entrenamiento (esquema, categorías, rangos, reglas).
2. **Pipeline** ajustado solo con entrenamiento: imputación del saldo por categoría y del resto por mediana
   o moda, escalamiento, codificación one-hot.
3. **SVM RBF** (`C = {hp['C']:g}`, `γ = {hp['gamma']:g}`, pesos de clase balanceados): puntaje de riesgo.
4. **Calibración de Platt**: convierte el puntaje en probabilidad sin cambiar el orden de los clientes.
5. **Decisión** con el umbral que minimiza el costo 5:1 (elegido por validación cruzada en entrenamiento).
6. **Explicación SHAP** a nivel de las 11 variables del formulario.

#### Límites de uso
- Entrenado con **datos simulados**: mide qué tan bien se recupera una estructura de riesgo diseñada, no el
  comportamiento de clientes reales. No debe usarse para decisiones reales.
- El umbral depende de la matriz de costos 5:1; con otra relación de costos cambia la política.
- Con 22.5% de malos, rechaza a cerca de la mitad de los buenos clientes: es la consecuencia de la matriz de
  costos, y tiene implicaciones de inclusión financiera (sobre todo para no bancarizados y personas con mora
  previa) que una entidad debe decidir explícitamente.
- El género está en los datos, pero el modelo no lo usa (contribución SHAP ≈ 0). En un despliegue real
  convendría retirarlo y monitorear la equidad por grupo.
""")
