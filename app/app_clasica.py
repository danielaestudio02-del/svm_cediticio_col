"""
Despliegue básico del modelo final (Entrega final, 8.3): versión con la interfaz
nativa de Streamlit, que ejecuta el modelo en Python. Es la aplicación descrita en
el informe final; la que se publica es app/app.py, con la interfaz web.

Aplicación de Streamlit que recibe los datos de una solicitud de crédito (o un
lote en CSV), verifica que sean coherentes con los datos de entrenamiento,
aplica el `Pipeline` completo del modelo final (imputación, escalamiento,
one-hot y SVM RBF) y devuelve:
  - la decisión con el umbral de costo mínimo 5:1 elegido en entrenamiento,
  - la probabilidad calibrada de "malo" (Platt),
  - la explicación SHAP de la predicción, a nivel de las 11 variables,
  - el mapa de decisión: la frontera del SVM en el plano monto × plazo y el cambio
    de monto o de plazo que cambiaría la decisión.

Ejecución (desde la raíz del repositorio):
    python src/modelo_final.py      # solo si models/modelo_final.joblib no existe
    streamlit run app/app_clasica.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from interpretacion import (COLUMNAS, NOMBRES, CorteMontoPlazo, ModeloFinal,  # noqa: E402
                            crear_explicador, filas_con_error, tabla_explicacion, verificar_coherencia)

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
    """Carga el modelo guardado; si falta o no abre con las librerías instaladas
    (p. ej. otra versión de NumPy en el servidor), lo reentrena con semilla 42
    (1 minuto con 8 núcleos; varios minutos en servidores con pocos núcleos)."""
    try:
        artefacto = joblib.load(RUTA_MODELO)
    except Exception:
        import modelo_final
        with st.spinner("Preparando el modelo por primera vez (puede tardar unos minutos)…"):
            artefacto = modelo_final.entrenar()
    modelo = ModeloFinal(artefacto)
    explicar = crear_explicador(modelo, artefacto["fondo_shap"].iloc[:50])
    return artefacto, modelo, explicar


artefacto, modelo, explicar = cargar()
perfil = artefacto["perfil"]
P_UMBRAL = modelo.umbral_probabilidad


def cop(v):
    return f"$ {v:,.0f}".replace(",", ".")


COLUMNA_POR_NOMBRE = {v: k for k, v in NOMBRES.items()}


def con_etiquetas(tabla, fila):
    """Muestra las categorías con su etiqueta en español (con tildes) en la explicación."""
    tabla = tabla.copy()
    tabla["Valor"] = [ETIQUETAS.get(COLUMNA_POR_NOMBRE[n], {}).get(fila[COLUMNA_POR_NOMBRE[n]], v)
                      for n, v in zip(tabla["Variable"], tabla["Valor"])]
    return tabla


def en_minuscula(texto):
    """Primera letra en minúscula, sin tocar siglas como (COP)."""
    return texto[:1].lower() + texto[1:]


def leer_csv(archivo):
    """Lee un CSV detectando el separador (coma o punto y coma, como lo exporta Excel en
    español) y la codificación (UTF-8 o Latin-1). Devuelve (DataFrame, None) o (None, error)."""
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            archivo.seek(0)
            return pd.read_csv(archivo, sep=None, engine="python", encoding=codificacion), None
        except UnicodeDecodeError:
            continue
        except Exception as e:  # archivo vacío, filas de distinto largo, etc.
            return None, f"No se pudo leer el archivo como CSV ({type(e).__name__}: {e})."
    return None, "No se pudo leer el archivo: la codificación no es UTF-8 ni Latin-1."


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


def corto(v):
    """Monto abreviado para el gráfico: $500 mil, $1 M, $2,5 M."""
    if v < 1e6:
        return f"${v / 1e3:.0f} mil"
    return "$" + f"{v / 1e6:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " M"


def grafico_mapa(corte, caminos, p):
    """Mapa de decisión: P(malo) en el plano monto × plazo (las otras 9 variables fijas), la
    frontera del SVM en el umbral y los caminos hacia la otra decisión."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    montos, plazos, S = corte.malla()
    logit = modelo.a * S + modelo.b                 # logit de P(malo) (calibración de Platt)
    lu = modelo.a * modelo.umbral + modelo.b        # el umbral, en la misma escala
    colores = LinearSegmentedColormap.from_list(
        "decision", ["#1E8449", "#82E0AA", "#F8F9F9", "#F1948A", "#C0392B"])
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    calor = ax.pcolormesh(montos, plazos, np.clip(logit, lu - 2, lu + 2), cmap=colores,
                          norm=TwoSlopeNorm(vmin=lu - 2, vcenter=lu, vmax=lu + 2),
                          shading="gouraud", rasterized=True)
    ax.contour(montos, plazos, S, levels=[modelo.umbral], colors="#17202A", linewidths=2.4,
               linestyles="solid")  # el umbral es negativo y matplotlib punteaba los niveles negativos
    ax.set_xscale("log")
    marcas = [t for t in [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7] if corte.monto_min <= t <= corte.monto_max]
    ax.set_xticks(marcas, [corto(t) for t in marcas])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([3, 12, 24, 36, 48, 60, 72])
    ax.set_xlim(corte.monto_min, corte.monto_max)
    ax.set_ylim(corte.plazo_min, corte.plazo_max)
    ax.set_xlabel("Monto del crédito (COP, escala logarítmica)")
    ax.set_ylabel("Plazo (meses)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    # Caminos hacia la otra decisión (verde: para aprobar; rojo: margen antes del rechazo)
    m0 = float(np.clip(corte.monto, corte.monto_min, corte.monto_max))
    p0 = float(np.clip(corte.plazo, corte.plazo_min, corte.plazo_max))
    tono = "#145A32" if not caminos["aprobado"] else "#922B21"
    flecha = dict(arrowstyle="-|>", color=tono, lw=1.8, linestyle="--", shrinkA=9, shrinkB=6)
    caja = dict(boxstyle="round,pad=0.3", fc="white", ec=tono, lw=1)
    destinos = []
    if caminos["monto"] is not None and caminos["monto"] != corte.monto:
        destinos.append(((caminos["monto"], p0), corto(caminos["monto"]), (0, 10), "center", "bottom"))
    if caminos["plazo"] is not None and caminos["plazo"] != corte.plazo:
        destinos.append(((m0, caminos["plazo"]), f"{caminos['plazo']:.0f} meses", (10, 0), "left", "center"))
    for xy, texto, desplazamiento, ha, va in destinos:
        ax.annotate("", xy=xy, xytext=(m0, p0), arrowprops=flecha, zorder=4)
        ax.scatter(*xy, s=60, facecolor="white", edgecolor=tono, linewidth=2, zorder=5)
        ax.annotate(texto, xy, xytext=desplazamiento, textcoords="offset points", ha=ha, va=va,
                    fontsize=9, fontweight="bold", color=tono, bbox=caja, zorder=7)

    # La solicitud, con la etiqueta del lado contrario a los caminos
    fx = np.log(m0 / corte.monto_min) / np.log(corte.monto_max / corte.monto_min)
    fy = (p0 - corte.plazo_min) / (corte.plazo_max - corte.plazo_min)
    derecha = fx < 0.65 if not caminos["aprobado"] else fx < 0.3
    arriba = fy < 0.8 if not caminos["aprobado"] else fy < 0.2
    ax.scatter([m0], [p0], s=180, color="white", edgecolor="#17202A", linewidth=2.6, zorder=6)
    ax.annotate(f"Esta solicitud · {p:.1%}", (m0, p0), xytext=(12 if derecha else -12, 12 if arriba else -12),
                textcoords="offset points", ha="left" if derecha else "right", va="bottom" if arriba else "top",
                fontsize=9, fontweight="bold", color="#17202A", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#17202A", lw=0.8))

    barra = fig.colorbar(calor, ax=ax, pad=0.02, fraction=0.045)
    niveles = [q for q in [0.03, 0.05, P_UMBRAL, 0.3, 0.5] if lu - 2 <= np.log(q / (1 - q)) <= lu + 2]
    barra.set_ticks([np.log(q / (1 - q)) for q in niveles],
                    labels=[f"{q:.1%}" if q == P_UMBRAL else f"{q:.0%}" for q in niveles])
    barra.set_label("P(malo)")
    fig.tight_layout()
    return fig


def texto_caminos(corte, caminos):
    """Lo que cambiaría la decisión, moviendo solo el monto o solo el plazo."""
    if not caminos["aprobado"]:
        lineas = ["**Para pasar a Aprobar**, cambiando una sola variable:",
                  f"- con un monto de hasta **{cop(caminos['monto'])}** y el mismo plazo;" if caminos["monto"] is not None
                  else f"- ni con el monto mínimo de entrenamiento ({cop(corte.monto_min)}) pasaría;",
                  f"- con un plazo de **{caminos['plazo']:.0f} meses** y el mismo monto." if caminos["plazo"] is not None
                  else "- ni con el plazo mínimo (3 meses) pasaría."]
        if caminos["monto"] is None and caminos["plazo"] is None:
            lineas.append("\nEl riesgo viene de las otras variables (vea **¿Por qué?**).")
    else:
        lineas = ["**Margen antes del rechazo**, cambiando una sola variable:",
                  f"- seguiría aprobada con un monto de hasta **{cop(caminos['monto'])}**;" if caminos["monto"] is not None
                  else f"- seguiría aprobada con cualquier monto del rango de entrenamiento (hasta {cop(corte.monto_max)});"]
        if caminos["plazo"] is None:
            lineas.append("- seguiría aprobada con cualquier plazo, hasta 72 meses.")
        elif caminos["plazo"] > corte.plazo:
            lineas.append(f"- seguiría aprobada con un plazo de hasta **{caminos['plazo']:.0f} meses**.")
        else:
            lineas.append("- con un plazo más largo pasaría a rechazo.")
    return "\n".join(lineas)


def mostrar_figura(fig):
    import matplotlib.pyplot as plt

    st.pyplot(fig, width="stretch")
    plt.close(fig)  # libera la memoria: cada explicación crea una figura nueva


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
        tabla = con_etiquetas(tabla_explicacion(exp[0], X.iloc[0]), X.iloc[0])
        mostrar_figura(grafico_shap(tabla))
        sube = tabla[tabla["Contribución"] > 0].head(3)
        baja = tabla[tabla["Contribución"] < 0].head(3)
        frases = []
        if len(sube):
            frases.append("**Aumentan el riesgo:** " + "; ".join(
                f"{en_minuscula(v)} = {x}" for v, x in zip(sube["Variable"], sube["Valor"])) + ".")
        if len(baja):
            frases.append("**Lo reducen:** " + "; ".join(
                f"{en_minuscula(v)} = {x}" for v, x in zip(baja["Variable"], baja["Valor"])) + ".")
        st.markdown("  \n".join(frases))
        st.caption("Cada barra es lo que esa variable suma o resta al puntaje del SVM frente a un "
                   "solicitante promedio de entrenamiento (valores SHAP). Las barras suman la diferencia "
                   "entre el puntaje de esta solicitud y el promedio.")

    st.subheader("Mapa de decisión")
    st.caption("Cómo cambiaría la decisión del SVM con otro monto o plazo, con las demás variables fijas "
               "en los valores de esta solicitud.")
    corte = CorteMontoPlazo(modelo, X, perfil)
    caminos = corte.caminos()
    c1, c2 = st.columns([1.7, 1], gap="large")
    with c1:
        mostrar_figura(grafico_mapa(corte, caminos, p))
    with c2:
        st.markdown(f"**Esta solicitud:** monto {cop(corte.monto)} · plazo {corte.plazo:.0f} meses · "
                    f"P(malo) {p:.1%}")
        st.markdown(texto_caminos(corte, caminos))
        notas = []
        if pd.isna(pd.to_numeric(X["monto_credito_cop"].iloc[0], errors="coerce")) or \
                pd.isna(pd.to_numeric(X["plazo_meses"].iloc[0], errors="coerce")):
            notas.append("El monto o el plazo no se informó: el mapa usa la mediana de entrenamiento, "
                         "igual que el pipeline.")
        if not (corte.monto_min <= corte.monto <= corte.monto_max):
            notas.append("El monto está fuera del rango de entrenamiento: el punto se dibuja en el borde.")
        st.caption(" ".join(notas + [
            f"Verde: se aprueba; rojo: se rechaza. La línea oscura es la frontera del SVM, donde "
            f"P(malo) = {P_UMBRAL:.1%}. Es un corte del modelo: las otras 9 variables quedan fijas en los "
            "valores de esta solicitud, y cada camino mueve una sola variable."]))


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
        lote, error_lectura = leer_csv(archivo)
        if error_lectura:
            st.error(error_lectura)
    elif usar_ejemplo and RUTA_EJEMPLO.exists():
        lote = pd.read_csv(RUTA_EJEMPLO)

    if lote is not None:
        hall = verificar_coherencia(lote, perfil)
        st.subheader("Verificación de coherencia con entrenamiento")
        if hall.empty:
            st.success("Sin hallazgos: el lote es coherente con los datos de entrenamiento.")
        else:
            cuenta = hall["nivel"].value_counts()
            etiquetas = {"error": "errores", "advertencia": "advertencias", "info": "observaciones informativas"}
            st.markdown(" · ".join(f"**{cuenta.get(n, 0)}** {etiquetas[n]}" for n in etiquetas))
            st.dataframe(hall.assign(variable=hall["variable"].map(lambda v: NOMBRES.get(v, v))),
                         hide_index=True, width="stretch")

    if lote is not None and any(c not in lote.columns for c in COLUMNAS):
        # Sin st.stop(): detendría todo el script y dejaría en blanco la pestaña "Acerca del modelo"
        st.error("El archivo no tiene todas las columnas que necesita el modelo; no se puntúa.")
    elif lote is not None:
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
                # Solo filas con etiqueta conocida; el recall exige al menos un "malo"
                con_etiqueta = res["riesgo"].isin(["bueno", "malo"]).to_numpy()
                real = (res["riesgo"] == "malo").to_numpy()[con_etiqueta]
                pred = (prob >= P_UMBRAL)[con_etiqueta]
                if real.any():
                    costo = (5 * (real & ~pred).sum() + (~real & pred).sum()) / len(real)
                    st.caption(f"El archivo trae la etiqueta real ({len(real)} filas): recall de 'malo' = "
                               f"{pred[real].mean():.3f}, costo medio 5:1 = {costo:.3f}.")
            st.dataframe(res, hide_index=True, width="stretch",
                         column_config={"prob_malo": st.column_config.ProgressColumn(
                             "P(malo)", min_value=0.0, max_value=1.0, format="%.3f")})
            # utf-8-sig: Excel en español abre el archivo con las tildes correctas
            st.download_button("Descargar resultados (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                               "resultados_riesgo.csv", "text/csv")

            st.subheader("Explicar una solicitud del lote")
            pos = st.number_input("Fila (posición en la tabla de resultados)", min_value=1,
                                  max_value=len(res), value=1, step=1)
            if st.button("Explicar"):
                resultado_individual(validas.iloc[[int(pos) - 1]].reset_index(drop=True))
        else:
            st.warning("Ninguna fila del archivo se puede puntuar: todas tienen errores de captura.")

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
