"""
Construye la versión web (HTML autocontenido) de la aplicación de riesgo crediticio.

La página reproduce la aplicación de Streamlit (`app/app.py`) sin servidor: el
modelo final se exporta a JSON y se evalúa en el navegador con JavaScript
(`web/src/motor.js`), con los mismos pasos del `Pipeline`:

  1. imputación del saldo con la mediana de su categoría y del resto con la
     mediana o la moda de entrenamiento;
  2. escalamiento de las numéricas y one-hot de las categóricas;
  3. SVM con kernel RBF (los 3.645 vectores de soporte, sus coeficientes y el
     intercepto), calibración de Platt y umbral de costo mínimo 5:1.

Cada vector de soporte tiene exactamente una categoría activa por variable
(se verifica abajo), así que se guarda como 7 índices de categoría y 4 valores
numéricos escalados, en lugar de 32 columnas: el resultado es idéntico y el
archivo, mucho más liviano.

Uso (desde la raíz del repositorio, con el modelo ya entrenado):
    python web/construir_web.py
Genera `web/index.html`, que se abre directamente en el navegador o se publica
en cualquier hosting estático (GitHub Pages, Netlify...).
"""
import ast
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from interpretacion import COLUMNAS, NOMBRES  # noqa: E402
from preprocesamiento import CAT_COLS, NUM_COLS  # noqa: E402

DIR_WEB = Path(__file__).resolve().parent
DIR_SRC = DIR_WEB / "src"
RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
RUTA_EJEMPLO = RAIZ / "app" / "ejemplo_solicitudes.csv"
RUTA_APP = RAIZ / "app" / "app.py"
SALIDA = DIR_WEB / "index.html"
TAM_FONDO_SHAP = 50   # el mismo fondo que usa la aplicación de Streamlit
URL_REPOSITORIO = "https://github.com/danielaestudio02-del/svm_cediticio_col"


def etiquetas_de_la_app():
    """Lee el diccionario ETIQUETAS de app/app.py (una sola fuente de verdad)."""
    arbol = ast.parse(RUTA_APP.read_text(encoding="utf-8"))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(getattr(t, "id", None) == "ETIQUETAS" for t in nodo.targets):
            return ast.literal_eval(nodo.value)
    raise RuntimeError("No se encontró ETIQUETAS en app/app.py")


def a_json(valor):
    """Convierte tipos de NumPy y NaN a tipos de JSON (NaN -> null)."""
    if isinstance(valor, dict):
        return {str(k): a_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [a_json(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return None if math.isnan(valor) else float(valor)
    return valor


def exportar_modelo():
    a = joblib.load(RUTA_MODELO)
    p = a["pipeline"]
    imp_saldo = p.named_steps["imputar_saldo"]
    ct = p.named_steps["imputar_escalar"]
    num = ct.named_transformers_["num"]
    cat = ct.named_transformers_["cat"]
    ohe = p.named_steps["codificar"].named_transformers_["cat"]
    svc = p.named_steps["modelo"]

    # Orden de columnas que asume la exportación (se verifica, no se supone)
    assert list(ct.get_feature_names_out()) == NUM_COLS + CAT_COLS
    salida = list(p.named_steps["codificar"].get_feature_names_out())
    categorias = [list(map(str, c)) for c in ohe.categories_]
    esperado = [f"{c}_{v}" for c, cats in zip(CAT_COLS, categorias) for v in cats] + NUM_COLS
    assert salida == esperado, "El orden de las columnas del pipeline cambió"
    assert svc.kernel == "rbf" and list(svc.classes_) == [0, 1]

    # Vectores de soporte: un único 1 por bloque one-hot -> índice de categoría
    sv = svc.support_vectors_
    n_onehot = sum(len(c) for c in categorias)
    indices, inicio = [], 0
    for cats in categorias:
        bloque = sv[:, inicio:inicio + len(cats)]
        assert set(np.unique(bloque)) <= {0.0, 1.0} and ((bloque == 1).sum(axis=1) == 1).all()
        indices.append(bloque.argmax(axis=1))
        inicio += len(cats)
    sv_cat = np.stack(indices, axis=1)                  # (n_sv, 7)
    sv_num = sv[:, n_onehot:]                           # (n_sv, 4), ya escaladas

    a_, b_ = a["calibrador"]
    umbral_prob = 1 / (1 + math.exp(-(a_ * a["umbral"] + b_)))

    modelo = {
        "meta": {
            "nombre": a["nombre"],
            "hiperparametros": a["hiperparametros"],
            "n_vectores_soporte": a["n_vectores_soporte"],
            "metricas_test": a["metricas_test"],
            "umbral": a["umbral"],
            "umbral_probabilidad": umbral_prob,
            "platt": {"a": a_, "b": b_},
            "costos": a["costos"],
            "fecha_entrenamiento": a["fecha_entrenamiento"],
            "versiones": a["versiones"],
            "repositorio": URL_REPOSITORIO,
        },
        "columnas": COLUMNAS,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "nombres": NOMBRES,
        "etiquetas": etiquetas_de_la_app(),
        "preprocesamiento": {
            "medianas_saldo": imp_saldo.medianas_.to_dict(),
            "mediana_num": num.named_steps["imputer"].statistics_,
            "media": num.named_steps["scaler"].mean_,
            "escala": num.named_steps["scaler"].scale_,
            "moda_cat": [str(v) for v in cat.statistics_],
            "categorias": categorias,
        },
        "svm": {
            "gamma": float(svc._gamma),
            "intercepto": float(svc.intercept_[0]),
            "alfa": svc.dual_coef_[0],
            "sv_num": sv_num.ravel(),
            # 7 dígitos por vector de soporte (cada índice de categoría es < 10)
            "sv_cat": "".join("".join(str(int(v)) for v in fila) for fila in sv_cat),
        },
        "perfil": a["perfil"],
        "fondo": a["fondo_shap"].iloc[:TAM_FONDO_SHAP][COLUMNAS].astype(object)
                  .where(a["fondo_shap"].iloc[:TAM_FONDO_SHAP][COLUMNAS].notna(), None).values.tolist(),
        "ejemplo_csv": RUTA_EJEMPLO.read_text(encoding="utf-8"),
    }
    assert max(len(c) for c in categorias) <= 10
    return a_json(modelo), a


def verificar_exportacion(modelo, artefacto):
    """Reimplementa el cálculo con los parámetros exportados y lo compara con el pipeline."""
    from preprocesamiento import cargar_particion

    _, X_test, _, _ = cargar_particion()
    pre, svm = modelo["preprocesamiento"], modelo["svm"]
    n_sv = len(svm["alfa"])
    sv_num = np.array(svm["sv_num"]).reshape(n_sv, 4)
    sv_cat = np.array(list(svm["sv_cat"]), dtype=int).reshape(n_sv, 7)
    puntajes = []
    for _, fila in X_test.iterrows():
        z = []
        for k, c in enumerate(NUM_COLS):
            v = fila[c]
            if c == "saldo_cuenta_cop" and (v is None or math.isnan(v)):
                v = pre["medianas_saldo"].get(fila["saldo_categoria"], pre["mediana_num"][k]) \
                    if isinstance(fila["saldo_categoria"], str) else pre["mediana_num"][k]
            elif v is None or math.isnan(v):
                v = pre["mediana_num"][k]
            z.append((v - pre["media"][k]) / pre["escala"][k])
        idx = []
        for k, c in enumerate(CAT_COLS):
            v = fila[c] if isinstance(fila[c], str) else pre["moda_cat"][k]
            idx.append(pre["categorias"][k].index(v) if v in pre["categorias"][k] else -1)
        d2 = ((sv_num - np.array(z)) ** 2).sum(axis=1)
        idx = np.array(idx)
        d2 += np.where(sv_cat == idx, 0, np.where(idx < 0, 1, 2)).sum(axis=1)
        puntajes.append(np.dot(svm["alfa"], np.exp(-svm["gamma"] * d2)) + svm["intercepto"])
    referencia = artefacto["pipeline"].decision_function(X_test)
    dif = np.abs(np.array(puntajes) - referencia).max()
    assert dif < 1e-9, f"La exportación no reproduce el pipeline (diferencia {dif})"
    return dif


def construir(modelo):
    plantilla = (DIR_SRC / "plantilla.html").read_text(encoding="utf-8")
    datos = json.dumps(modelo, ensure_ascii=False, separators=(",", ":"))
    datos = datos.replace("</", "<\\/")          # seguro dentro de <script>
    reemplazos = {
        "/*__ESTILOS__*/": (DIR_SRC / "estilos.css").read_text(encoding="utf-8"),
        "/*__MOTOR__*/": (DIR_SRC / "motor.js").read_text(encoding="utf-8"),
        "/*__INTERFAZ__*/": (DIR_SRC / "interfaz.js").read_text(encoding="utf-8"),
        "__MODELO_JSON__": datos,
    }
    for marca, contenido in reemplazos.items():
        assert plantilla.count(marca) == 1, marca
        plantilla = plantilla.replace(marca, contenido)
    SALIDA.write_text(plantilla, encoding="utf-8")
    return SALIDA.stat().st_size


if __name__ == "__main__":
    modelo, artefacto = exportar_modelo()
    dif = verificar_exportacion(modelo, artefacto)
    print(f"Exportación verificada: diferencia máxima con el pipeline = {dif:.2e} (1.300 clientes de prueba)")
    tam = construir(modelo)
    print(f"Página generada: {SALIDA.relative_to(RAIZ)} ({tam / 1024:.0f} KB)")
