"""
Evaluación compartida de modelos (Secciones 7.3 y 7.4).

Reúne en un solo módulo las piezas de evaluación que el notebook 04 definía en
línea (costo 5:1, umbral de costo mínimo, prueba t corregida) y agrega la
validación cruzada anidada que usa el notebook 05.

Procedimiento de ajuste de un modelo (`ajustar_modelo`), idéntico en cada fold
externo y en el ajuste final sobre todo train:
  1. Búsqueda de hiperparámetros con validación cruzada interna (ROC-AUC).
  2. Con los mejores hiperparámetros, puntajes fuera de fold en la misma
     validación interna, y sobre ellos el umbral que minimiza el costo 5:1.
  3. Reajuste del mejor pipeline con todos los datos recibidos.

Nada de esto ve el fold externo (o el test): el umbral y los hiperparámetros se
eligen solo con los datos de entrenamiento de cada partición, así que la
estimación de la validación anidada no está sesgada por la selección.

Convención: `y` binaria, 1 = "malo" (clase positiva).
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import t as t_student
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict

COSTO_FN, COSTO_FP = 5, 1  # matriz de costos del German Credit / del artículo


def costo_medio(y_true, y_pred, c_fn=COSTO_FN, c_fp=COSTO_FP):
    """Costo promedio por cliente con la matriz 5:1 (y: 1 = malo)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return (c_fn * fn + c_fp * fp) / len(y_true)


def umbral_costo_minimo(y_true, puntajes):
    """Umbral sobre el puntaje que minimiza el costo 5:1."""
    candidatos = np.unique(np.quantile(puntajes, np.linspace(0.01, 0.99, 197)))
    costos = [costo_medio(y_true, (puntajes >= u).astype(int)) for u in candidatos]
    return candidatos[int(np.argmin(costos))]


def metodo_puntaje(estimador):
    """`predict_proba` si el modelo la tiene; si no (SVM), `decision_function`."""
    return "predict_proba" if hasattr(estimador, "predict_proba") else "decision_function"


def puntajes(estimador, X):
    """Puntaje de riesgo de la clase "malo" (mayor = más riesgo)."""
    if metodo_puntaje(estimador) == "predict_proba":
        return estimador.predict_proba(X)[:, 1]
    return estimador.decision_function(X)


def metricas(y_true, punt, y_pred):
    """Métricas de la clase "malo". La accuracy se reporta solo como referencia."""
    return {
        "F1 (malo)": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, punt),
        "Recall (malo)": recall_score(y_true, y_pred, zero_division=0),
        "Precisión (malo)": precision_score(y_true, y_pred, zero_division=0),
        "Costo medio (5:1)": costo_medio(y_true, y_pred),
        "Accuracy (ref.)": accuracy_score(y_true, y_pred),
    }


def p_valor_corregido(puntajes_a, puntajes_b, n_train, k):
    """Prueba t pareada corregida de Nadeau y Bengio (2003) sobre puntajes por fold.

    La varianza de las diferencias se multiplica por (1/k + n_test/n_train)
    porque los folds comparten datos de entrenamiento.
    """
    d = np.asarray(puntajes_a) - np.asarray(puntajes_b)
    n_test = n_train / k
    var = d.var(ddof=1) * (1 / k + n_test / (n_train - n_test))
    if var == 0:
        return np.nan
    return 2 * t_student.sf(abs(d.mean() / np.sqrt(var)), df=k - 1)


class ModeloAjustado:
    """Resultado de `ajustar_modelo`: pipeline reajustado, umbral e hiperparámetros."""

    def __init__(self, estimador, umbral, mejores_params):
        self.estimador = estimador
        self.umbral = umbral
        self.mejores_params = mejores_params

    def puntajes(self, X):
        return puntajes(self.estimador, X)

    def predecir(self, X):
        if self.umbral is None:  # modelos sin puntaje útil (clasificador de mayoría)
            return np.asarray(self.estimador.predict(X)).astype(int)
        return (self.puntajes(X) >= self.umbral).astype(int)


def ajustar_modelo(pipeline, espacio, X, y, cv_interna, n_iter=None, ajustar_umbral=True,
                   seed=42):
    """Búsqueda de hiperparámetros + umbral de costo mínimo, solo con (X, y).

    - `espacio` vacío: no hay búsqueda (p. ej. el clasificador de mayoría).
    - `n_iter=None`: búsqueda en malla (`GridSearchCV`); si no, aleatoria.
    """
    if espacio:
        if n_iter is None:
            busqueda = GridSearchCV(pipeline, espacio, cv=cv_interna, scoring="roc_auc", n_jobs=-1)
        else:
            busqueda = RandomizedSearchCV(pipeline, espacio, n_iter=n_iter, cv=cv_interna,
                                          scoring="roc_auc", n_jobs=-1, random_state=seed)
        busqueda.fit(X, y)
        mejor = busqueda.best_estimator_
        mejores_params = {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()}
    else:
        mejor = clone(pipeline).fit(X, y)
        mejores_params = {}

    umbral = None
    if ajustar_umbral:
        punt_oof = cross_val_predict(clone(mejor), X, y, cv=cv_interna,
                                     method=metodo_puntaje(mejor), n_jobs=-1)
        if punt_oof.ndim == 2:
            punt_oof = punt_oof[:, 1]
        umbral = umbral_costo_minimo(np.asarray(y), punt_oof)
    return ModeloAjustado(mejor, umbral, mejores_params)


def validacion_anidada(pipeline, espacio, X, y, cv_externa, cv_interna, n_iter=None,
                       ajustar_umbral=True, importancia=True, n_repeticiones=10, seed=42):
    """Validación cruzada anidada de un procedimiento de ajuste completo.

    En cada fold externo se ejecuta `ajustar_modelo` solo con la parte de
    entrenamiento, y se evalúa una única vez en la parte externa. Devuelve:
      - `folds`: DataFrame con las métricas e hiperparámetros de cada fold externo.
      - `oof`: DataFrame con puntaje y predicción fuera de fold de cada cliente.
      - `importancias`: DataFrame (fold x variable) con la importancia por
        permutación (caída de ROC-AUC) medida en cada fold externo, o None.
    """
    y = pd.Series(np.asarray(y), index=X.index)
    filas, oof, imps = [], [], []
    for k, (i_tr, i_te) in enumerate(cv_externa.split(X, y)):
        t0 = time.time()
        X_tr, X_te, y_tr, y_te = X.iloc[i_tr], X.iloc[i_te], y.iloc[i_tr], y.iloc[i_te]
        ajuste = ajustar_modelo(pipeline, espacio, X_tr, y_tr, cv_interna, n_iter=n_iter,
                                ajustar_umbral=ajustar_umbral, seed=seed)
        punt = ajuste.puntajes(X_te)
        pred = ajuste.predecir(X_te)
        fila = {"fold": k + 1, **metricas(y_te, punt, pred), "umbral": ajuste.umbral,
                "hiperparámetros": ", ".join(f"{p}={_fmt(v)}" for p, v in ajuste.mejores_params.items())}
        oof.append(pd.DataFrame({"fold": k + 1, "puntaje": punt, "pred": pred, "y": y_te.to_numpy()},
                                index=X_te.index))
        if importancia:
            r = permutation_importance(ajuste.estimador, X_te, y_te, scoring="roc_auc",
                                       n_repeats=n_repeticiones, random_state=seed, n_jobs=-1)
            imps.append(pd.Series(r.importances_mean, index=X.columns, name=k + 1))
        fila["tiempo (s)"] = round(time.time() - t0, 1)
        filas.append(fila)
    importancias = pd.DataFrame(imps) if importancia else None
    return pd.DataFrame(filas).set_index("fold"), pd.concat(oof).sort_index(), importancias


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
