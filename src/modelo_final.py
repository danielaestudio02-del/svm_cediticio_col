"""
Entrena y guarda el modelo final (Entrega final, 8.1 y 8.3).

Modelo: SVM con kernel RBF y pesos de clase balanceados, dentro del `Pipeline`
compartido (imputación -> escalamiento -> one-hot -> SVM). Se ajusta con el
mismo procedimiento que se evaluó en la validación cruzada anidada del
notebook 05 (`evaluacion.ajustar_modelo`):
  1. búsqueda aleatoria de C y gamma (12 combinaciones, 3 folds, ROC-AUC);
  2. umbral de costo mínimo 5:1 sobre puntajes fuera de fold;
  3. reajuste con todo el entrenamiento (5.200 clientes).

Además, para que la aplicación muestre una probabilidad y no solo el margen del
SVM, se ajusta una calibración de Platt (regresión logística de una variable)
sobre los mismos puntajes fuera de fold. Es monótona, así que no cambia el
orden de los clientes, el ROC-AUC ni la decisión: solo traduce el puntaje a
una probabilidad de "malo".

El artefacto (`models/modelo_final.joblib`) guarda el pipeline, el umbral, la
calibración, el perfil de los datos de entrenamiento (para verificar los datos
de uso), una muestra de fondo para SHAP y las métricas de referencia.

Uso (desde la raíz del repositorio):
    python src/modelo_final.py
"""
import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocesamiento import SEED, RAIZ, cargar_particion, construir_pipeline  # noqa: E402
from evaluacion import ajustar_modelo, metricas  # noqa: E402
from interpretacion import COLUMNAS, perfil_datos  # noqa: E402

RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
RUTA_CONFIG = RAIZ / "docs" / "resultados" / "06_configuracion_modelo_final.json"

NOMBRE = "SVM RBF"
ESPACIO = {"modelo__C": [0.1, 0.3, 1, 3, 10],
           "modelo__gamma": [0.001, 0.003, 0.01, 0.03, 0.1]}   # el mismo del notebook 05
N_ITER = 12
TAM_FONDO_SHAP = 100


def entrenar(seed=SEED):
    X_train, X_test, y_train_txt, y_test_txt = cargar_particion()
    y_train = (y_train_txt == "malo").astype(int).to_numpy()
    y_test = (y_test_txt == "malo").astype(int).to_numpy()
    cv_interna = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    ajuste = ajustar_modelo(construir_pipeline(SVC(kernel="rbf", class_weight="balanced")),
                            ESPACIO, X_train, y_train, cv_interna, n_iter=N_ITER, seed=seed)

    # Calibración de Platt sobre los mismos puntajes fuera de fold del umbral
    s_oof = cross_val_predict(clone(ajuste.estimador), X_train, y_train, cv=cv_interna,
                              method="decision_function", n_jobs=-1)
    platt = LogisticRegression(C=1e6).fit(s_oof.reshape(-1, 1), y_train)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])

    punt_test = ajuste.puntajes(X_test)
    pred_test = ajuste.predecir(X_test)
    m_test = {k: round(float(v), 4) for k, v in metricas(y_test, punt_test, pred_test).items()}

    artefacto = {
        "nombre": NOMBRE,
        "pipeline": ajuste.estimador,
        "umbral": float(ajuste.umbral),
        "calibrador": (a, b),
        "hiperparametros": {k: float(v) for k, v in ajuste.mejores_params.items()},
        "columnas": COLUMNAS,
        "perfil": perfil_datos(X_train),
        "fondo_shap": X_train[COLUMNAS].sample(TAM_FONDO_SHAP, random_state=seed),
        "metricas_test": m_test,
        "n_vectores_soporte": int(ajuste.estimador.named_steps["modelo"].n_support_.sum()),
        "costos": {"falso_negativo": 5, "falso_positivo": 1},
        "fecha_entrenamiento": date.today().isoformat(),
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "numpy": np.__version__},
    }
    return artefacto


def main():
    artefacto = entrenar()
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, RUTA_MODELO, compress=3)

    a, b = artefacto["calibrador"]
    p_umbral = 1 / (1 + np.exp(-(a * artefacto["umbral"] + b)))
    config = {k: artefacto[k] for k in ["nombre", "hiperparametros", "umbral", "metricas_test",
                                        "n_vectores_soporte", "costos", "fecha_entrenamiento",
                                        "versiones"]}
    config["calibracion_platt"] = {"a": a, "b": b, "umbral_en_probabilidad": round(float(p_umbral), 4)}
    RUTA_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RUTA_CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en {RUTA_MODELO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
