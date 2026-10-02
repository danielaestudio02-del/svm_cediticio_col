"""
Preprocesamiento compartido del proyecto (Sección 7.2).

Todos los notebooks de modelamiento (líneas base, SVM, Random Forest,
Boosting) importan de aquí la carga de datos, la partición y el pipeline, para
que las transformaciones sean idénticas y trazables entre modelos.

Orden de las etapas del pipeline (todas se ajustan solo con train):
  1. Imputación del saldo con la mediana de su `saldo_categoria`.
  2. Imputación del resto (mediana / moda) y escalamiento de las numéricas.
  3. Balanceo con SMOTENC (opcional, `balancear=True`), sobre numéricas ya
     escaladas y categóricas todavía sin codificar.
  4. Codificación one-hot de las categóricas.
  5. Selección de variables con LASSO (opcional, `seleccionar=True`): una
     regresión logística con penalización L1 descarta las columnas cuyo
     coeficiente se anula, como en el artículo.
  6. Clasificador.

Balanceo: en el notebook 03, la validación cruzada sobre train mostró que
`class_weight="balanced"` en el clasificador logra el mismo ROC-AUC que no
balancear y más recall de "malo" que SMOTENC. Por eso `balancear=False` es el
valor por defecto y se recomienda usar `class_weight` (o `scale_pos_weight` en
Boosting); SMOTENC queda disponible para modelos que no admiten pesos.

Uso:
    import sys; sys.path.append("../src")
    from preprocesamiento import cargar_particion, construir_pipeline
    pipe = construir_pipeline(SVC(class_weight="balanced"))
"""
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "simulado" / "credito_simulado_con_faltantes.csv"

NUM_COLS = ["edad", "saldo_cuenta_cop", "monto_credito_cop", "plazo_meses"]
CAT_COLS = ["genero", "tipo_ocupacion", "tipo_vivienda", "nivel_ahorro",
            "saldo_categoria", "proposito_credito", "estado_pago_previo"]
OBJETIVO = "riesgo"
CLASE_POSITIVA = "malo"


def cargar_particion(ruta=RUTA_DATOS, test_size=0.20, seed=SEED):
    """Carga la base de trabajo y devuelve la partición estratificada 80/20."""
    df = pd.read_csv(ruta).drop(columns=["id"])
    X, y = df.drop(columns=[OBJETIVO]), df[OBJETIVO]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


class ImputadorSaldoPorCategoria(BaseEstimator, TransformerMixin):
    """Imputa `saldo_cuenta_cop` con la mediana de su `saldo_categoria`.

    La mediana global mezclaría clientes sin cuenta (saldo 0) con clientes de
    saldo alto; la mediana por categoría respeta la coherencia entre ambas
    columnas. Las medianas se aprenden en `fit` (solo con train).
    """

    def __init__(self, col_saldo="saldo_cuenta_cop", col_categoria="saldo_categoria"):
        self.col_saldo = col_saldo
        self.col_categoria = col_categoria

    def fit(self, X, y=None):
        self.medianas_ = X.groupby(self.col_categoria)[self.col_saldo].median()
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        faltante = X[self.col_saldo].isna()
        X.loc[faltante, self.col_saldo] = X.loc[faltante, self.col_categoria].map(self.medianas_)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def construir_preprocesador_inicial():
    """Etapas 1-2: imputación y escalamiento (salida en DataFrame)."""
    imputar_escalar = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUM_COLS),
            ("cat", SimpleImputer(strategy="most_frequent"), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return [("imputar_saldo", ImputadorSaldoPorCategoria()),
            ("imputar_escalar", imputar_escalar)]


def construir_codificador():
    """Etapa 4: one-hot de las categóricas; las numéricas pasan tal cual."""
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def construir_selector_lasso(C=0.1, seed=SEED):
    """Etapa 5: conserva las columnas con coeficiente no nulo en una logística L1.

    `C` controla la fuerza de la selección (más pequeño = menos columnas) y se
    puede ajustar por validación cruzada como `seleccion__estimator__C`.
    """
    lasso = LogisticRegression(l1_ratio=1, solver="liblinear", C=C,
                               class_weight="balanced", random_state=seed)
    return SelectFromModel(lasso)


def construir_pipeline(clasificador, balancear=False, seleccionar=False, seed=SEED):
    """Pipeline: imputación -> escalamiento -> [SMOTENC] -> one-hot -> [LASSO] -> modelo."""
    pasos = construir_preprocesador_inicial()
    if balancear:
        pasos.append(("balanceo", SMOTENC(categorical_features=CAT_COLS, random_state=seed)))
    pasos.append(("codificar", construir_codificador()))
    if seleccionar:
        pasos.append(("seleccion", construir_selector_lasso(seed=seed)))
    pasos.append(("modelo", clasificador))
    return ImbPipeline(steps=pasos)
