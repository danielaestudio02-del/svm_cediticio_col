"""
Generación de la base de datos simulada — Riesgo crediticio (contexto colombiano)
Proyecto: Mejora de los modelos de calificación crediticia mediante SVM
Curso: Machine Learning II — Universidad Externado de Colombia

Este script construye el conjunto de datos simulado descrito en la Primera entrega
(Sección 3. Estrategia de datos), reproduciendo la estructura de variables del
German Credit Dataset (Statlog) adaptada a nombres, unidades y categorías del
mercado de crédito de consumo colombiano.

Principios de diseño (tomados de la Primera entrega):
  - Vía: simulación fundamentada (no hay dataset real disponible por la Ley
    1266 de 2008 de habeas data financiero).
  - Tamaño: >= 6.000 registros.
  - Variable objetivo: riesgo crediticio (bueno/malo), clase minoritaria (malo)
    entre 20% y 25%.
  - Relaciones: no tener cuenta bancaria y mayor monto de crédito -> mayor riesgo
    malo; buen historial de pago previo -> menor riesgo; propósito y duración
    modulan la probabilidad.
  - Versión base sin valores faltantes; versión posterior con 3%-5% de faltantes
    y ruido controlado en variables numéricas.

Reproducibilidad: semilla aleatoria fija (SEED). Todo el flujo se ejecuta con
numpy y pandas; no depende de fuentes externas. El notebook
`notebooks/01_Base_de_datos_simulada.ipynb` replica este mismo código paso a paso
y verifica, al final, que su resultado es idéntico al de `generar_base()`.

Uso:
    python src/generar_datos_simulados.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuración general
# ---------------------------------------------------------------------------
SEED = 42
N = 6500  # por encima del mínimo de 5.000 exigido por la guía del proyecto

RAIZ = Path(__file__).resolve().parents[1]
OUT_DIR = RAIZ / "data" / "simulado"

# Probabilidades de las variables categóricas. Las que tienen análogo en
# german.data se calibran contra sus proporciones; las demás son supuestos
# del contexto colombiano, documentados como tales.
OCUPACION_CATS = ["informal", "formal_no_calificado", "formal_calificado", "alta_calificacion"]
OCUPACION_P = [0.44, 0.24, 0.23, 0.09]  # supuesto: informalidad laboral colombiana

VIVIENDA_CATS = ["propia", "arrendada", "familiar"]
VIVIENDA_P = [0.38, 0.42, 0.20]  # ajuste deliberado: más arriendo que en el original

# savings: A61=60.3%, A62=10.3%, A63=6.3%, A64=4.8%, A65 (desconocido/sin
# ahorro)=18.3%. A61..A64 se mapean en orden a bajo..muy_alto y el 18.3% de
# A65 se reparte proporcionalmente entre ellas: 60.3/81.7 = 0.74, etc.
AHORRO_CATS = ["bajo", "medio", "alto", "muy_alto"]
AHORRO_P = [0.74, 0.13, 0.07, 0.06]

# checking: A14 (sin cuenta)=39.4%, A11 (< 0 DM)=27.4%, A12 (0-200 DM)=26.9%,
# A13 (>= 200 DM)=6.3%, mapeados en orden a sin_cuenta/bajo/medio/alto.
SALDO_CATS = ["sin_cuenta", "bajo", "medio", "alto"]
SALDO_P = [0.39, 0.27, 0.27, 0.07]
SALDO_LOGNORMAL = {          # (mu, sigma) del log-saldo en COP por tramo
    "bajo":  (12.0, 0.60),   # e^12.0 ~ 163 mil COP
    "medio": (13.3, 0.55),   # e^13.3 ~ 600 mil COP
    "alto":  (14.6, 0.60),   # e^14.6 ~ 2.2 M COP
}

PROPOSITO_CATS = ["libre_inversion", "vehiculo", "electrodomesticos",
                  "remodelacion", "educacion", "negocio", "otros"]
PROPOSITO_P = [0.30, 0.18, 0.13, 0.12, 0.10, 0.10, 0.07]
PROPOSITO_MU_SHIFT = {  # desplazamiento del log-monto según el propósito
    "vehiculo": 0.55, "negocio": 0.45, "remodelacion": 0.15,
    "educacion": 0.10, "libre_inversion": 0.0, "electrodomesticos": -0.35,
    "otros": -0.10,
}

# credit_history -> estado_pago_previo:
#   A30+A31 (sin créditos, o todos pagados a tiempo en este banco) ~ 9%
#   A32 (créditos vigentes pagados a tiempo hasta ahora)            ~ 53%
#   A33 (retrasos en el pasado)                                     ~ 9%
#   A34 (cuenta crítica / otros créditos en otras entidades)        ~ 29%
# Las proporciones se toman del original; la DIRECCIÓN del riesgo no, porque
# en el cruce univariado de german.data A30/A31 tienen la mayor tasa de
# "malo" y A34 la menor (efecto de confusión, igual que con A14). Ver
# docs/construccion_base_datos.md, Paso 1.
PAGO_PREVIO_CATS = ["sin_creditos_previos", "al_dia", "mora_leve", "mora_severa"]
PAGO_PREVIO_P = [0.09, 0.53, 0.09, 0.29]

PREVALENCIA_OBJETIVO = 0.225  # punto medio del rango 20%-25%


# ---------------------------------------------------------------------------
# 1. Funciones de muestreo
# ---------------------------------------------------------------------------
def sample_edad(n, rng):
    """Gamma desplazada y truncada en [18, 75].

    Gamma(2.4, 7.3) + 18 da media ~35.3, desviación ~10.9 y asimetría ~1.1,
    frente a 35.5 / 11.4 / 1.02 de `age` en german.data.
    """
    edad = rng.gamma(2.4, 7.3, size=n) + 18
    return np.round(np.clip(edad, 18, 75)).astype(int)


def sample_plazo(n, rng):
    """Gamma desplazada, agrupada en múltiplos de 3 meses y acotada a [3, 72].

    Gamma(2.2, 7.7) + 4 da media ~20.9, desviación ~11.4 y asimetría ~1.2,
    frente a 20.9 / 12.1 / 1.09 de `duration` en german.data.
    """
    plazo = rng.gamma(2.2, 7.7, size=n) + 4
    plazo = np.round(plazo / 3) * 3  # múltiplos de 3 meses (uso común en Colombia)
    return np.clip(plazo, 3, 72).astype(int)


def sample_monto(plazo_meses, proposito_credito, rng):
    """Log-normal condicionada al propósito y al plazo.

    log(monto) = 15.2 + shift(propósito) + 0.95 * log(plazo / 21) + e,
    e ~ N(0, 0.47). La elasticidad de 0.95 respecto al plazo reproduce la
    correlación duration-credit_amount del original (~0.62) y sigma = 0.47 su
    dispersión relativa (CV ~0.85). La asimetría queda algo mayor que en el
    original (~2.8 vs. 1.95), propio de la cola log-normal.
    """
    shift = np.array([PROPOSITO_MU_SHIFT[p] for p in proposito_credito])
    mu = 15.2 + shift + 0.95 * np.log(plazo_meses / 21)
    monto = rng.lognormal(mu, 0.47)
    return np.round(np.clip(monto, 500_000, 80_000_000), -3)


# Pesos del puntaje latente de riesgo (escala logit). Se exponen como
# constantes para poder contrastarlos con lo que aprenden los modelos.
PESOS_CATEGORICOS = {
    # Cuenta bancaria: no tener cuenta -> mayor riesgo; saldo alto -> menor riesgo
    "saldo_categoria": {"sin_cuenta": 0.85, "bajo": 0.30, "medio": -0.25, "alto": -0.70},
    # Nivel de ahorro: mayor ahorro -> menor riesgo
    "nivel_ahorro": {"bajo": 0.35, "medio": 0.0, "alto": -0.35, "muy_alto": -0.60},
    # Historial de pago previo: la relación más fuerte. Orden de riesgo:
    # al_dia < sin_creditos_previos (historial delgado) < mora_leve < mora_severa
    "estado_pago_previo": {"al_dia": -0.55, "sin_creditos_previos": 0.10,
                           "mora_leve": 0.55, "mora_severa": 1.10},
    # Propósito: negocio y libre inversión con algo más de riesgo; vehículo y
    # educación algo menos (garantía real / inversión en capital humano)
    "proposito_credito": {"negocio": 0.30, "libre_inversion": 0.15, "electrodomesticos": 0.05,
                          "remodelacion": 0.0, "vehiculo": -0.20, "educacion": -0.25,
                          "otros": 0.10},
    # Ocupación: informalidad -> mayor riesgo; alta calificación -> menor riesgo
    "tipo_ocupacion": {"informal": 0.45, "formal_no_calificado": 0.10,
                       "formal_calificado": -0.20, "alta_calificacion": -0.55},
    # Vivienda: vivienda propia -> menor riesgo (estabilidad patrimonial)
    "tipo_vivienda": {"propia": -0.35, "arrendada": 0.20, "familiar": 0.10},
}
# Numéricas: peso por desviación estándar (monto en log); edad con forma en U
PESOS_NUMERICOS = {"log_monto": 0.45, "plazo": 0.35, "edad": -0.10, "edad_cuadrado": 0.15}
SD_RUIDO_LATENTE = 0.85  # factores no observados


def puntaje_latente(df, rng=None, incluir_ruido=True):
    """Puntaje de riesgo latente z (sin intercepto) a partir de los predictores.

    Con `incluir_ruido=False` devuelve solo la parte observable (sin el ruido
    idiosincrático), que no consume números aleatorios.
    """
    z = np.zeros(len(df))
    for col in ["saldo_categoria", "nivel_ahorro", "estado_pago_previo"]:
        z += df[col].map(PESOS_CATEGORICOS[col]).to_numpy()

    # Monto (estandarizado en log): a mayor monto, mayor riesgo
    log_monto = np.log(df["monto_credito_cop"].to_numpy())
    z += PESOS_NUMERICOS["log_monto"] * (log_monto - log_monto.mean()) / log_monto.std()

    # Plazo: créditos más largos -> mayor riesgo (análogo a Duration)
    plazo = df["plazo_meses"].to_numpy()
    z += PESOS_NUMERICOS["plazo"] * (plazo - plazo.mean()) / plazo.std()

    for col in ["proposito_credito", "tipo_ocupacion", "tipo_vivienda"]:
        z += df[col].map(PESOS_CATEGORICOS[col]).to_numpy()

    # Edad: forma en U leve (más riesgo en adultos jóvenes)
    edad = df["edad"].to_numpy()
    edad_std = (edad - edad.mean()) / edad.std()
    z += PESOS_NUMERICOS["edad_cuadrado"] * edad_std**2 + PESOS_NUMERICOS["edad"] * edad_std

    # Ruido idiosincrático (factores no observados)
    if incluir_ruido:
        z += rng.normal(0, SD_RUIDO_LATENTE, size=len(df))
    return z


def probabilidad_observable(z_observable, intercepto, n_nodos=60):
    """P(malo | variables observadas): integra el ruido latente N(0, SD_RUIDO_LATENTE^2).

    Es la mejor predicción posible con las variables disponibles (el techo
    teórico de cualquier modelo). Se calcula con cuadratura de Gauss-Hermite.
    """
    nodos, pesos = np.polynomial.hermite_e.hermegauss(n_nodos)
    pesos = pesos / pesos.sum()
    eta = z_observable[:, None] + intercepto + SD_RUIDO_LATENTE * nodos[None, :]
    return (1 / (1 + np.exp(-eta))) @ pesos


def calibrar_intercepto(z, objetivo=PREVALENCIA_OBJETIVO):
    """Bisección sobre el intercepto para que la probabilidad media sea `objetivo`."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() < objetivo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# 2. Base sin faltantes
# ---------------------------------------------------------------------------
def generar_base(seed=SEED, n=N, devolver_probabilidad=False):
    """Genera la base simulada sin faltantes.

    Si `devolver_probabilidad=True`, devuelve además P(malo | variables
    observadas), la probabilidad que tendría un modelo perfecto que conociera la
    estructura generadora pero no el ruido idiosincrático (techo teórico de
    discriminación, ver notebook 04).
    """
    rng = np.random.default_rng(seed)

    # Sociodemográficas
    edad = sample_edad(n, rng)
    genero = rng.choice(["M", "F"], size=n, p=[0.54, 0.46])
    tipo_ocupacion = rng.choice(OCUPACION_CATS, size=n, p=OCUPACION_P)

    # Patrimoniales
    tipo_vivienda = rng.choice(VIVIENDA_CATS, size=n, p=VIVIENDA_P)
    nivel_ahorro = rng.choice(AHORRO_CATS, size=n, p=AHORRO_P)
    saldo_categoria = rng.choice(SALDO_CATS, size=n, p=SALDO_P)
    saldo_cuenta_cop = np.zeros(n)  # 0 si no tiene cuenta
    for cat, (mu, sigma) in SALDO_LOGNORMAL.items():
        mask = saldo_categoria == cat
        saldo_cuenta_cop[mask] = rng.lognormal(mu, sigma, size=mask.sum())
    saldo_cuenta_cop = np.round(saldo_cuenta_cop, -3)

    # Crédito solicitado
    proposito_credito = rng.choice(PROPOSITO_CATS, size=n, p=PROPOSITO_P)
    plazo_meses = sample_plazo(n, rng)
    monto_credito_cop = sample_monto(plazo_meses, proposito_credito, rng)

    # Comportamiento previo
    estado_pago_previo = rng.choice(PAGO_PREVIO_CATS, size=n, p=PAGO_PREVIO_P)

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "edad": edad,
        "genero": genero,
        "tipo_ocupacion": tipo_ocupacion,
        "tipo_vivienda": tipo_vivienda,
        "nivel_ahorro": nivel_ahorro,
        "saldo_categoria": saldo_categoria,
        "saldo_cuenta_cop": saldo_cuenta_cop,
        "monto_credito_cop": monto_credito_cop,
        "plazo_meses": plazo_meses,
        "proposito_credito": proposito_credito,
        "estado_pago_previo": estado_pago_previo,
    })

    # Variable objetivo
    z = puntaje_latente(df, rng)
    intercepto = calibrar_intercepto(z)
    p_malo = 1 / (1 + np.exp(-(z + intercepto)))
    df["riesgo"] = np.where(rng.uniform(size=n) < p_malo, "malo", "bueno")

    if devolver_probabilidad:
        z_observable = puntaje_latente(df, incluir_ruido=False)
        return df, probabilidad_observable(z_observable, intercepto)
    return df


# ---------------------------------------------------------------------------
# 3. Versión con faltantes y ruido controlado (3%-5%)
# ---------------------------------------------------------------------------
TASA_FALTANTES = {"edad": 0.03, "saldo_cuenta_cop": 0.05, "monto_credito_cop": 0.04,
                  "plazo_meses": 0.03, "tipo_ocupacion": 0.04, "nivel_ahorro": 0.05}


def agregar_faltantes_y_ruido(df, seed=SEED + 1):
    """Introduce faltantes (3%-5% por columna) y ruido leve de medición.

    - `saldo_cuenta_cop`: faltante MAR, solo entre quienes SÍ tienen cuenta
      (para `sin_cuenta` el saldo es 0 por definición y no hay nada que
      reportar); es 1.6 veces más probable en saldos por debajo de la mediana
      y 0.5 veces por encima (subreporte típico de saldos bajos). La tasa se
      reescala para que el total de la columna siga siendo ~5%.
    - Demás columnas: faltantes MCAR (completamente al azar).
    - Ruido gaussiano del 2% del valor en ~3% de los registros de `edad`,
      `saldo_cuenta_cop` y `monto_credito_cop` (error de reporte). El plazo no
      lleva ruido: es un término contractual, no una variable reportada.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(df)

    for col, tasa in TASA_FALTANTES.items():
        if col == "saldo_cuenta_cop":
            con_cuenta = df["saldo_categoria"] != "sin_cuenta"
            mediana = df.loc[con_cuenta, col].median()
            bajo = df[col] < mediana
            tasa_con_cuenta = tasa / con_cuenta.mean()
            # multiplicador medio = (1.6 + 0.5) / 2 = 1.05 -> se divide para no inflar la tasa
            p = np.where(bajo, 1.6, 0.5) / 1.05 * tasa_con_cuenta
            p = np.where(con_cuenta, p, 0.0)
        else:
            p = np.full(n, tasa)
        out.loc[rng.uniform(size=n) < p, col] = np.nan

    for col, decimales, (lo, hi) in [("edad", 0, (18, 75)),
                                     ("saldo_cuenta_cop", -3, (0, np.inf)),
                                     ("monto_credito_cop", -3, (500_000, 80_000_000))]:
        mask = (rng.uniform(size=n) < 0.03) & out[col].notna() & (out[col] > 0)
        valores = out.loc[mask, col]
        ruido = rng.normal(0, 0.02, size=len(valores)) * valores
        out.loc[mask, col] = np.clip(np.round(valores + ruido, decimales), lo, hi)

    return out


# ---------------------------------------------------------------------------
# 4. Ejecución
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generar_base()
    df.to_csv(OUT_DIR / "credito_simulado_base.csv", index=False)
    print("=== Base simulada (sin faltantes) ===")
    print(df.shape)
    print(df["riesgo"].value_counts(normalize=True).round(4))

    df_faltantes = agregar_faltantes_y_ruido(df)
    df_faltantes.to_csv(OUT_DIR / "credito_simulado_con_faltantes.csv", index=False)
    print("\n=== Versión con faltantes/ruido ===")
    print(df_faltantes.isna().mean().loc[lambda s: s > 0].round(4))

    print("\nArchivos generados en", OUT_DIR)
