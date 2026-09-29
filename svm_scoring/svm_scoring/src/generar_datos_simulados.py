"""
Generación de la base de datos simulada — Riesgo crediticio (contexto colombiano)
Proyecto: Mejora de los modelos de calificación crediticia mediante SVM
Curso: Machine Learning II — Universidad Externado de Colombia

Este script construye el conjunto de datos simulado descrito en la Primera entrega
(Sección 3. Estrategia de datos), reproduciendo la estructura de variables del
German Credit Dataset (Statlog) adaptada a nombres, unidades y categorías del
mercado de crédito de consumo colombiano.

Principios de diseño (tomados literalmente del informe de la Primera entrega):
  - Vía: simulación fundamentada (no hay dataset real disponible por Ley 1266/2008
    de habeas data financiero).
  - Tamaño: >= 6.000 registros.
  - Variable objetivo: riesgo crediticio (bueno/malo), clase minoritaria (malo)
    entre 20% y 25%.
  - Relaciones: no tener cuenta bancaria y mayor monto de crédito -> mayor riesgo
    malo; buen historial de pago previo -> menor riesgo; propósito y duración
    modulan la probabilidad.
  - Versión base sin valores faltantes; versión posterior con 3%-5% de faltantes
    y ruido controlado en variables numéricas.

Reproducibilidad: semilla aleatoria fija (SEED). Todo el flujo se ejecuta con
numpy y pandas; no depende de fuentes externas.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Configuración general
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N = 6500  # por encima del mínimo de 5.000 exigido por la guía del proyecto

OUT_DIR = "/home/claude"

# ---------------------------------------------------------------------------
# 1. Variables sociodemográficas
# ---------------------------------------------------------------------------

# Edad: calibrada sobre la forma de la distribución del German Credit Dataset
# (media ~35.5, sd ~11.4, min 19, max 75) usando una Gamma truncada en [18, 75].
def sample_edad(n):
    shape, scale = 4.3, 4.0  # media del componente gamma ~17 -> +18 = ~35, cola derecha moderada
    edad = rng.gamma(shape, scale, size=n) + 18
    edad = np.clip(edad, 18, 75)
    return np.round(edad).astype(int)

edad = sample_edad(N)

# Género
genero = rng.choice(["M", "F"], size=N, p=[0.54, 0.46])

# Tipo de ocupación: proxy de "Occupation" del artículo, recalibrado a la
# informalidad laboral colombiana (~55%-58% en centros urbanos, DANE/GEIH,
# usado aquí como supuesto de calibración, no como cifra oficial verificada
# dato a dato).
ocupacion_cats = ["informal", "formal_no_calificado", "formal_calificado", "alta_calificacion"]
ocupacion_p = [0.44, 0.24, 0.23, 0.09]
tipo_ocupacion = rng.choice(ocupacion_cats, size=N, p=ocupacion_p)

# ---------------------------------------------------------------------------
# 2. Variables patrimoniales
# ---------------------------------------------------------------------------

# Tipo de vivienda (tenencia urbana colombiana: propiedad más baja que en
# Alemania; arriendo predominante en población que solicita crédito de consumo)
vivienda_cats = ["propia", "arrendada", "familiar"]
vivienda_p = [0.38, 0.42, 0.20]
tipo_vivienda = rng.choice(vivienda_cats, size=N, p=vivienda_p)

# Nivel de ahorro (proxy de "Savings account", altamente sesgado hacia niveles
# bajos, igual que en el dataset original: A61=60.3%, A65=18.3%, A62=10.3%,
# A63=6.3%, A64=4.8% -> aquí colapsado a 4 categorías)
ahorro_cats = ["bajo", "medio", "alto", "muy_alto"]
ahorro_p = [0.55, 0.25, 0.12, 0.08]
nivel_ahorro = rng.choice(ahorro_cats, size=N, p=ahorro_p)

# Saldo en cuenta de ahorro/corriente (proxy de "Checking account balance").
# Se incluye la categoría "sin_cuenta", clave para la relación documentada en
# la Primera entrega (no tener cuenta -> mayor riesgo). Proporciones calibradas
# sobre A11/A12/A13/A14 del dataset original (27.4% / 26.9% / 6.3% / 39.4%).
saldo_cat_cats = ["sin_cuenta", "bajo", "medio", "alto"]
saldo_cat_p = [0.33, 0.30, 0.27, 0.10]
saldo_categoria = rng.choice(saldo_cat_cats, size=N, p=saldo_cat_p)

# Monto del saldo en pesos colombianos, coherente con la categoría (0 si no
# tiene cuenta). Log-normal dentro de cada tramo.
saldo_cuenta_cop = np.zeros(N)
rangos_saldo = {
    "bajo":  (12.0, 0.6),   # log-mean ~ e^12.0 ~ 163k COP
    "medio": (13.3, 0.55),  # ~ 600k COP
    "alto":  (14.6, 0.6),   # ~ 2.2M COP
}
for cat, (mu, sigma) in rangos_saldo.items():
    mask = saldo_categoria == cat
    saldo_cuenta_cop[mask] = rng.lognormal(mu, sigma, size=mask.sum())
saldo_cuenta_cop = np.round(saldo_cuenta_cop, -3)  # redondeo a miles de COP

# ---------------------------------------------------------------------------
# 3. Variables del crédito solicitado
# ---------------------------------------------------------------------------

# Propósito del crédito (recategorización del "Purpose" original a productos
# de consumo típicos en Colombia)
proposito_cats = ["libre_inversion", "vehiculo", "electrodomesticos",
                   "remodelacion", "educacion", "negocio", "otros"]
proposito_p = [0.30, 0.18, 0.13, 0.12, 0.10, 0.10, 0.07]
proposito_credito = rng.choice(proposito_cats, size=N, p=proposito_p)

# Plazo en meses: forma similar a "Duration" (media ~21, sd ~12, rango 4-72)
def sample_plazo(n):
    shape, scale = 3.2, 6.3
    plazo = rng.gamma(shape, scale, size=n) + 4
    plazo = np.clip(plazo, 4, 72)
    return np.round(plazo / 3) * 3  # se agrupa en múltiplos de 3 meses (uso común en Colombia)

plazo_meses = sample_plazo(N).astype(int)

# Monto del crédito en COP: log-normal, con media condicionada al propósito
# (vehículo y negocio piden montos más altos; electrodomésticos, más bajos) y
# al plazo (créditos más largos -> montos más altos), replicando la
# correlación duration-amount del dataset original (corr ~0.62).
proposito_mu_shift = {
    "vehiculo": 0.55, "negocio": 0.45, "remodelacion": 0.15,
    "educacion": 0.10, "libre_inversion": 0.0, "electrodomesticos": -0.35,
    "otros": -0.10,
}
base_mu = 15.2  # log COP; e^15.2 ~ 4.0M COP
monto_credito_cop = np.zeros(N)
for i in range(N):
    mu = base_mu + proposito_mu_shift[proposito_credito[i]] + 0.012 * (plazo_meses[i] - 21)
    monto_credito_cop[i] = rng.lognormal(mu, 0.55)
monto_credito_cop = np.round(np.clip(monto_credito_cop, 500_000, 80_000_000), -3)

# ---------------------------------------------------------------------------
# 4. Variable de comportamiento previo
# ---------------------------------------------------------------------------

# Estado de pago de créditos anteriores (proxy de "Credit history")
# Proporciones calibradas sobre credit_history original:
#   A30+A31 (sin créditos / pagados en otro banco) ~ 9%   -> sin_creditos_previos
#   A32 (pagados a tiempo hasta ahora)             ~ 53%  -> al_dia
#   A33 (retraso en el pasado)                     ~ 9%   -> mora_leve
#   A34 (cuenta crítica / otros créditos existentes)~ 29% -> mora_severa
pago_previo_cats = ["sin_creditos_previos", "al_dia", "mora_leve", "mora_severa"]
pago_previo_p = [0.09, 0.53, 0.09, 0.29]
estado_pago_previo = rng.choice(pago_previo_cats, size=N, p=pago_previo_p)

# ---------------------------------------------------------------------------
# 5. Variable objetivo: riesgo crediticio (bueno / malo)
# ---------------------------------------------------------------------------
# Se construye un puntaje de riesgo latente (logit) combinando las variables
# según la dirección documentada en la Primera entrega, con magnitudes
# inspiradas en la importancia relativa reportada por los autores del
# artículo para los coeficientes del SVM lineal. El intercepto se calibra
# para que la prevalencia de "malo" quede en el rango 20%-25%.

z = np.zeros(N)

# Cuenta bancaria: no tener cuenta -> mayor riesgo; saldo alto -> menor riesgo
z += np.select(
    [saldo_categoria == "sin_cuenta", saldo_categoria == "bajo",
     saldo_categoria == "medio", saldo_categoria == "alto"],
    [0.85, 0.30, -0.25, -0.70],
)

# Nivel de ahorro: mayor ahorro -> menor riesgo
z += np.select(
    [nivel_ahorro == "bajo", nivel_ahorro == "medio",
     nivel_ahorro == "alto", nivel_ahorro == "muy_alto"],
    [0.35, 0.0, -0.35, -0.60],
)

# Historial de pago previo: la relación más fuerte, según el artículo
z += np.select(
    [estado_pago_previo == "sin_creditos_previos", estado_pago_previo == "al_dia",
     estado_pago_previo == "mora_leve", estado_pago_previo == "mora_severa"],
    [0.10, -0.55, 0.55, 1.10],
)

# Monto del crédito (estandarizado en log): a mayor monto, mayor riesgo
log_monto = np.log(monto_credito_cop)
z += 0.45 * (log_monto - log_monto.mean()) / log_monto.std()

# Plazo: créditos más largos -> mayor riesgo (relación análoga a Duration)
z += 0.35 * (plazo_meses - plazo_meses.mean()) / plazo_meses.std()

# Propósito del crédito: negocio y libre inversión con algo más de riesgo;
# vehículo y educación algo menos (garantía real / inversión en capital humano)
z += np.select(
    [proposito_credito == "negocio", proposito_credito == "libre_inversion",
     proposito_credito == "electrodomesticos", proposito_credito == "remodelacion",
     proposito_credito == "vehiculo", proposito_credito == "educacion",
     proposito_credito == "otros"],
    [0.30, 0.15, 0.05, 0.0, -0.20, -0.25, 0.10],
)

# Ocupación: informalidad laboral -> mayor riesgo; alta calificación -> menor riesgo
z += np.select(
    [tipo_ocupacion == "informal", tipo_ocupacion == "formal_no_calificado",
     tipo_ocupacion == "formal_calificado", tipo_ocupacion == "alta_calificacion"],
    [0.45, 0.10, -0.20, -0.55],
)

# Vivienda: vivienda propia -> menor riesgo (proxy de estabilidad patrimonial)
z += np.select(
    [tipo_vivienda == "propia", tipo_vivienda == "arrendada", tipo_vivienda == "familiar"],
    [-0.35, 0.20, 0.10],
)

# Edad: forma en U leve (mayor riesgo en adultos jóvenes, menor riesgo en edades medias)
edad_std = (edad - edad.mean()) / edad.std()
z += 0.15 * edad_std**2 - 0.10 * edad_std

# Ruido idiosincrático (representa factores no observados)
z += rng.normal(0, 0.85, size=N)

# Calibración del intercepto para lograr una prevalencia de "malo" en [20%, 25%]
def prevalencia(intercepto):
    p = 1 / (1 + np.exp(-(z + intercepto)))
    return p.mean()

lo, hi = -5.0, 5.0
objetivo = 0.225  # punto medio del rango 20%-25%
for _ in range(60):
    mid = (lo + hi) / 2
    if prevalencia(mid) < objetivo:
        lo = mid
    else:
        hi = mid
intercepto = (lo + hi) / 2

p_malo = 1 / (1 + np.exp(-(z + intercepto)))
malo = rng.uniform(size=N) < p_malo
riesgo = np.where(malo, "malo", "bueno")

# ---------------------------------------------------------------------------
# 6. Ensamble del DataFrame base (sin valores faltantes)
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "id": np.arange(1, N + 1),
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
    "riesgo": riesgo,
})

df.to_csv(f"{OUT_DIR}/credito_simulado_base.csv", index=False)

print("=== Base simulada (sin faltantes) ===")
print(df.shape)
print(df["riesgo"].value_counts(normalize=True))
print(df.describe(include="all").T[["count", "unique", "top", "freq"]].head(20))

# ---------------------------------------------------------------------------
# 7. Versión con faltantes y ruido controlado (3%-5%)
# ---------------------------------------------------------------------------
rng2 = np.random.default_rng(SEED + 1)
df_ruido = df.copy()

# Variables numéricas -> ruido gaussiano leve (2% del valor) en una submuestra,
# y faltantes MAR (más probable en registros con menor saldo, simulando
# subreporte típico de ingresos/saldos bajos)
num_cols = ["edad", "saldo_cuenta_cop", "monto_credito_cop", "plazo_meses"]
miss_rate = {"edad": 0.03, "saldo_cuenta_cop": 0.05, "monto_credito_cop": 0.04,
             "plazo_meses": 0.03, "tipo_ocupacion": 0.04, "nivel_ahorro": 0.05}

for col, rate in miss_rate.items():
    if col in num_cols:
        # probabilidad de faltante algo mayor para saldos/montos bajos (MAR)
        base_p = np.full(N, rate)
        if col == "saldo_cuenta_cop":
            low = df["saldo_cuenta_cop"] < df["saldo_cuenta_cop"].median()
            base_p = np.where(low, rate * 1.6, rate * 0.5)
        mask = rng2.uniform(size=N) < base_p
        df_ruido.loc[mask, col] = np.nan
    else:
        mask = rng2.uniform(size=N) < rate
        df_ruido.loc[mask, col] = np.nan

# Ruido leve en variables numéricas (sin generar faltantes), 3% de los registros
for col in num_cols:
    mask = rng2.uniform(size=N) < 0.03
    valores = df_ruido.loc[mask & df_ruido[col].notna(), col]
    ruido = rng2.normal(0, 0.02, size=len(valores)) * valores
    df_ruido.loc[valores.index, col] = np.round(valores + ruido)

df_ruido.to_csv(f"{OUT_DIR}/credito_simulado_con_faltantes.csv", index=False)

print("\n=== Versión con faltantes/ruido ===")
print(df_ruido.isna().mean().sort_values(ascending=False).head(10))

print("\nArchivos generados:")
print(f" - {OUT_DIR}/credito_simulado_base.csv")
print(f" - {OUT_DIR}/credito_simulado_con_faltantes.csv")
