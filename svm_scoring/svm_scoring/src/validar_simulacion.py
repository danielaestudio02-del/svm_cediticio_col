"""
Validación de la base simulada frente a la estructura prevista en la Primera
entrega y frente al German Credit Dataset original.

Cada verificación tiene un criterio de aceptación explícito; si alguna no se
cumple, el script termina con código de salida 1.

Genera en docs/validacion/:
  - resumen_validacion.md   (tablas y verificación textual)
  - fig_balance_clase.png
  - fig_distribuciones.png
  - fig_relaciones_riesgo.png

Uso:
    python src/validar_simulacion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
OUT_DIR = RAIZ / "docs" / "validacion"

COLS_ORIGINAL = [
    "checking", "duration", "credit_history", "purpose", "credit_amount", "savings",
    "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target",
]

# Orden de riesgo esperado (de menor a mayor tasa de "malo")
ORDEN_RIESGO = {
    "saldo_categoria": ["alto", "medio", "bajo", "sin_cuenta"],
    "estado_pago_previo": ["al_dia", "sin_creditos_previos", "mora_leve", "mora_severa"],
    "nivel_ahorro": ["muy_alto", "alto", "medio", "bajo"],
    "tipo_ocupacion": ["alta_calificacion", "formal_calificado", "formal_no_calificado", "informal"],
}


def tasa_malo(df, por):
    return df.groupby(por, observed=True)["riesgo"].apply(lambda s: (s == "malo").mean())


def es_creciente(serie):
    v = serie.to_numpy()
    return bool(np.all(np.diff(v) > 0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAIZ / "data" / "simulado" / "credito_simulado_base.csv")
    df_na = pd.read_csv(RAIZ / "data" / "simulado" / "credito_simulado_con_faltantes.csv")
    orig = pd.read_csv(RAIZ / "data" / "original" / "german.data",
                       sep=" ", header=None, names=COLS_ORIGINAL)
    orig["target_label"] = orig["target"].map({1: "bueno", 2: "malo"})

    checks = {}
    rep = ["# Validación de la base de datos simulada\n",
           f"Registros simulados: **{len(df):,}** (mínimo exigido: 5.000)\n"]
    checks["Tamaño >= 5.000"] = len(df) >= 5000

    # 1. Balance de clases ---------------------------------------------------
    prop_sim = df["riesgo"].value_counts(normalize=True)
    prop_orig = orig["target_label"].value_counts(normalize=True)
    checks["Balance 20%-25% de 'malo'"] = 0.20 <= prop_sim["malo"] <= 0.25
    rep += ["## 1. Balance de la variable objetivo\n",
            f"- Simulado: bueno={prop_sim['bueno']:.1%}, malo={prop_sim['malo']:.1%} "
            "(criterio de la Primera entrega: 20%-25% de 'malo').",
            f"- German Credit original: bueno={prop_orig['bueno']:.1%}, malo={prop_orig['malo']:.1%}.\n"]

    # 2. Relaciones con el objetivo -------------------------------------------
    rep += ["## 2. Relaciones variable-objetivo (monotonicidad)\n",
            "Criterio: la tasa de 'malo' debe crecer estrictamente en el orden de riesgo esperado.\n",
            "| Variable | Orden esperado (menor -> mayor riesgo) | Tasa de 'malo' | ¿Cumple? |",
            "|---|---|---|---|"]
    tablas = {}
    for col, orden in ORDEN_RIESGO.items():
        t = tasa_malo(df, col).reindex(orden)
        tablas[col] = t
        ok = es_creciente(t)
        checks[f"Monotonicidad {col}"] = ok
        rep.append(f"| {col} | {' < '.join(orden)} | "
                   f"{', '.join(f'{v:.1%}' for v in t)} | {'Sí' if ok else 'No'} |")
    for col, etiqueta in [("monto_credito_cop", "Cuartil de monto"), ("plazo_meses", "Cuartil de plazo")]:
        q = pd.qcut(df[col], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        t = tasa_malo(df.assign(q=q), "q")
        tablas[col] = t
        ok = es_creciente(t)
        checks[f"Monotonicidad {col}"] = ok
        rep.append(f"| {etiqueta} | Q1 < Q2 < Q3 < Q4 | "
                   f"{', '.join(f'{v:.1%}' for v in t)} | {'Sí' if ok else 'No'} |")

    # 3. Forma de las distribuciones numéricas --------------------------------
    rep += ["\n## 3. Forma de las distribuciones numéricas (simulado vs. original)\n",
            "Criterio: asimetría positiva en las tres variables y coeficiente de variación "
            "(CV) a menos de 0.10 del original.\n",
            "| Variable | Media sim. | CV sim. | Asimetría sim. | Media orig. | CV orig. | Asimetría orig. |",
            "|---|---|---|---|---|---|---|"]
    for sim, org, nombre in [("edad", "age", "Edad (años)"),
                             ("plazo_meses", "duration", "Plazo (meses)"),
                             ("monto_credito_cop", "credit_amount", "Monto (COP vs. DM)")]:
        s, o = df[sim], orig[org]
        cv_s, cv_o = s.std() / s.mean(), o.std() / o.mean()
        checks[f"Forma {sim}"] = s.skew() > 0 and abs(cv_s - cv_o) < 0.10
        rep.append(f"| {nombre} | {s.mean():,.1f} | {cv_s:.2f} | {s.skew():.2f} | "
                   f"{o.mean():,.1f} | {cv_o:.2f} | {o.skew():.2f} |")

    # 4. Estructura de dependencia -------------------------------------------
    corr_sim = df["plazo_meses"].corr(df["monto_credito_cop"])
    corr_orig = orig["duration"].corr(orig["credit_amount"])
    checks["Correlación plazo-monto a ±0.05 del original"] = abs(corr_sim - corr_orig) < 0.05
    rep += ["\n## 4. Estructura de dependencia\n",
            f"- Correlación de Pearson plazo-monto: simulado **{corr_sim:.3f}**, "
            f"original (duration-credit_amount) **{corr_orig:.3f}**. Criterio: diferencia < 0.05.\n"]

    # 5. Proporciones de categorías frente al original ------------------------
    rep += ["## 5. Proporciones de categorías calibradas contra el original\n",
            "| Variable simulada | Categoría | Simulado | Original (mapeado) |",
            "|---|---|---|---|"]
    mapas = {
        "saldo_categoria": ("checking", {"A14": "sin_cuenta", "A11": "bajo", "A12": "medio", "A13": "alto"}),
        "estado_pago_previo": ("credit_history", {"A30": "sin_creditos_previos", "A31": "sin_creditos_previos",
                                                  "A32": "al_dia", "A33": "mora_leve", "A34": "mora_severa"}),
    }
    for col, (col_o, mapa) in mapas.items():
        p_s = df[col].value_counts(normalize=True)
        p_o = orig[col_o].map(mapa).value_counts(normalize=True)
        for cat in p_o.index:
            rep.append(f"| {col} | {cat} | {p_s.get(cat, 0):.1%} | {p_o[cat]:.1%} |")
    rep.append("\n`nivel_ahorro` se calibró con A61-A64 redistribuyendo A65 (desconocido); "
               "`tipo_vivienda` y `tipo_ocupacion` son supuestos del contexto colombiano.\n")

    # 6. Versión con faltantes -------------------------------------------------
    tasas = df_na.isna().mean()
    tasas = tasas[tasas > 0]
    saldo_na_sin_cuenta = (df_na["saldo_cuenta_cop"].isna() & (df_na["saldo_categoria"] == "sin_cuenta")).sum()
    checks["Faltantes entre 2.5% y 5.5% por columna"] = bool(tasas.between(0.025, 0.055).all())
    checks["Sin saldos faltantes en 'sin_cuenta'"] = saldo_na_sin_cuenta == 0
    rep += ["## 6. Versión con faltantes\n",
            "| Columna | % faltante |", "|---|---|"]
    rep += [f"| {c} | {v:.2%} |" for c, v in tasas.items()]
    rep.append(f"\nSaldos faltantes en clientes `sin_cuenta` (debe ser 0): {saldo_na_sin_cuenta}\n")

    # Resumen -------------------------------------------------------------------
    rep += ["## Resumen de verificaciones\n", "| Verificación | Resultado |", "|---|---|"]
    rep += [f"| {k} | {'CUMPLE' if v else 'NO CUMPLE'} |" for k, v in checks.items()]
    (OUT_DIR / "resumen_validacion.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    # Figuras ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    prop_sim.reindex(["bueno", "malo"]).plot(kind="bar", ax=ax[0], color=["#4C78A8", "#E45756"])
    ax[0].set_title(f"Simulado (n={len(df):,})")
    ax[0].set_ylabel("Proporción")
    prop_orig.reindex(["bueno", "malo"]).plot(kind="bar", ax=ax[1], color=["#4C78A8", "#E45756"])
    ax[1].set_title("German Credit original (n=1.000)")
    for a in ax:
        a.set_ylim(0, 1)
        a.tick_params(axis="x", rotation=0)
    fig.suptitle("Balance de la variable objetivo")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_balance_clase.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for a, sim, org, titulo in [(axes[0], "edad", "age", "Edad (años)"),
                                (axes[1], "plazo_meses", "duration", "Plazo (meses)")]:
        a.hist(orig[org], bins=25, density=True, alpha=0.55, label="Original", color="#4C78A8")
        a.hist(df[sim], bins=25, density=True, alpha=0.55, label="Simulado", color="#F58518")
        a.set_title(titulo)
        a.legend()
    z_o = (orig["credit_amount"] - orig["credit_amount"].mean()) / orig["credit_amount"].std()
    z_s = (df["monto_credito_cop"] - df["monto_credito_cop"].mean()) / df["monto_credito_cop"].std()
    axes[2].hist(z_o, bins=30, density=True, alpha=0.55, label="Original", color="#4C78A8")
    axes[2].hist(z_s, bins=30, density=True, alpha=0.55, label="Simulado", color="#F58518")
    axes[2].set_title("Monto estandarizado (z-score)")
    axes[2].legend()
    fig.suptitle("Distribuciones numéricas: original vs. simulado")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_distribuciones.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    tablas["saldo_categoria"].plot(kind="bar", ax=axes[0], color="#E45756")
    axes[0].set_title("Tasa de 'malo' según saldo en cuenta")
    axes[0].set_ylabel("Tasa de riesgo 'malo'")
    tablas["estado_pago_previo"].plot(kind="bar", ax=axes[1], color="#E45756")
    axes[1].set_title("Tasa de 'malo' según historial de pago")
    for a in axes:
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_relaciones_riesgo.png", dpi=140)
    plt.close(fig)

    for k, v in checks.items():
        print(f"[{'OK' if v else 'FALLA'}] {k}")
    print(f"\nResultados guardados en {OUT_DIR}")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
