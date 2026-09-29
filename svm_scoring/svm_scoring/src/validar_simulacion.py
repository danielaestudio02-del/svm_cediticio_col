"""
Validación de la base simulada frente a la estructura prevista en la Primera
entrega y frente al German Credit Dataset original.

Genera:
  - resumen_validacion.md   (tablas y verificación textual)
  - fig_balance_clase.png
  - fig_distribuciones.png
  - fig_relaciones_riesgo.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/claude"

df = pd.read_csv(f"{OUT_DIR}/credito_simulado_base.csv")

cols_original = ["checking","duration","credit_history","purpose","credit_amount","savings",
        "employment_since","installment_rate","personal_status_sex","other_debtors",
        "residence_since","property","age","other_installment_plans","housing",
        "existing_credits","job","num_dependents","telephone","foreign_worker","target"]
orig = pd.read_csv("/root/.claude/uploads/56d904e8-ae2b-547f-acf1-cda653c1ae92/be88bc40-german.data",
                    sep=" ", header=None, names=cols_original)
orig["target_label"] = orig["target"].map({1: "bueno", 2: "malo"})

report = []
report.append("# Validación de la base de datos simulada\n")
report.append(f"Registros simulados: **{len(df)}** (mínimo exigido: 5.000)\n")

# 1. Balance de clases
prop_sim = df["riesgo"].value_counts(normalize=True)
prop_orig = orig["target_label"].value_counts(normalize=True)
report.append("## 1. Balance de la variable objetivo\n")
report.append(f"- Simulado: bueno={prop_sim['bueno']:.1%}, malo={prop_sim['malo']:.1%} "
              f"(objetivo documentado en Primera entrega: 20%-25% de 'malo')\n")
report.append(f"- German Credit original: bueno={prop_orig['bueno']:.1%}, malo={prop_orig['malo']:.1%} "
              f"(desbalance de referencia: 30%)\n")
ok_balance = 0.20 <= prop_sim["malo"] <= 0.25
report.append(f"- **Verificación:** {'CUMPLE' if ok_balance else 'NO CUMPLE'} el rango 20%-25% especificado.\n")

# 2. Relaciones variable-objetivo (dirección esperada)
report.append("\n## 2. Verificación de relaciones documentadas\n")
report.append("| Relación esperada (Primera entrega) | Tasa de 'malo' observada en la simulación |")
report.append("|---|---|")

t = df.groupby("saldo_categoria")["riesgo"].apply(lambda s: (s == "malo").mean())
t = t.reindex(["sin_cuenta", "bajo", "medio", "alto"])
report.append(f"| No tener cuenta -> mayor riesgo | {', '.join(f'{k}: {v:.1%}' for k, v in t.items())} |")

t2 = df.groupby("estado_pago_previo")["riesgo"].apply(lambda s: (s == "malo").mean())
t2 = t2.reindex(["sin_creditos_previos", "al_dia", "mora_leve", "mora_severa"])
report.append(f"| Buen historial de pago -> menor riesgo | {', '.join(f'{k}: {v:.1%}' for k, v in t2.items())} |")

q = pd.qcut(df["monto_credito_cop"], 4, labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"])
t3 = df.groupby(q, observed=True)["riesgo"].apply(lambda s: (s == "malo").mean())
report.append(f"| Mayor monto de crédito -> mayor riesgo | {', '.join(f'{k}: {v:.1%}' for k, v in t3.items())} |")

qp = pd.qcut(df["plazo_meses"], 4, labels=["Q1 (corto)", "Q2", "Q3", "Q4 (largo)"], duplicates="drop")
t4 = df.groupby(qp, observed=True)["riesgo"].apply(lambda s: (s == "malo").mean())
report.append(f"| Mayor plazo -> mayor riesgo | {', '.join(f'{k}: {v:.1%}' for k, v in t4.items())} |")

# monotonicidad
mono_saldo = t.iloc[0] > t.iloc[1] > t.iloc[2] > t.iloc[3] if not t.isna().any() else False
mono_pago = t2["mora_severa"] > t2["al_dia"] and t2["mora_leve"] > t2["al_dia"]
mono_monto = t3.iloc[-1] > t3.iloc[0]
mono_plazo = t4.iloc[-1] > t4.iloc[0]
report.append(f"\n- **Verificación de monotonicidad:** saldo {'OK' if mono_saldo else 'REVISAR'}, "
              f"historial de pago {'OK' if mono_pago else 'REVISAR'}, "
              f"monto {'OK' if mono_monto else 'REVISAR'}, "
              f"plazo {'OK' if mono_plazo else 'REVISAR'}.\n")

# 3. Comparación de forma de distribuciones numéricas (coef. de variación, asimetría)
report.append("## 3. Comparación de forma de distribuciones (simulado vs. original)\n")
report.append("| Variable | Media (sim.) | CV (sim.) | Asimetría (sim.) | CV (original) | Asimetría (original) |")
report.append("|---|---|---|---|---|---|")

def stats(s):
    return s.mean(), s.std() / s.mean(), s.skew()

m_sim, cv_sim, sk_sim = stats(df["edad"])
m_o, cv_o, sk_o = stats(orig["age"])
report.append(f"| Edad | {m_sim:.1f} | {cv_sim:.2f} | {sk_sim:.2f} | {cv_o:.2f} | {sk_o:.2f} |")

m_sim, cv_sim, sk_sim = stats(df["plazo_meses"])
m_o, cv_o, sk_o = stats(orig["duration"])
report.append(f"| Plazo/Duration | {m_sim:.1f} meses | {cv_sim:.2f} | {sk_sim:.2f} | {cv_o:.2f} | {sk_o:.2f} |")

m_sim, cv_sim, sk_sim = stats(df["monto_credito_cop"])
m_o, cv_o, sk_o = stats(orig["credit_amount"])
report.append(f"| Monto/Credit amount | ${m_sim:,.0f} COP | {cv_sim:.2f} | {sk_sim:.2f} | {cv_o:.2f} (DM) | {sk_o:.2f} |")

report.append("\nEl coeficiente de variación (CV) y la asimetría de las variables numéricas simuladas "
              "reproducen la forma general (sesgo a la derecha, dispersión relativa) de las variables "
              "análogas del German Credit Dataset, ajustadas a la escala monetaria colombiana (COP).\n")

# 4. Proporciones de categorías clave vs. las especificadas en la Primera entrega
report.append("## 4. Proporciones de categorías frente a lo especificado\n")
report.append("| Variable | Categoría | Proporción simulada |")
report.append("|---|---|---|")
for col in ["tipo_ocupacion", "tipo_vivienda", "nivel_ahorro", "saldo_categoria", "estado_pago_previo"]:
    props = df[col].value_counts(normalize=True)
    for cat, p in props.items():
        report.append(f"| {col} | {cat} | {p:.1%} |")

with open(f"{OUT_DIR}/resumen_validacion.md", "w") as f:
    f.write("\n".join(report))

# --- Gráficas ---
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
prop_sim.plot(kind="bar", ax=ax[0], color=["#4C78A8", "#E45756"])
ax[0].set_title("Simulado (n={:,})".format(len(df)))
ax[0].set_ylabel("Proporción")
ax[0].set_ylim(0, 1)
prop_orig.plot(kind="bar", ax=ax[1], color=["#4C78A8", "#E45756"])
ax[1].set_title("German Credit original (n=1000)")
ax[1].set_ylim(0, 1)
fig.suptitle("Balance de la variable objetivo: riesgo crediticio")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_balance_clase.png", dpi=140)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].hist(df["edad"], bins=30, color="#4C78A8", alpha=0.8)
axes[0].set_title("Edad (simulado)")
axes[1].hist(df["plazo_meses"], bins=30, color="#72B7B2", alpha=0.8)
axes[1].set_title("Plazo en meses (simulado)")
axes[2].hist(df["monto_credito_cop"] / 1e6, bins=40, color="#F58518", alpha=0.8)
axes[2].set_title("Monto del crédito (millones COP)")
fig.suptitle("Distribuciones de variables numéricas — base simulada")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_distribuciones.png", dpi=140)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t.plot(kind="bar", ax=axes[0], color="#E45756")
axes[0].set_title("Tasa de riesgo 'malo' según saldo en cuenta")
axes[0].set_ylabel("% riesgo malo")
t2.plot(kind="bar", ax=axes[1], color="#E45756")
axes[1].set_title("Tasa de riesgo 'malo' según historial de pago previo")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_relaciones_riesgo.png", dpi=140)
plt.close(fig)

print("Validación completa. Archivos generados:")
print(" - resumen_validacion.md")
print(" - fig_balance_clase.png")
print(" - fig_distribuciones.png")
print(" - fig_relaciones_riesgo.png")
print("\nProporción de 'malo' simulada:", prop_sim["malo"])
