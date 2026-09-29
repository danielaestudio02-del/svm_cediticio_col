# Construcción de la base de datos simulada

Documento de referencia para la Sección 7.1 (Consolidación de los datos) de la
Segunda entrega. Complementa el notebook `notebooks/Base_de_datos_simulada.ipynb`,
que contiene el código ejecutado y sus resultados; aquí se documenta el **por
qué** de cada decisión, para no tener que rastrearlo celda por celda.

---

## 1. Contexto y motivación

El proyecto reproduce y adapta el método de Xu & Cheng (*Credit Scoring Models
Enhancement Using Support Vector Machines*), que usa el German Credit Dataset
(Statlog) para clasificar solicitantes de crédito en riesgo bueno/malo
mediante SVM con kernel RBF.

**¿Por qué simular en vez de usar un dataset real colombiano?**
La información crediticia individual en Colombia (score de centrales de
riesgo, historial de pagos, ingresos declarados) está protegida por el
habeas data financiero (Ley 1266 de 2008) y no está disponible en
repositorios públicos con el nivel de detalle por individuo que exige este
proyecto. Esta restricción no es exclusiva de Colombia — la mayoría de países
de la región (México, Perú, Chile, Argentina, Brasil) tienen marcos
normativos equivalentes. Por eso el equipo optó por una **simulación
fundamentada**: un conjunto de datos generado por el equipo que reproduce
casos realistas del fenómeno, basándose en la estructura documentada en el
artículo y en el propio German Credit Dataset.

**¿Por qué "fundamentada" y no simplemente aleatoria?**
Simular bien no es generar números al azar con una forma bonita. Exige
comprender la **estructura generadora de los datos**: qué distribución sigue
cada variable, cómo se relacionan entre sí, y cómo esas variables determinan
el resultado (riesgo). Sin esto, un modelo entrenado sobre la simulación no
estaría aprendiendo nada parecido a un problema real de riesgo crediticio.

---

## 2. Metodología general

La construcción siguió tres pasos, en este orden:

1. **Entender la base de referencia.** Antes de simular, se analizó
   estadísticamente el German Credit Dataset original (`data/original/`):
   distribución de cada variable, proporciones de las categorías, y —lo más
   importante— la relación entre cada variable y el riesgo (tasa de "malo"
   por categoría).

2. **Decidir qué preservar de esa estructura:**
   - *Distribuciones marginales*: la forma de cada variable (p. ej., que el
     monto del crédito tenga una cola larga a la derecha).
   - *Estructura de dependencia*: cómo se relacionan las variables entre sí
     (p. ej., créditos más largos tienden a ser de mayor monto).
   - *Relación predictores → objetivo*: cómo cada variable afecta la
     probabilidad de que el riesgo sea "malo".

3. **Elegir el método de generación.** Se usó **simulación paramétrica
   (Monte Carlo)**: se postula una distribución por variable, se ajustan sus
   parámetros contra la base de referencia, y la dependencia (incluida la
   relación con el objetivo) se construye explícitamente combinando las
   variables. Es el método apropiado cuando no se dispone de datos reales
   locales de los cuales entrenar un modelo generativo (GAN, SDV) o estimar
   una matriz de correlación (cópulas) — solo se tiene una base de
   referencia de otro contexto para calibrar forma y dirección.

---

## 3. Paso a paso de la construcción

### Paso 1 — Análisis exploratorio de la base original
Se cargó `german.data` (1.000 registros, 20 atributos + objetivo) y se
calcularon: balance de clases (70% bueno / 30% malo), estadísticas de las
variables numéricas (`age`, `duration`, `credit_amount`: media, asimetría), y
proporciones de las variables categóricas clave (`checking`, `savings`,
`credit_history`, `housing`, `job`). También se cruzó cada categórica contra
el riesgo, para tener evidencia de la dirección de cada relación.

**Hallazgo relevante:** en el cruce univariado, "no tener cuenta corriente"
(`A14`) tiene la *menor* tasa de mora, no la mayor — un efecto de confusión
típico de este dataset (variables no observadas, como edad o antigüedad
laboral, mezcladas en ese cruce simple). Por eso la dirección de las
relaciones simuladas se basó en los **coeficientes reportados por los
autores del artículo** (que sí controlan por las demás variables), no en
estos cruces univariados de una sola variable a la vez.

### Paso 2 — Diseño: qué variable se calibra contra qué evidencia

| Bloque | Variables | Fuente de calibración |
|---|---|---|
| Sociodemográficas | edad, género, tipo de ocupación | forma de `age`; informalidad laboral colombiana como supuesto |
| Patrimoniales | tipo de vivienda, nivel de ahorro, saldo en cuenta (COP) | proporciones de `housing`, `savings`, `checking` |
| Crédito solicitado | monto (COP), plazo (meses), propósito | forma y correlación de `credit_amount` y `duration` |
| Comportamiento previo | estado de pago de créditos anteriores | proporciones de `credit_history` |
| Objetivo | riesgo (bueno/malo) | función logística de las variables anteriores, calibrada a 20%-25% de "malo" |

**Parámetros generales:** N = 6.500 registros (> mínimo de 5.000 exigido),
semilla = 42 (reproducibilidad exigida por la guía del proyecto).

Regla general aplicada en cada fila: si existe un análogo numérico o
categórico directo en `german.data`, se calibra la forma/proporciones contra
esa columna; si no existe (por ejemplo, tipo de ocupación, que no tiene
equivalente limpio en el original), se usa un supuesto de calibración
externo, documentado explícitamente como tal — nunca presentado como dato
verificado.

### Paso 3 — Generación variable por variable
Cada bloque se generó con una distribución apropiada a su forma real:
- **Edad y plazo**: distribución Gamma (naturalmente asimétrica a la
  derecha), ajustada para que la media replique la de `age`/`duration`.
- **Monto del crédito**: log-normal, generado **como función del plazo y el
  propósito** (no de forma independiente), para preservar la correlación
  monto-plazo (~0.62) observada en el original.
- **Variables categóricas** (ocupación, vivienda, ahorro, saldo, propósito,
  historial de pago): muestreo categórico ponderado, con proporciones
  calibradas contra el original o contra el supuesto documentado.

### Paso 4 — Construcción de la variable objetivo
El riesgo no se asignó al azar: se construyó un puntaje de riesgo latente
(`z`) como combinación ponderada de todas las variables anteriores, con
signos que respetan las relaciones documentadas en la Primera entrega (no
tener cuenta → mayor riesgo; buen historial → menor riesgo; mayor monto/plazo
→ mayor riesgo) más un término de ruido idiosincrático. El intercepto de la
función logística se calibró por bisección hasta lograr una prevalencia de
"malo" en el rango 20%-25% ya comprometido en la Primera entrega.

### Paso 5 — Ensamble y dos versiones de la base
- `credito_simulado_base.csv`: los 6.500 registros ensamblados, **sin
  valores faltantes** — sirve como base de control para validar que la
  estructura es correcta antes de introducir defectos.
- `credito_simulado_con_faltantes.csv`: la misma base con 3%-5% de valores
  faltantes (tipo MAR, más probables en saldos/montos bajos) y ruido leve en
  variables numéricas — **es la base de trabajo real** para el flujo de
  modelamiento (imputación, pipeline, entrenamiento), tal como se comprometió
  en la Primera entrega.

### Paso 6 — Validación
Se verificó que la simulación reproduce la estructura prevista en tres
frentes:
1. **Balance de clases**: 22.7% "malo" — dentro del rango 20%-25% fijado.
2. **Monotonicidad de las relaciones documentadas**: las cuatro relaciones
   clave (cuenta bancaria, historial de pago, monto, plazo) son monótonas en
   la dirección esperada.
3. **Forma de las distribuciones**: coeficiente de variación y asimetría de
   `edad`, `plazo` y `monto` comparables a sus análogas del original.
4. **Comparación visual directa**: histogramas superpuestos (edad, plazo) y
   estandarizados (monto), y proporciones por categoría lado a lado
   (vivienda, historial de pago, saldo, balance de clases) — ver notebook,
   Sección 8.

---

## 4. Limitaciones conocidas de este proceso

- **Circularidad en la etiqueta de riesgo.** La variable objetivo se
  construyó como función determinística (+ ruido) de las mismas variables
  que luego se usan como predictores. Esto significa que un modelo
  suficientemente flexible (SVM incluido) puede recuperar esa relación con
  métricas artificialmente altas — un AUC alto sobre esta base **no**
  constituye evidencia de que el modelo funcionaría igual de bien con datos
  reales, solo que logra invertir la función logística usada para generar
  el riesgo. El ruido idiosincrático incluido en `z` mitiga parcialmente
  este efecto, pero no lo elimina.
- **Plazos continuos vs. discretos.** El plazo simulado usa una distribución
  continua (redondeada a múltiplos de 3 meses), mientras que en la práctica
  bancaria real los plazos suelen ser valores discretos preestablecidos
  (6, 12, 24, 36 meses).
- **Mapeo aproximado en algunas comparaciones categóricas.** La comparación
  de "saldo en cuenta" contra `checking` (original) usa un mapeo aproximado
  por orden conceptual, no una correspondencia literal, porque las escalas
  monetarias (DM vs. COP) no tienen un umbral de conversión directo.
- **Supuestos de calibración sin verificación dato a dato.** Proporciones
  como la informalidad laboral (44%) o el desbalance de cartera colombiana
  (20%-25%) son supuestos razonables basados en cifras agregadas públicas,
  no verificados registro por registro como sí lo estaría un dataset real.

---

## 5. Referencias

- Xu, J. & Cheng, Y. — *Credit Scoring Models Enhancement Using Support
  Vector Machines*.
- Hofmann, H. — *Statlog (German Credit Data)*. UCI Machine Learning
  Repository, Institut für Statistik und Ökonometrie, Universität Hamburg.
- Ley 1266 de 2008 (Habeas Data Financiero), Colombia.
- Sobre la circularidad de etiquetas generadas por reglas en datos
  sintéticos: *Alternative-Data Credit Scoring for Migrant Workers: A
  Synthetic Data Simulation Study*, NHSJS.
- Sobre métodos de generación de datos sintéticos de crédito (Monte Carlo,
  agent-based, generativos): Open Risk Management —
  *Machine learning approaches to synthetic credit data*.
- Marco de validación de datos sintéticos (fidelidad / utilidad /
  privacidad): AWS Machine Learning Blog — *How to evaluate the quality of
  synthetic data*.
