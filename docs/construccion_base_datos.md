# Construcción de la base de datos simulada

Documento de referencia para la Sección 7.1 (Consolidación de los datos) de la
Segunda entrega. Complementa el notebook `notebooks/01_Base_de_datos_simulada.ipynb`,
que contiene el código ejecutado y sus resultados; aquí se documenta el **por
qué** de cada decisión, para no tener que rastrearlo celda por celda.

---

## 1. Contexto y motivación

El proyecto reproduce y adapta el método de Xu & Cheng (*Credit Scoring Models
Enhancement Using Support Vector Machines*), que usa el German Credit Dataset
(Statlog) para clasificar solicitantes de crédito en riesgo bueno/malo
mediante SVM.

**¿Por qué simular en vez de usar un dataset real colombiano?**
La información crediticia individual en Colombia (score de centrales de
riesgo, historial de pagos, ingresos declarados) está protegida por el
habeas data financiero (Ley 1266 de 2008) y no está disponible en
repositorios públicos con el nivel de detalle por individuo que exige este
proyecto. Esta restricción no es exclusiva de Colombia: la mayoría de países
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
   distribución de cada variable, proporciones de las categorías, y la
   relación entre cada variable y el riesgo (tasa de "malo" por categoría).

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
   una matriz de correlación (cópulas): solo se tiene una base de referencia
   de otro contexto para calibrar forma y dirección.

---

## 3. Paso a paso de la construcción

### Paso 1. Análisis exploratorio de la base original
Se cargó `german.data` (1.000 registros, 20 atributos + objetivo) y se
calcularon: balance de clases (70% bueno / 30% malo), estadísticas de las
variables numéricas (`age`, `duration`, `credit_amount`: media, desviación,
asimetría y correlación), y proporciones de las variables categóricas clave
(`checking`, `savings`, `credit_history`, `housing`, `job`). También se cruzó
cada categórica contra el riesgo.

**Hallazgo relevante.** En el German Credit, dos relaciones van en contra de
la intuición de scoring:

- "No tener cuenta corriente" (`A14`) tiene la *menor* tasa de "malo" (11.7%).
- En el historial de crédito, `A30` y `A31` (sin créditos o todos pagados a
  tiempo) tienen las tasas más altas (62.5% y 57.1%) y `A34` (cuenta crítica)
  la más baja (17.1%).

Estas relaciones no son solo un artefacto del cruce univariado: el SVM lineal
de nuestra reproducción sobre el German Credit (notebook 04, Parte A), que
controla por las demás variables, también asigna coeficientes negativos
(menor riesgo) a `A14` y a `A34`. En la Primera entrega la dirección "no tener
cuenta aumenta el riesgo" se justificó con nuestra lectura de los coeficientes
del SVM lineal de los autores; como nuestra propia estimación da el signo
contrario, la dirección de estas variables en la simulación **no se apoya en el
dataset alemán**: es una **decisión explícita de adaptación al contexto
colombiano**. En Colombia, no tener cuenta bancaria se
asume asociado a informalidad y a ausencia de historial en centrales de riesgo,
y la mora previa es el predictor clásico de riesgo; ambos supuestos se
documentan como tales. Del original se toman las **proporciones** de cada
categoría, no la dirección de su tasa de mora.

### Paso 2. Diseño: qué variable se calibra contra qué evidencia

| Bloque | Variables | Fuente de calibración |
|---|---|---|
| Sociodemográficas | edad, género, tipo de ocupación | media, desviación y asimetría de `age`; informalidad laboral colombiana como supuesto |
| Patrimoniales | tipo de vivienda, nivel de ahorro, saldo en cuenta (COP) | proporciones de `savings` y `checking`; vivienda como supuesto colombiano |
| Crédito solicitado | monto (COP), plazo (meses), propósito | forma de `duration` y `credit_amount` y su correlación (0.62) |
| Comportamiento previo | estado de pago de créditos anteriores | proporciones de `credit_history` |
| Objetivo | riesgo (bueno/malo) | función logística de las variables anteriores, calibrada a 20%-25% de "malo" |

**Parámetros generales:** N = 6.500 registros (> mínimo de 5.000 exigido),
semilla = 42 (reproducibilidad exigida por la guía del proyecto).

Regla general aplicada en cada fila: si existe un análogo numérico o
categórico directo en `german.data`, se calibra la forma o las proporciones
contra esa columna; si no existe (por ejemplo, el tipo de ocupación, que no
tiene equivalente limpio en el original), se usa un supuesto de calibración
externo, documentado explícitamente como tal y nunca presentado como dato
verificado.

### Paso 3. Generación variable por variable
Cada bloque se generó con una distribución apropiada a su forma real:

- **Edad**: Gamma(2.4, 7.3) desplazada 18 años y truncada en [18, 75]. Da
  media 35.5, desviación 11.0 y asimetría 1.1, frente a 35.5 / 11.4 / 1.0 de
  `age`.
- **Plazo**: Gamma(2.2, 7.7) desplazada 4 meses, redondeada a múltiplos de 3
  meses y acotada en [3, 72]. Da media 20.9 y desviación 11.4, frente a
  20.9 / 12.1 de `duration`.
- **Monto del crédito**: log-normal generada **como función del plazo y del
  propósito**, no de forma independiente:
  `log(monto) = 15.2 + shift(propósito) + 0.95·log(plazo/21) + ε`,
  con `ε ~ N(0, 0.47²)`. La elasticidad de 0.95 reproduce la correlación
  plazo-monto del original (0.60 simulada frente a 0.62) y σ = 0.47 su
  dispersión relativa (CV 0.83 frente a 0.86).
- **Variables categóricas** (ocupación, vivienda, ahorro, saldo, propósito,
  historial de pago): muestreo categórico ponderado.
  - `saldo_categoria`: `A14`/`A11`/`A12`/`A13` → sin_cuenta/bajo/medio/alto
    (39/27/27/7%).
  - `nivel_ahorro`: `A61`-`A64` → bajo..muy_alto, con el 18.3% de `A65`
    (desconocido) repartido proporcionalmente (74/13/7/6%).
  - `estado_pago_previo`: `A30`+`A31` → sin_creditos_previos, `A32` → al_dia,
    `A33` → mora_leve, `A34` → mora_severa (9/53/9/29%).
  - `tipo_vivienda` (38% propia, 42% arrendada) y `tipo_ocupacion` (44%
    informal): supuestos del contexto colombiano.
- **Saldo en cuenta (COP)**: 0 para `sin_cuenta` y log-normal dentro de cada
  tramo para los demás, coherente con `saldo_categoria`.

### Paso 4. Construcción de la variable objetivo
El riesgo no se asignó al azar: se construyó un puntaje de riesgo latente
(`z`) como combinación ponderada de todas las variables anteriores, con signos
que respetan las relaciones documentadas en la Primera entrega (no tener
cuenta → mayor riesgo; buen historial → menor riesgo; mayor monto/plazo →
mayor riesgo), más un término de ruido idiosincrático `N(0, 0.85²)`. En el
historial de pago, el orden de riesgo es al día < sin créditos previos
(historial delgado) < mora leve < mora severa. Los pesos están en las
constantes `PESOS_CATEGORICOS` y `PESOS_NUMERICOS` del script. El intercepto
de la función logística se calibró por bisección hasta lograr una prevalencia
de "malo" de 22.5%, el punto medio del rango 20%-25% comprometido en la
Primera entrega.

### Paso 5. Ensamble y dos versiones de la base
- `credito_simulado_base.csv`: los 6.500 registros ensamblados, **sin
  valores faltantes**. Sirve como base de control para validar que la
  estructura es correcta antes de introducir defectos.
- `credito_simulado_con_faltantes.csv`: la misma base con faltantes y ruido
  leve. **Es la base de trabajo** para el flujo de modelamiento (imputación,
  pipeline, entrenamiento), tal como se comprometió en la Primera entrega:
  - Faltantes de 3% a 5% por columna en edad, plazo, monto, saldo, ocupación
    y nivel de ahorro (22% de las filas tiene al menos uno).
  - El saldo tiene faltantes **MAR**: solo pueden faltar en clientes que sí
    tienen cuenta (para `sin_cuenta` el saldo es 0 por definición) y son más
    probables en saldos bajos (12.1% frente a 3.5%), simulando subreporte.
    Las demás columnas tienen faltantes MCAR (completamente al azar).
  - Ruido gaussiano del 2% del valor en ~3% de los registros de edad, saldo y
    monto (error de reporte). El plazo no lleva ruido porque es un término
    contractual, no un dato reportado.

### Paso 6. Validación
`src/validar_simulacion.py` verifica 14 criterios de aceptación y genera
`docs/validacion/resumen_validacion.md` con las tablas y figuras. Todos se
cumplen:

1. **Tamaño y balance de clases**: 6.500 registros y 22.5% de "malo", dentro
   del rango 20%-25%.
2. **Monotonicidad**: la tasa de "malo" crece estrictamente en el orden de
   riesgo esperado en saldo en cuenta, historial de pago, nivel de ahorro,
   tipo de ocupación y cuartiles de monto y de plazo.
3. **Forma de las distribuciones**: asimetría positiva y coeficiente de
   variación a menos de 0.10 del original en edad, plazo y monto.
4. **Estructura de dependencia**: correlación plazo-monto a menos de 0.05 de
   la del original.
5. **Faltantes**: entre 2.5% y 5.5% por columna, sin saldos faltantes en
   clientes sin cuenta.

Además, los notebooks 01 y 02 muestran la comparación visual directa
(histogramas superpuestos y proporciones por categoría lado a lado), y el
notebook 01 verifica que su base es idéntica a la del script.

---

## 4. Limitaciones conocidas de este proceso

- **Circularidad en la etiqueta de riesgo.** La variable objetivo se
  construyó como función de las mismas variables que luego se usan como
  predictores, más ruido. Un modelo entrenado sobre esta base aprende a
  invertir esa función, no necesariamente un comportamiento real. Como
  conocemos la estructura generadora, el notebook 04 cuantifica el efecto: el
  ROC-AUC máximo alcanzable con estas variables (usando la probabilidad
  verdadera `P(malo | x)`) es de ~0.78, y la regresión logística y los SVM ya
  se acercan a ese techo. Por eso, en esta base las diferencias entre modelos
  están acotadas por construcción, y un AUC alto **no** es evidencia de que el
  modelo funcionaría igual con datos reales.
- **Relación casi lineal en el logit.** El puntaje latente es aditivo, con una
  única no linealidad leve (la forma en U de la edad). Esto favorece a los
  modelos lineales y limita la ventaja que pueden mostrar el SVM RBF o los
  ensambles sobre esta base.
- **Dirección de `checking` y `credit_history`.** Es una decisión de
  adaptación al contexto colombiano que contradice lo que se observa en el
  German Credit (Paso 1). Si el contexto colombiano se pareciera al alemán en
  este punto, la simulación exageraría el riesgo de no tener cuenta.
- **Cola del monto.** La log-normal reproduce la media relativa, la
  dispersión y la correlación con el plazo, pero su asimetría (2.8) es mayor
  que la del original (1.95).
- **Plazos agrupados.** El plazo se genera con una distribución continua
  redondeada a múltiplos de 3 meses, mientras que en la práctica bancaria real
  los plazos suelen ser pocos valores preestablecidos (6, 12, 24, 36 meses).
- **Mapeo aproximado en algunas comparaciones categóricas.** La comparación de
  "saldo en cuenta" contra `checking` usa un mapeo por orden conceptual, no una
  correspondencia literal, porque las escalas monetarias (DM vs. COP) no
  tienen un umbral de conversión directo.
- **Supuestos de calibración sin verificación dato a dato.** Proporciones
  como la informalidad laboral (44%), la tenencia de vivienda o el balance de
  cartera (20%-25%) son supuestos razonables basados en cifras agregadas, no
  verificados registro por registro como sí lo estaría un dataset real.

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
