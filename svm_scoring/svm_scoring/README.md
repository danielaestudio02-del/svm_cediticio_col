# Mejora de los modelos de calificación crediticia mediante SVM

Proyecto integrador — Machine Learning II — Universidad Externado de Colombia

**Artículo base:** Xu, J. & Cheng, Y. — *Credit Scoring Models Enhancement Using
Support Vector Machines*.

## Estructura del repositorio

```
data/
  original/     Base original del artículo (German Credit Dataset - Statlog).
  simulado/     Base simulada (contexto colombiano), generada a partir de la
                estructura de la base original.
notebooks/      Notebook con el proceso completo: análisis de la base
                original -> diseño -> generación -> validación.
src/            Scripts reproducibles (.py) usados por el notebook.
docs/           Documentación de referencia (razones y paso a paso de cada
                fase del proyecto).
```

**Para entender por qué y cómo se construyó la base simulada**, ver
[`docs/construccion_base_datos.md`](docs/construccion_base_datos.md) —
documenta la motivación, la metodología y el paso a paso, como complemento al
código del notebook.

### `data/original/`

- `german.data` — 1.000 registros, 20 atributos + variable objetivo (formato
  categórico/simbólico, separado por espacios, sin encabezado).
- `german.data-numeric` — misma base, versión numérica (24 atributos)
  preparada por Strathclyde University para algoritmos que requieren
  variables numéricas.
- `german.doc` — documentación oficial de los atributos (códigos `A_ _`),
  incluida la matriz de costos (falso negativo = 5x más costoso que falso
  positivo).
- `Index` — índice de archivos original del repositorio UCI.
- **Fuente:** Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie,
  Universität Hamburg (UCI Machine Learning Repository).

### `data/simulado/`

- `credito_simulado_base.csv` — 6.500 registros, sin valores faltantes.
  Base de referencia para verificar que la estructura (distribuciones,
  relaciones, balance de clases) es correcta antes de introducir defectos.
- `credito_simulado_con_faltantes.csv` — misma base con 3%-5% de valores
  faltantes y ruido controlado en variables numéricas. **Esta es la base de
  trabajo para el flujo de modelamiento** (imputación, pipeline, entrenamiento).

### `notebooks/`

- `Base_de_datos_simulada.ipynb` — proceso completo y documentado: análisis
  exploratorio de la base original, diseño de la simulación, generación
  variable por variable, construcción de la variable objetivo, ensamble,
  versión con faltantes, y validación (incluida comparación visual directa
  original vs. simulado).

### `src/`

- `generar_datos_simulados.py` — script que genera los dos CSV de
  `data/simulado/` (semilla fija = 42, reproducible).
- `validar_simulacion.py` — script de validación (genera
  `resumen_validacion.md` y las figuras de verificación).

## Próximos pasos

Pipeline de preprocesamiento (imputación, codificación, escalamiento) y
entrenamiento/comparación de modelos (SVM lineal, SVM-RBF, Random Forest,
Gradient Boosting) frente a las líneas base, según el plan definido en la
Primera entrega.
