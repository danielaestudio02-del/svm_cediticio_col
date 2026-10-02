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
notebooks/      Proceso completo, en orden (01 -> 05).
src/            Scripts y módulos reproducibles (.py) que usan los notebooks.
docs/           Documentación de referencia, validación de la simulación y
                resultados de modelamiento (docs/resultados/).
requirements.txt  Dependencias con versiones fijas.
```

**Para entender por qué y cómo se construyó la base simulada**, ver
[`docs/construccion_base_datos.md`](docs/construccion_base_datos.md), que
documenta la motivación, la metodología, el paso a paso y las limitaciones,
como complemento al código de los notebooks.

### `data/original/`

- `german.data`: 1.000 registros, 20 atributos + variable objetivo (formato
  categórico/simbólico, separado por espacios, sin encabezado).
- `german.data-numeric`: misma base, versión numérica (24 atributos)
  preparada por Strathclyde University para algoritmos que requieren
  variables numéricas.
- `german.doc`: documentación oficial de los atributos (códigos `A_ _`),
  incluida la matriz de costos (falso negativo = 5 veces más costoso que un
  falso positivo).
- `Index`: índice de archivos original del repositorio UCI.
- **Fuente:** Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie,
  Universität Hamburg (UCI Machine Learning Repository).

### `data/simulado/`

- `credito_simulado_base.csv`: 6.500 registros, sin valores faltantes, con
  22.5% de riesgo "malo". Base de control para verificar que la estructura
  (distribuciones, relaciones, balance de clases) es correcta antes de
  introducir defectos.
- `credito_simulado_con_faltantes.csv`: misma base con 3%-5% de valores
  faltantes por columna y ruido leve de medición. **Esta es la base de trabajo
  para el modelamiento** (imputación, pipeline, entrenamiento).

### `notebooks/`

| Notebook | Sección de la guía | Contenido |
|---|---|---|
| `01_Base_de_datos_simulada.ipynb` | 7.1 | Análisis de la base original, diseño de la simulación, generación variable por variable, variable objetivo y validación frente al original. |
| `02_EDA_y_calidad_datos.ipynb` | 7.1 | EDA de la base simulada, verificación de lo planeado y calidad de la base con faltantes (transformaciones necesarias). |
| `03_Pipeline_particion_lineas_base.ipynb` | 7.2 | Partición estratificada, `Pipeline` sin fuga, comparación de estrategias de balanceo y líneas base (mayoría y regresión logística). |
| `04_SVM_reproduccion_metodo.ipynb` | 7.3 | Reproducción del SVM sobre el German Credit (75.5% de accuracy frente al 75% del artículo), adaptación a la base simulada con ajuste de `C` y `γ` por validación cruzada, comparación de kernels (lineal, gaussiano, polinómico y sigmoide), curvas de validación, fronteras de decisión, coeficientes del SVM lineal y justificación de la adaptación. |
| `05_Ensambles_validacion_anidada.ipynb` | 7.3 y 7.4 | Árbol de decisión, Random Forest, Gradient Boosting y SVM RBF con selección de variables por LASSO, con sus hiperparámetros; validación cruzada anidada (5 folds externos × 3 internos) de todos los modelos frente a las líneas base, con F1, ROC-AUC y recall de "malo" (media ± desviación; accuracy solo como referencia); prueba t corregida; evaluación en test, matrices de confusión y análisis de errores; importancia por permutación del mejor modelo, su estabilidad entre folds y su contraste con la estructura verdadera de la simulación. |

### `src/`

- `generar_datos_simulados.py`: genera los dos CSV de `data/simulado/`
  (semilla fija = 42, reproducible). Expone también los pesos del puntaje
  latente y la probabilidad teórica `P(malo | x)` usada como techo en el
  notebook 04.
- `validar_simulacion.py`: verifica 14 criterios de aceptación de la
  simulación y genera `docs/validacion/resumen_validacion.md` y sus figuras.
  Termina con error si algún criterio no se cumple.
- `evaluacion.py`: costo 5:1, umbral de costo mínimo, prueba t corregida de
  Nadeau y Bengio y la validación cruzada anidada (`validacion_anidada`) que
  usa el notebook 05. En cada fold externo, la búsqueda de hiperparámetros y el
  umbral se eligen solo con la parte de entrenamiento.
- `preprocesamiento.py`: partición 80/20 y `Pipeline` compartido
  (imputación, escalamiento, balanceo opcional, codificación y selección de
  variables opcional por LASSO). Todos los
  notebooks de modelamiento lo importan, para que las transformaciones sean
  idénticas entre modelos.

## Cómo reproducir

Desde la carpeta que contiene este README:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python src/generar_datos_simulados.py   # regenera data/simulado/
python src/validar_simulacion.py        # verifica la simulación

# Ejecuta los notebooks en orden (el 04 y el 05 tardan unos 10-15 minutos cada uno)
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Para usar el `Pipeline` compartido en un notebook nuevo:

```python
import sys; sys.path.append("../src")
from preprocesamiento import cargar_particion, construir_pipeline
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = cargar_particion()
pipe = construir_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42))
```

## Próximos pasos

- Informe de la Segunda entrega: consolidar las tablas y figuras de
  `docs/resultados/` (generadas por el notebook 05).
- Explorar si un ajuste del umbral por segmento (p. ej. por historial de pago)
  reduce el costo sin sacrificar equidad, y revisar la calibración de las
  probabilidades del modelo elegido.

## Uso de inteligencia artificial generativa

Conforme a la guía del proyecto, el equipo declara que utilizó herramientas de
IA generativa como apoyo en la revisión del código, la depuración y la
redacción de la documentación. Todas las decisiones metodológicas fueron
revisadas por el equipo, que puede sustentarlas.
