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
notebooks/      Proceso completo, en orden (01 -> 06).
src/            Scripts y módulos reproducibles (.py) que usan los notebooks.
models/         Modelo final serializado (lo genera src/modelo_final.py).
app/            Aplicación de Streamlit (despliegue básico) y CSV de ejemplo.
docs/           Documentación de referencia, validación de la simulación,
                resultados de modelamiento (docs/resultados/) y capturas
                del despliegue (docs/despliegue/).
informe/        Informes de la Segunda entrega y de la Entrega final (PDF,
                fuente HTML y scripts que los generan).
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
| `06_Modelo_final_interpretacion.ipynb` | 8.1 a 8.3 | Selección y justificación del modelo final (desempeño, estabilidad e interpretabilidad), comparación pareada con la regresión logística por bootstrap, evaluación en test, calibración de Platt frente a la probabilidad verdadera, SHAP a nivel de las 11 variables contrastado con la estructura de la simulación, dependencia parcial e ICE, LIME frente a SHAP en casos individuales, tipos de error, equidad por género, sensibilidad a la matriz de costos y verificación de coherencia de datos de uso. |

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
- `modelo_final.py`: entrena el modelo final (SVM RBF) con el mismo
  procedimiento evaluado en la validación anidada, ajusta la calibración de
  Platt y guarda `models/modelo_final.joblib` con el pipeline, el umbral, el
  perfil de los datos de entrenamiento y la muestra de fondo para SHAP.
- `interpretacion.py`: envoltura del modelo final (puntaje, probabilidad y
  decisión), explicaciones SHAP a nivel de variable original y verificación
  de coherencia de los datos de uso frente a entrenamiento (esquema,
  categorías, rangos, reglas de negocio y deriva por PSI). Lo usan el
  notebook 06 y la aplicación.
- `preprocesamiento.py`: partición 80/20 y `Pipeline` compartido
  (imputación, escalamiento, balanceo opcional, codificación y selección de
  variables opcional por LASSO). Todos los notebooks de modelamiento lo
  importan, para que las transformaciones sean idénticas entre modelos.

### `informe/`

- `Informe_Segunda_Entrega.pdf`: informe de la Segunda entrega (secciones 7.1
  a 7.4 de la guía).
- `informe.html`: fuente editable del informe.
- `generar_informe.py`: reúne las figuras desde `docs/` y el notebook 04 y
  convierte el HTML a PDF con Microsoft Edge o Chrome.

## Cómo reproducir

Desde la carpeta que contiene este README:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python src/generar_datos_simulados.py   # regenera data/simulado/
python src/validar_simulacion.py        # verifica la simulación

# Ejecuta los notebooks en orden (el 04 tarda unos 15 minutos y el 05 unos 20)
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb

python informe/generar_informe.py       # regenera el PDF de la Segunda entrega

python src/modelo_final.py              # entrena y guarda el modelo final (~20 s)
python informe/generar_informe_final.py # regenera el PDF de la Entrega final
```

El notebook 06 tarda unos 10 minutos (casi todo es el cálculo de SHAP).

## Aplicación (despliegue básico)

```bash
python src/modelo_final.py      # solo si models/modelo_final.joblib no existe
streamlit run app/app.py        # abre http://localhost:8501
```

La aplicación tiene tres pestañas:

- **Evaluar una solicitud**: formulario con las 11 variables. Verifica los
  datos, aplica el `Pipeline` completo y devuelve la decisión (umbral de costo
  mínimo 5:1), la probabilidad calibrada de "malo" y la explicación SHAP de la
  predicción.
- **Evaluar un lote (CSV)**: verifica un archivo completo frente al perfil de
  entrenamiento (esquema, categorías, rangos, contradicciones y deriva por
  PSI), excluye las filas con errores, puntúa el resto y permite descargar los
  resultados y explicar cualquier fila. `app/ejemplo_solicitudes.csv` trae 200
  solicitudes de la partición de prueba y 4 con errores de captura.
- **Acerca del modelo**: flujo de cada predicción y límites de uso.

`models/modelo_final.joblib` se generó con las versiones de `requirements.txt`
(salvo NumPy 2.5 y Python 3.13). Si al abrir la aplicación aparece un error al
cargar el modelo, basta con regenerarlo con `python src/modelo_final.py`: el
entrenamiento es determinista (semilla 42) y reproduce las mismas cifras.

Es un prototipo académico entrenado con datos simulados: no debe usarse para
decisiones reales de crédito.

Conviene ejecutar un solo notebook a la vez: el 04 y el 05 usan todos los
núcleos del procesador (`n_jobs=-1`), y en Windows dos ejecuciones simultáneas
pueden bloquearse entre sí.

Para usar el `Pipeline` compartido en un notebook nuevo:

```python
import sys; sys.path.append("../src")
from preprocesamiento import cargar_particion, construir_pipeline
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = cargar_particion()
pipe = construir_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42))
```

## Uso de inteligencia artificial generativa

Conforme a la guía del proyecto, el equipo declara que utilizó herramientas de
IA generativa como apoyo en la revisión del código, la depuración y la
redacción de la documentación. Todas las decisiones metodológicas fueron
revisadas por el equipo, que puede sustentarlas.
