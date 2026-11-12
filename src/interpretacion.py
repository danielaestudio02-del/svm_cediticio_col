"""
Interpretación y verificación de datos de uso (Entrega final, 8.2 y 8.3).

Reúne lo que comparten el notebook 06 y la aplicación de Streamlit:

- `ModeloFinal`: envoltura del artefacto guardado por `modelo_final.py`
  (pipeline, umbral de costo mínimo y calibración de Platt). Da el puntaje del
  SVM, la probabilidad calibrada de "malo" y la decisión.
- `crear_explicador`: valores SHAP a nivel de las 11 variables originales
  (no de las 32 columnas one-hot), para que la explicación se lea en los
  términos del formulario de solicitud.
- `perfil_datos` y `verificar_coherencia`: perfil de los datos de
  entrenamiento y comparación con los datos que llegan a la aplicación
  (esquema, categorías, rangos, reglas de negocio y deriva por PSI).
- `CorteMontoPlazo`: el mapa de decisión de la aplicación. Evalúa el modelo en
  el plano monto × plazo con las otras 9 variables fijas y busca el cambio de
  una sola de las dos que cambia la decisión.

Convención: 1 = "malo" (clase positiva). El puntaje del SVM es
`decision_function`: mayor = más riesgo, en unidades del margen.
"""
import numpy as np
import pandas as pd

from preprocesamiento import CAT_COLS, NUM_COLS

COLUMNAS = ["edad", "genero", "tipo_ocupacion", "tipo_vivienda", "nivel_ahorro",
            "saldo_categoria", "saldo_cuenta_cop", "monto_credito_cop", "plazo_meses",
            "proposito_credito", "estado_pago_previo"]

NOMBRES = {
    "edad": "Edad",
    "genero": "Género",
    "tipo_ocupacion": "Tipo de ocupación",
    "tipo_vivienda": "Tipo de vivienda",
    "nivel_ahorro": "Nivel de ahorro",
    "saldo_categoria": "Categoría de saldo",
    "saldo_cuenta_cop": "Saldo en cuenta (COP)",
    "monto_credito_cop": "Monto del crédito (COP)",
    "plazo_meses": "Plazo (meses)",
    "proposito_credito": "Propósito del crédito",
    "estado_pago_previo": "Historial de pago",
}

# Orden lógico de las categorías (de menor a mayor riesgo esperado o natural)
ORDEN_CATEGORIAS = {
    "genero": ["F", "M"],
    "tipo_ocupacion": ["alta_calificacion", "formal_calificado", "formal_no_calificado", "informal"],
    "tipo_vivienda": ["propia", "familiar", "arrendada"],
    "nivel_ahorro": ["muy_alto", "alto", "medio", "bajo"],
    "saldo_categoria": ["alto", "medio", "bajo", "sin_cuenta"],
    "proposito_credito": ["educacion", "vehiculo", "remodelacion", "electrodomesticos",
                          "otros", "libre_inversion", "negocio"],
    "estado_pago_previo": ["al_dia", "sin_creditos_previos", "mora_leve", "mora_severa"],
}


# ---------------------------------------------------------------------------
# 1. Modelo final
# ---------------------------------------------------------------------------
class ModeloFinal:
    """Envoltura del artefacto: puntaje, probabilidad calibrada y decisión."""

    def __init__(self, artefacto):
        self.artefacto = artefacto
        self.pipeline = artefacto["pipeline"]
        self.umbral = artefacto["umbral"]
        self.a, self.b = artefacto["calibrador"]

    def puntaje(self, X):
        return self.pipeline.decision_function(X[COLUMNAS])

    def probabilidad(self, X=None, puntaje=None):
        """P(malo) calibrada con Platt sobre puntajes fuera de fold de entrenamiento."""
        s = self.puntaje(X) if puntaje is None else np.asarray(puntaje)
        return 1 / (1 + np.exp(-(self.a * s + self.b)))

    def predecir(self, X):
        return (self.puntaje(X) >= self.umbral).astype(int)

    @property
    def umbral_probabilidad(self):
        return float(self.probabilidad(puntaje=self.umbral))


# ---------------------------------------------------------------------------
# 2. SHAP a nivel de variable original
# ---------------------------------------------------------------------------
class Codificador:
    """Convierte las 11 variables en una matriz numérica (categorías -> índice) y viceversa.

    SHAP reemplaza valores por los de una muestra de fondo, y para eso necesita
    una matriz numérica. Las categóricas se pasan como índice y se decodifican
    antes de llegar al pipeline; un faltante viaja como NaN y lo imputa el
    pipeline, igual que en producción.
    """

    def __init__(self, categorias):
        self.categorias = categorias

    def codificar(self, X):
        M = np.empty((len(X), len(COLUMNAS)))
        for j, c in enumerate(COLUMNAS):
            if c in CAT_COLS:
                mapa = {v: i for i, v in enumerate(self.categorias[c])}
                M[:, j] = X[c].map(mapa).astype(float).to_numpy()
            else:
                M[:, j] = X[c].astype(float).to_numpy()
        return M

    def decodificar(self, M):
        datos = {}
        for j, c in enumerate(COLUMNAS):
            if c in CAT_COLS:
                cats = np.asarray(self.categorias[c], dtype=object)
                col = M[:, j]
                salida = np.full(len(col), np.nan, dtype=object)
                ok = ~np.isnan(col)
                salida[ok] = cats[col[ok].astype(int)]
                datos[c] = salida
            else:
                datos[c] = M[:, j]
        return pd.DataFrame(datos, columns=COLUMNAS)


def crear_explicador(modelo, fondo, max_evals=2 * len(COLUMNAS) * 10 + 1, seed=42):
    """Explicador SHAP (algoritmo de permutación) del puntaje del SVM.

    - `fondo`: muestra de entrenamiento que define la "predicción de referencia".
    - Devuelve una función `explicar(X) -> shap.Explanation` cuyos valores suman,
      junto con el valor base, el puntaje (`decision_function`) de cada cliente.
    """
    import shap

    cod = Codificador(modelo.artefacto["perfil"]["categorias"])
    f = lambda M: modelo.puntaje(cod.decodificar(M))  # noqa: E731
    masker = shap.maskers.Independent(cod.codificar(fondo), max_samples=len(fondo))
    explicador = shap.Explainer(f, masker, algorithm="permutation",
                                feature_names=[NOMBRES[c] for c in COLUMNAS], seed=seed)

    def explicar(X):
        exp = explicador(cod.codificar(X), max_evals=max_evals, silent=True)
        exp.data = cod.codificar(X)
        exp.display_data = X[COLUMNAS].astype(object).to_numpy()
        return exp

    return explicar


def tabla_explicacion(exp_fila, X_fila):
    """Tabla legible de una explicación individual, ordenada por |SHAP|."""
    t = pd.DataFrame({
        "Variable": [NOMBRES[c] for c in COLUMNAS],
        "Valor": [_mostrar(X_fila[c]) for c in COLUMNAS],
        "Contribución": exp_fila.values,
    })
    t["Efecto"] = np.where(t["Contribución"] > 0, "aumenta el riesgo", "reduce el riesgo")
    return t.reindex(t["Contribución"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _miles(v):
    return f"{v:,.0f}".replace(",", ".")


def _mostrar(v):
    if isinstance(v, float) and np.isnan(v):
        return "(faltante)"
    if isinstance(v, (int, float, np.floating)):
        return f"{v:,.0f}".replace(",", ".")
    return str(v).replace("_", " ")


# ---------------------------------------------------------------------------
# 3. Perfil de entrenamiento y verificación de coherencia de los datos de uso
# ---------------------------------------------------------------------------
def perfil_datos(X):
    """Perfil de los datos de entrenamiento que se guarda junto al modelo."""
    perfil = {"n": len(X), "categorias": {}, "proporciones": {}, "numericas": {},
              "faltantes": X[COLUMNAS].isna().mean().round(4).to_dict(), "saldo_por_categoria": {}}
    for c in CAT_COLS:
        presentes = set(X[c].dropna().unique())
        orden = [v for v in ORDEN_CATEGORIAS[c] if v in presentes] + sorted(presentes - set(ORDEN_CATEGORIAS[c]))
        perfil["categorias"][c] = orden
        perfil["proporciones"][c] = X[c].value_counts(normalize=True).reindex(orden).fillna(0).round(4).to_dict()
    for c in NUM_COLS:
        v = X[c].dropna()
        cortes = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        perfil["numericas"][c] = {
            "min": float(v.min()), "max": float(v.max()), "mediana": float(v.median()),
            "p01": float(v.quantile(0.01)), "p99": float(v.quantile(0.99)),
            "cortes": cortes.tolist(),
            "proporciones": (np.histogram(v, bins=_bordes(cortes))[0] / len(v)).round(4).tolist(),
        }
    for cat, g in X.dropna(subset=["saldo_cuenta_cop"]).groupby("saldo_categoria"):
        s = g["saldo_cuenta_cop"]
        perfil["saldo_por_categoria"][cat] = {"min": float(s.min()), "max": float(s.max())}
    return perfil


def _bordes(cortes):
    b = np.asarray(cortes, dtype=float).copy()
    b[0], b[-1] = -np.inf, np.inf
    return b


def psi(esperado, observado, eps=1e-4):
    """Índice de estabilidad poblacional entre dos distribuciones (proporciones por tramo)."""
    e = np.clip(np.asarray(esperado, dtype=float), eps, None)
    o = np.clip(np.asarray(observado, dtype=float), eps, None)
    return float(np.sum((o - e) * np.log(o / e)))


def verificar_coherencia(X, perfil, deriva=True, min_filas_deriva=100):
    """Compara datos de uso con el perfil de entrenamiento.

    Devuelve un DataFrame de hallazgos con columnas `nivel` ("error",
    "advertencia" o "info"), `variable`, `filas` (afectadas) y `detalle`.
    - error: el modelo no puede o no debe puntuar esa fila (columna ausente,
      categoría desconocida, valor imposible o contradicción).
    - advertencia: fuera del rango visto en entrenamiento, faltantes o deriva:
      el modelo puntúa, pero extrapola.
    """
    h = []

    def agregar(nivel, var, mascara_o_n, detalle):
        n = int(np.sum(mascara_o_n)) if not np.isscalar(mascara_o_n) else int(mascara_o_n)
        if n > 0:
            h.append({"nivel": nivel, "variable": var, "filas": n, "detalle": detalle})

    ausentes = [c for c in COLUMNAS if c not in X.columns]
    for c in ausentes:
        agregar("error", c, len(X), "La columna no está en los datos; el modelo la necesita.")
    extra = [c for c in X.columns if c not in COLUMNAS + ["id", "riesgo"]]
    if extra:
        h.append({"nivel": "info", "variable": ", ".join(extra), "filas": len(X),
                  "detalle": "Columnas adicionales que el modelo ignora."})
    if ausentes:
        return pd.DataFrame(h, columns=["nivel", "variable", "filas", "detalle"])

    for var, mascara, detalle in _reglas_de_error(X, perfil):
        agregar("error", var, mascara, detalle)

    for c in NUM_COLS:
        v = pd.to_numeric(X[c], errors="coerce")
        p = perfil["numericas"][c]
        agregar("advertencia", c, (v < p["min"]) | (v > p["max"]),
                f"Fuera del rango de entrenamiento [{_miles(p['min'])}, {_miles(p['max'])}]; el modelo extrapola.")

    saldo = pd.to_numeric(X["saldo_cuenta_cop"], errors="coerce")
    for cat, r in perfil["saldo_por_categoria"].items():
        if cat == "sin_cuenta":
            continue
        fuera = (X["saldo_categoria"] == cat) & ((saldo < r["min"]) | (saldo > r["max"]))
        agregar("advertencia", "saldo_cuenta_cop", fuera,
                f"Saldo fuera del rango visto para la categoría '{cat}' "
                f"[{_miles(r['min'])}, {_miles(r['max'])}].")

    for c in COLUMNAS:
        n_falt = int(X[c].isna().sum())
        if n_falt:
            agregar("advertencia" if c in ("estado_pago_previo", "saldo_categoria") else "info", c, n_falt,
                    "Valores faltantes: el pipeline los imputa con la mediana o la moda de entrenamiento"
                    + (" (variable muy influyente: la imputación pesa en el resultado)."
                       if c in ("estado_pago_previo", "saldo_categoria") else "."))

    if deriva and len(X) >= min_filas_deriva:
        for c in CAT_COLS:
            esp = pd.Series(perfil["proporciones"][c])
            obs = X[c].value_counts(normalize=True).reindex(esp.index).fillna(0)
            _hallazgo_psi(h, c, psi(esp, obs))
        for c in NUM_COLS:
            p = perfil["numericas"][c]
            v = pd.to_numeric(X[c], errors="coerce").dropna()
            obs = np.histogram(v, bins=_bordes(p["cortes"]))[0] / max(len(v), 1)
            _hallazgo_psi(h, c, psi(p["proporciones"], obs))
    return pd.DataFrame(h, columns=["nivel", "variable", "filas", "detalle"])


def _reglas_de_error(X, perfil):
    """Reglas que impiden puntuar una fila: (variable, máscara por fila, detalle)."""
    reglas = []
    for c in CAT_COLS:
        valores = X[c]
        desconocidas = valores.notna() & ~valores.isin(perfil["categorias"][c])
        ejemplos = ", ".join(map(str, valores[desconocidas].unique()[:3]))
        reglas.append((c, desconocidas, f"Categoría no vista en entrenamiento ({ejemplos})."))
    for c in NUM_COLS:
        v = pd.to_numeric(X[c], errors="coerce")
        reglas.append((c, X[c].notna() & v.isna(), "Valor no numérico."))
        if c != "saldo_cuenta_cop":
            reglas.append((c, v <= 0, "Valor menor o igual a cero."))
        else:
            reglas.append((c, v < 0, "Saldo negativo."))
    edad = pd.to_numeric(X["edad"], errors="coerce")
    reglas.append(("edad", edad < 18, "Menor de edad: no es sujeto de crédito."))
    saldo = pd.to_numeric(X["saldo_cuenta_cop"], errors="coerce")
    sin_cuenta = X["saldo_categoria"] == "sin_cuenta"
    reglas.append(("saldo_cuenta_cop", sin_cuenta & (saldo > 0),
                   "Contradicción: categoría 'sin cuenta' con saldo mayor que cero."))
    reglas.append(("saldo_cuenta_cop", ~sin_cuenta & X["saldo_categoria"].notna() & (saldo == 0),
                   "Contradicción: tiene cuenta pero el saldo es cero."))
    return reglas


def filas_con_error(X, perfil):
    """Máscara de filas que no deben puntuarse (alguna regla de error se cumple)."""
    if any(c not in X.columns for c in COLUMNAS):
        return pd.Series(True, index=X.index)
    mascara = pd.Series(False, index=X.index)
    for _, m, _ in _reglas_de_error(X, perfil):
        mascara |= m.fillna(False).astype(bool)
    return mascara


def _hallazgo_psi(h, c, valor):
    if valor >= 0.25:
        h.append({"nivel": "advertencia", "variable": c, "filas": 0,
                  "detalle": f"Deriva importante frente a entrenamiento (PSI = {valor:.2f} ≥ 0.25)."})
    elif valor >= 0.10:
        h.append({"nivel": "info", "variable": c, "filas": 0,
                  "detalle": f"Deriva moderada frente a entrenamiento (PSI = {valor:.2f})."})


# ---------------------------------------------------------------------------
# 4. Mapa de decisión: el modelo en el plano monto × plazo
# ---------------------------------------------------------------------------
class CorteMontoPlazo:
    """El modelo como función del monto y del plazo, con las otras 9 variables fijas en los
    valores de una solicitud: un corte ceteris paribus (una curva ICE en dos dimensiones).

    `caminos()` busca, para cada una de las dos variables por separado, hasta dónde se puede
    mover sin que cambie la decisión (si se aprueba) o hasta dónde hay que moverla para que
    cambie (si se rechaza). La versión web (web/src/motor.js) hace el mismo cálculo.

    Cálculo rápido: con las otras 9 variables fijas, el kernel RBF se separa en
    exp(-γ·d_resto) · exp(-γ·(monto - sv)²) · exp(-γ·(plazo - sv)²), así que el puntaje de
    toda la malla sale de un producto de matrices en vez de miles de llamadas al pipeline.
    La solicitud pasa una sola vez por el pipeline (imputación, escalamiento y one-hot), y el
    resultado se contrasta con `decision_function`: si no coincide, se usa el pipeline.
    """

    PLAZOS = np.arange(3, 73, 3)                  # el formulario usa múltiplos de 3 meses
    N_BARRIDO, N_BISECCION, REDONDEO = 240, 40, 10_000

    def __init__(self, modelo, x, perfil):
        self.modelo = modelo
        self.x = x[COLUMNAS].iloc[[0]].reset_index(drop=True)
        rango = perfil["numericas"]
        self.monto_min, self.monto_max = rango["monto_credito_cop"]["min"], rango["monto_credito_cop"]["max"]
        self.plazo_min, self.plazo_max = rango["plazo_meses"]["min"], rango["plazo_meses"]["max"]
        pipe = modelo.pipeline
        num = pipe.named_steps["imputar_escalar"].named_transformers_["num"]
        # Si falta el monto o el plazo, el pipeline usa la mediana de entrenamiento
        mediana = dict(zip(NUM_COLS, num.named_steps["imputer"].statistics_))
        valor = {c: float(pd.to_numeric(self.x.loc[0, c], errors="coerce"))
                 for c in ["monto_credito_cop", "plazo_meses"]}
        self.monto = mediana["monto_credito_cop"] if np.isnan(valor["monto_credito_cop"]) else valor["monto_credito_cop"]
        self.plazo = mediana["plazo_meses"] if np.isnan(valor["plazo_meses"]) else valor["plazo_meses"]
        self.umbral = modelo.umbral

        # Piezas del cálculo rápido
        svc, escala = pipe.named_steps["modelo"], num.named_steps["scaler"]
        columnas = list(pipe.named_steps["codificar"].get_feature_names_out())
        jm, jp = columnas.index("monto_credito_cop"), columnas.index("plazo_meses")
        km, kp = NUM_COLS.index("monto_credito_cop"), NUM_COLS.index("plazo_meses")
        self._esc = (escala.mean_[km], escala.scale_[km], escala.mean_[kp], escala.scale_[kp])
        z = np.asarray(pipe[:-1].transform(self.x), dtype=float)[0]
        sv = svc.support_vectors_
        otras = [j for j in range(sv.shape[1]) if j not in (jm, jp)]
        self._gamma = float(svc._gamma)
        self._peso = svc.dual_coef_[0] * np.exp(-self._gamma * ((sv[:, otras] - z[otras]) ** 2).sum(axis=1))
        self._sv_monto, self._sv_plazo, self._b = sv[:, jm], sv[:, jp], float(svc.intercept_[0])
        referencia = float(modelo.puntaje(self.x)[0])
        self._rapido = abs(self._puntajes_rapidos(np.array([self.monto]), np.array([self.plazo]))[0] - referencia) < 1e-8
        self.aprobado = bool(referencia < self.umbral)

    def _factores(self, valores, media, desv, sv):
        z = (np.asarray(valores, dtype=float).ravel() - media) / desv
        return np.exp(-self._gamma * (z[:, None] - sv[None, :]) ** 2)

    def _puntajes_rapidos(self, montos, plazos):
        mm, sm, mp, sp = self._esc
        fm = self._factores(montos, mm, sm, self._sv_monto)
        fp = self._factores(plazos, mp, sp, self._sv_plazo)
        return self._b + (fm * fp) @ self._peso

    def puntajes(self, montos, plazos):
        """Puntaje del SVM para pares (monto, plazo); acepta escalares o arreglos."""
        montos, plazos = np.broadcast_arrays(np.asarray(montos, dtype=float), np.asarray(plazos, dtype=float))
        if self._rapido:
            return self._puntajes_rapidos(montos.ravel(), plazos.ravel()).reshape(montos.shape)
        X = self.x.loc[np.zeros(montos.size, dtype=int)].reset_index(drop=True)
        X["monto_credito_cop"] = montos.ravel()
        X["plazo_meses"] = plazos.ravel()
        return self.modelo.puntaje(X).reshape(montos.shape)

    def malla(self, n_monto=120, n_plazo=80):
        """Montos (escala log), plazos y matriz de puntajes (filas = plazos)."""
        montos = np.geomspace(self.monto_min, self.monto_max, n_monto)
        plazos = np.linspace(self.plazo_min, self.plazo_max, n_plazo)
        if not self._rapido:
            return montos, plazos, self.puntajes(*np.meshgrid(montos, plazos))
        mm, sm, mp, sp = self._esc
        fm = self._factores(montos, mm, sm, self._sv_monto)                 # n_monto × SV
        fp = self._factores(plazos, mp, sp, self._sv_plazo) * self._peso    # n_plazo × SV
        return montos, plazos, self._b + fp @ fm.T

    def _limite_monto(self):
        """Monto más alto con el que la solicitud queda aprobada, buscando desde el monto actual
        hacia abajo (si se rechaza) o hacia arriba (si se aprueba). None: no existe en el rango
        de entrenamiento (rechazo) o se aprueba en todo él (aprobación)."""
        hacia = self.monto_max if self.aprobado else self.monto_min
        if (self.monto >= hacia) if self.aprobado else (self.monto <= hacia):
            return None
        l0, l1 = np.log(self.monto), np.log(hacia)
        logs = l0 + np.arange(self.N_BARRIDO + 1) * ((l1 - l0) / self.N_BARRIDO)
        cambia = (lambda s: s >= self.umbral) if self.aprobado else (lambda s: s < self.umbral)
        c = cambia(self.puntajes(np.exp(logs), self.plazo))
        if not c[1:].any():
            return None
        j = int(np.argmax(c[1:])) + 1
        lo, hi = logs[j - 1], logs[j]                # lo: misma decisión; hi: decisión cambiada
        for _ in range(self.N_BISECCION):
            medio = (lo + hi) / 2
            if cambia(self.puntajes(np.exp(medio), self.plazo)[()]):
                hi = medio
            else:
                lo = medio
        valor = np.floor(np.exp(lo if self.aprobado else hi) / self.REDONDEO) * self.REDONDEO
        for _ in range(5):                           # el redondeo no debe cruzar la frontera
            if valor > self.REDONDEO and self.puntajes(valor, self.plazo)[()] >= self.umbral:
                valor -= self.REDONDEO
            else:
                break
        return float(max(valor, self.monto) if self.aprobado else valor)

    def _limite_plazo(self):
        """Plazo más largo (múltiplo de 3) con el que la solicitud queda aprobada, con la misma
        lógica y el mismo significado de None que `_limite_monto`."""
        if self.aprobado:
            mayores = self.PLAZOS[self.PLAZOS > self.plazo]
            rechaza = self.puntajes(self.monto, mayores) >= self.umbral
            if not rechaza.any():
                return None
            j = int(np.argmax(rechaza))
            return float(mayores[j - 1]) if j > 0 else float(self.plazo)
        menores = self.PLAZOS[self.PLAZOS < self.plazo][::-1]
        aprueba = self.puntajes(self.monto, menores) < self.umbral
        return float(menores[int(np.argmax(aprueba))]) if aprueba.any() else None

    def caminos(self):
        return {"aprobado": self.aprobado, "monto": self._limite_monto(), "plazo": self._limite_plazo()}
