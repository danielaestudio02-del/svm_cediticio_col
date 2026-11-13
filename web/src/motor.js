/*
 * Motor del modelo final en el navegador (sin dependencias ni acceso al DOM).
 *
 * Reproduce, con los parámetros exportados por web/construir_web.py, lo que hacen
 * src/preprocesamiento.py, src/interpretacion.py y app/app_clasica.py en Python:
 *   - el Pipeline: imputación (saldo por categoría; mediana o moda en el resto),
 *     escalamiento y one-hot;
 *   - el SVM con kernel RBF, la calibración de Platt y el umbral de costo mínimo 5:1;
 *   - la verificación de coherencia de los datos de uso (esquema, categorías,
 *     rangos, reglas de negocio y deriva por PSI);
 *   - la lectura de CSV (separador "," o ";", UTF-8 o Latin-1, valores nulos de pandas);
 *   - la explicación SHAP de cada predicción a nivel de las 11 variables;
 *   - el mapa de decisión monto × plazo y la búsqueda del cambio que cambia la decisión.
 *
 * SHAP: la aplicación de Streamlit estima los valores de Shapley con permutaciones
 * aleatorias (shap.PermutationExplainer, fondo de 50 clientes). Aquí se calculan de
 * forma EXACTA para el mismo fondo: el kernel RBF es un producto de factores, uno
 * por variable, así que el juego de cada par (vector de soporte, cliente de fondo)
 * es un producto y su valor de Shapley sale de un polinomio en t (coeficiente de
 * t^k = coaliciones de tamaño k). El saldo y su categoría forman un bloque, porque
 * el saldo faltante se imputa con la mediana de su categoría.
 */
const Motor = (() => {
  "use strict";

  // Valores que pandas.read_csv interpreta como faltantes por defecto
  const VALORES_NULOS = new Set(["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]);

  const esNulo = (v) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));

  /** pd.to_numeric(errors="coerce"); con comaDecimal acepta "35,5" y "4.000.000" (Excel en español). */
  function aNumero(v, comaDecimal = false) {
    if (esNulo(v)) return NaN;
    if (typeof v === "number") return v;
    let s = String(v).trim();
    if (comaDecimal) {
      if (/^-?\d{1,3}(\.\d{3})+(,\d+)?$/.test(s)) s = s.replace(/\./g, "").replace(",", ".");
      else if (/^-?\d+,\d+$/.test(s)) s = s.replace(",", ".");
    }
    if (s === "") return NaN;
    const n = Number(s);
    return Number.isFinite(n) ? n : NaN;
  }

  /** f"{v:,.0f}".replace(",", ".") */
  function miles(v) {
    const n = Math.round(Math.abs(v));
    return (v < 0 && n !== 0 ? "-" : "") + String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  }

  // -------------------------------------------------------------------------
  // CSV
  // -------------------------------------------------------------------------
  /** Decodifica UTF-8 (estricto) y, si falla, Windows-1252/Latin-1, como lo exporta Excel. */
  function decodificar(buffer) {
    let texto;
    try {
      texto = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
    } catch (e) {
      texto = new TextDecoder("windows-1252").decode(buffer);
    }
    return texto.charCodeAt(0) === 0xfeff ? texto.slice(1) : texto;
  }

  function detectarSeparador(texto) {
    let enComillas = false;
    const cuenta = { ",": 0, ";": 0, "\t": 0 };
    for (const ch of texto) {
      if (ch === '"') enComillas = !enComillas;
      else if (!enComillas && (ch === "\n" || ch === "\r")) break;
      else if (!enComillas && ch in cuenta) cuenta[ch] += 1;
    }
    const mejor = Object.entries(cuenta).sort((a, b) => b[1] - a[1])[0];
    return mejor[1] > 0 ? mejor[0] : ",";
  }

  /** Lector CSV (RFC 4180). Devuelve {columnas, filas, separador} o {error}. */
  function leerCSV(texto) {
    if (!texto || !texto.trim()) return { error: "El archivo está vacío." };
    const sep = detectarSeparador(texto);
    const registros = [];
    let campo = "", registro = [], enComillas = false, i = 0;
    while (i < texto.length) {
      const ch = texto[i];
      if (enComillas) {
        if (ch === '"') {
          if (texto[i + 1] === '"') { campo += '"'; i += 1; } else enComillas = false;
        } else campo += ch;
      } else if (ch === '"') enComillas = true;
      else if (ch === sep) { registro.push(campo); campo = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && texto[i + 1] === "\n") i += 1;
        registro.push(campo); registros.push(registro); registro = []; campo = "";
      } else campo += ch;
      i += 1;
    }
    if (campo !== "" || registro.length) { registro.push(campo); registros.push(registro); }
    const datos = registros.filter((r) => !(r.length === 1 && r[0].trim() === ""));
    if (!datos.length) return { error: "El archivo está vacío." };

    const columnas = datos[0].map((c) => c.trim());
    const filas = [];
    for (let r = 1; r < datos.length; r++) {
      if (datos[r].length > columnas.length) {
        return { error: `La línea ${r + 1} tiene ${datos[r].length} campos y el encabezado tiene ${columnas.length}.` };
      }
      const fila = {};
      columnas.forEach((c, j) => {
        const v = j < datos[r].length ? datos[r][j].trim() : "";
        fila[c] = VALORES_NULOS.has(v) ? null : v;
      });
      filas.push(fila);
    }
    return { columnas, filas, separador: sep };
  }

  function escribirCSV(columnas, filas) {
    const celda = (v) => {
      if (esNulo(v)) return "";
      const s = String(v);
      return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    return [columnas.map(celda).join(","), ...filas.map((f) => columnas.map((c) => celda(f[c])).join(","))].join("\n");
  }

  // -------------------------------------------------------------------------
  // Modelo
  // -------------------------------------------------------------------------
  function crearMotor(M, opciones = {}) {
    const comaDecimal = !!opciones.comaDecimal;
    const pre = M.preprocesamiento, svm = M.svm, perfil = M.perfil;
    const COLUMNAS = M.columnas, NUM = M.num_cols, CAT = M.cat_cols;
    const K_SALDO = NUM.indexOf("saldo_cuenta_cop"), K_CAT_SALDO = CAT.indexOf("saldo_categoria");
    const nSV = svm.alfa.length, gamma = svm.gamma, b0 = svm.intercepto;
    const alfa = Float64Array.from(svm.alfa);
    const svNum = Float64Array.from(svm.sv_num);
    const svCat = new Int8Array(nSV * CAT.length);
    for (let i = 0; i < svCat.length; i++) svCat[i] = svm.sv_cat.charCodeAt(i) - 48;
    const indiceCat = pre.categorias.map((cats) => new Map(cats.map((v, i) => [v, i])));
    // Factor del kernel por bloque one-hot: misma categoría (0), desconocida (1) o distinta (2)
    const FACTOR_CAT = [1, Math.exp(-gamma), Math.exp(-2 * gamma)];
    const distCat = (ci, sc) => (ci === sc ? 0 : ci < 0 ? 1 : 2);

    const valorNum = (fila, c) => aNumero(fila[c], comaDecimal);
    const escalar = (k, v) => ((Number.isNaN(v) ? pre.mediana_num[k] : v) - pre.media[k]) / pre.escala[k];
    function escalarSaldo(saldo, categoria) {
      let s = saldo;
      if (Number.isNaN(s)) {
        s = !esNulo(categoria) && Object.prototype.hasOwnProperty.call(pre.medianas_saldo, categoria)
          ? pre.medianas_saldo[categoria] : pre.mediana_num[K_SALDO];
      }
      return (s - pre.media[K_SALDO]) / pre.escala[K_SALDO];
    }
    function indiceCategoria(k, v) {
      const valor = esNulo(v) ? pre.moda_cat[k] : String(v);
      const i = indiceCat[k].get(valor);
      return i === undefined ? -1 : i;
    }

    /** Fila cruda -> 4 numéricas escaladas + 7 índices de categoría (como el Pipeline). */
    function transformar(fila) {
      const z = new Float64Array(NUM.length), c = new Int8Array(CAT.length);
      NUM.forEach((col, k) => {
        z[k] = k === K_SALDO ? escalarSaldo(valorNum(fila, col), fila.saldo_categoria) : escalar(k, valorNum(fila, col));
      });
      CAT.forEach((col, k) => { c[k] = indiceCategoria(k, fila[col]); });
      return { z, c };
    }

    function puntajeTransformado({ z, c }) {
      let s = b0;
      for (let i = 0; i < nSV; i++) {
        let d = 0;
        const on = i * 4, oc = i * 7;
        for (let k = 0; k < 4; k++) { const t = z[k] - svNum[on + k]; d += t * t; }
        for (let k = 0; k < 7; k++) d += distCat(c[k], svCat[oc + k]);
        s += alfa[i] * Math.exp(-gamma * d);
      }
      return s;
    }

    const puntaje = (fila) => puntajeTransformado(transformar(fila));
    const probabilidadDePuntaje = (s) => 1 / (1 + Math.exp(-(M.meta.platt.a * s + M.meta.platt.b)));
    const probabilidad = (fila) => probabilidadDePuntaje(puntaje(fila));
    const umbralProbabilidad = probabilidadDePuntaje(M.meta.umbral);

    // -----------------------------------------------------------------------
    // SHAP exacto (ver comentario inicial)
    // -----------------------------------------------------------------------
    const N_JUG = COLUMNAS.length;                                  // 11 variables
    const PESOS = [];                                               // k!(M-1-k)!/M!
    const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
    for (let k = 0; k < N_JUG; k++) PESOS.push(fact(k) * fact(N_JUG - 1 - k) / fact(N_JUG));
    const J_SALDO = COLUMNAS.indexOf("saldo_cuenta_cop"), J_CAT_SALDO = COLUMNAS.indexOf("saldo_categoria");
    // Jugadores independientes: [columna, tipo, índice en NUM o CAT]
    const INDEP = COLUMNAS.map((col, j) => [j, NUM.includes(col) ? "num" : "cat", NUM.includes(col) ? NUM.indexOf(col) : CAT.indexOf(col)])
      .filter(([j]) => j !== J_SALDO && j !== J_CAT_SALDO);
    const NI = INDEP.length;                                        // 9

    /** Coordenadas por jugador de una fila: valor escalado o índice de categoría, y el bloque saldo. */
    function coordenadas(fila) {
      const v = INDEP.map(([j, tipo, k]) => (tipo === "num" ? escalar(k, valorNum(fila, COLUMNAS[j])) : indiceCategoria(k, fila[COLUMNAS[j]])));
      return { v, saldo: valorNum(fila, "saldo_cuenta_cop"), catSaldo: fila.saldo_categoria };
    }
    const factor = (tipo, k, v, i) => (tipo === "num"
      ? Math.exp(-gamma * (v - svNum[i * 4 + k]) ** 2)
      : FACTOR_CAT[distCat(v, svCat[i * 7 + k])]);

    /**
     * Crea el cálculo de la explicación de `fila` sobre el fondo de entrenamiento.
     * Se avanza por partes con paso(n) para no congelar la interfaz.
     */
    const ES_NUM = Uint8Array.from(INDEP, ([, tipo]) => (tipo === "num" ? 1 : 0));
    const K_JUG = Int8Array.from(INDEP, ([, , k]) => k);
    const L = NI + 1;                                               // largo de cada polinomio guardado

    function crearExplicacion(fila, fondo = M.fondo) {
      const filasFondo = fondo.map((f) => Object.fromEntries(COLUMNAS.map((c, j) => [c, Array.isArray(f) ? f[j] : f[c]])));
      const x = coordenadas(fila);
      const u = new Float64Array(nSV * NI);                          // factores de x, por vector de soporte
      for (let i = 0; i < nSV; i++) INDEP.forEach(([, tipo, k], p) => { u[i * NI + p] = factor(tipo, k, x.v[p], i); });
      const phi = new Float64Array(N_JUG);
      let base = 0, b = 0;
      const w = new Float64Array(NI), q = new Float64Array(L);
      // pre[p] = Π_{l<p} (w_l + u_l t)  y  suf[p] = Π_{l≥p} (w_l + u_l t), guardados en filas de largo L
      const pre = new Float64Array((NI + 1) * L), suf = new Float64Array((NI + 1) * L);

      function pesoSobre(p, o, grado, d0, d1, d2) {                // Σ_k PESOS[k] [t^k] p(t)(d0 + d1 t + d2 t²)
        let s = 0;
        for (let k = 0; k <= grado; k++) {
          s += p[o + k] * (d0 * PESOS[k] + (k + 1 < N_JUG ? d1 * PESOS[k + 1] : 0) + (k + 2 < N_JUG ? d2 * PESOS[k + 2] : 0));
        }
        return s;
      }

      function paso(n = 1) {
        for (let r = 0; r < n && b < filasFondo.length; r++, b++) {
          const fb = coordenadas(filasFondo[b]);
          // Bloque saldo: (saldo de x|fondo, categoría de x|fondo)
          const zs00 = escalarSaldo(fb.saldo, fb.catSaldo), zs01 = escalarSaldo(fb.saldo, x.catSaldo);
          const zs10 = escalarSaldo(x.saldo, fb.catSaldo), zs11 = escalarSaldo(x.saldo, x.catSaldo);
          const cb = indiceCategoria(K_CAT_SALDO, fb.catSaldo), cx = indiceCategoria(K_CAT_SALDO, x.catSaldo);
          for (let i = 0; i < nSV; i++) {
            const on = i * 4, oc = i * 7, ui = i * NI;
            for (let p = 0; p < NI; p++) {
              if (ES_NUM[p]) { const t = fb.v[p] - svNum[on + K_JUG[p]]; w[p] = Math.exp(-gamma * t * t); }
              else w[p] = FACTOR_CAT[distCat(fb.v[p], svCat[oc + K_JUG[p]])];
            }
            // h(saldo de x?, categoría de x?) para el bloque saldo
            const sv1 = svNum[on + K_SALDO], svc = svCat[oc + K_CAT_SALDO];
            const fcb = FACTOR_CAT[distCat(cb, svc)], fcx = FACTOR_CAT[distCat(cx, svc)];
            const h00 = Math.exp(-gamma * (zs00 - sv1) ** 2) * fcb, h01 = Math.exp(-gamma * (zs01 - sv1) ** 2) * fcx;
            const h10 = Math.exp(-gamma * (zs10 - sv1) ** 2) * fcb, h11 = Math.exp(-gamma * (zs11 - sv1) ** 2) * fcx;
            const a = alfa[i];

            // Productos prefijo y sufijo de (w_l + u_l t)
            pre.fill(0, 0, L); pre[0] = 1;
            for (let p = 0; p < NI; p++) {
              const o = p * L, o2 = o + L, wp = w[p], up = u[ui + p];
              pre[o2] = pre[o] * wp;
              for (let k = 1; k <= p + 1; k++) pre[o2 + k] = (k <= p ? pre[o + k] * wp : 0) + pre[o + k - 1] * up;
            }
            const oS = NI * L; suf.fill(0, oS, oS + L); suf[oS] = 1;
            for (let p = NI - 1; p >= 0; p--) {
              const o = p * L, o2 = o + L, wp = w[p], up = u[ui + p], g = NI - 1 - p;   // grado de suf[p+1]
              suf[o] = suf[o2] * wp;
              for (let k = 1; k <= g + 1; k++) suf[o + k] = (k <= g ? suf[o2 + k] * wp : 0) + suf[o2 + k - 1] * up;
            }
            base += a * pre[NI * L] * h00;                           // Π w_l (coeficiente t^0) · h00 = f(fondo)
            // Jugadores del bloque saldo (los 9 independientes, en pre[NI])
            phi[J_SALDO] += a * pesoSobre(pre, NI * L, NI, h10 - h00, h11 - h01, 0);
            phi[J_CAT_SALDO] += a * pesoSobre(pre, NI * L, NI, h01 - h00, h11 - h10, 0);
            // Jugadores independientes: Π_{l≠p} = pre[p] · suf[p+1]
            for (let p = 0; p < NI; p++) {
              const gp = p, gs = NI - 1 - p, op = p * L, os = (p + 1) * L;
              q.fill(0);
              for (let m = 0; m <= gp; m++) { const c = pre[op + m]; for (let n2 = 0; n2 <= gs; n2++) q[m + n2] += c * suf[os + n2]; }
              phi[INDEP[p][0]] += a * (u[ui + p] - w[p]) * pesoSobre(q, 0, NI - 1, h00, h10 + h01, h11);
            }
          }
        }
        return b / filasFondo.length;                                // avance (0-1)
      }

      function resultado() {
        const nB = filasFondo.length;
        const valores = Array.from(phi, (v) => v / nB);
        return { valores, base: base / nB + b0, puntaje: puntaje(fila) };
      }
      return { paso, resultado, terminado: () => b >= filasFondo.length };
    }

    function explicar(fila, fondo) {
      const e = crearExplicacion(fila, fondo);
      while (!e.terminado()) e.paso(10);
      return e.resultado();
    }

    // -----------------------------------------------------------------------
    // Mapa de decisión (interpretacion.CorteMontoPlazo): el modelo en el plano
    // monto × plazo con las otras 9 variables fijas. Con ellas fijas, el kernel
    // RBF se separa en (factor fijo) · (factor del monto) · (factor del plazo),
    // así que la malla completa sale de un producto de matrices.
    // -----------------------------------------------------------------------
    const K_MONTO = NUM.indexOf("monto_credito_cop"), K_PLAZO = NUM.indexOf("plazo_meses");
    const PLAZOS = Array.from({ length: 24 }, (_, i) => 3 * (i + 1));  // el formulario usa múltiplos de 3
    const N_BARRIDO = 240, N_BISECCION = 40, REDONDEO = 10000;

    function crearCorte(fila) {
      const { z, c } = transformar(fila);
      const peso = new Float64Array(nSV), svM = new Float64Array(nSV), svP = new Float64Array(nSV);
      for (let i = 0; i < nSV; i++) {
        let d = 0;
        const on = i * 4, oc = i * 7;
        for (let k = 0; k < 4; k++) if (k !== K_MONTO && k !== K_PLAZO) { const t = z[k] - svNum[on + k]; d += t * t; }
        for (let k = 0; k < 7; k++) d += distCat(c[k], svCat[oc + k]);
        peso[i] = alfa[i] * Math.exp(-gamma * d);
        svM[i] = svNum[on + K_MONTO]; svP[i] = svNum[on + K_PLAZO];
      }
      const zM = (m) => (m - pre.media[K_MONTO]) / pre.escala[K_MONTO];
      const zP = (p) => (p - pre.media[K_PLAZO]) / pre.escala[K_PLAZO];
      function puntajeEn(monto, plazo) {
        const a = zM(monto), b = zP(plazo);
        let s = b0;
        for (let i = 0; i < nSV; i++) { const t = a - svM[i], u = b - svP[i]; s += peso[i] * Math.exp(-gamma * (t * t + u * u)); }
        return s;
      }
      /** Puntajes en la malla montos × plazos; fila y = plazos[y]. */
      function malla(montos, plazos) {
        const nx = montos.length, ny = plazos.length, S = new Float64Array(nx * ny).fill(b0);
        const em = new Float64Array(nx), ep = new Float64Array(ny);
        const zm = montos.map(zM), zp = plazos.map(zP);
        for (let i = 0; i < nSV; i++) {
          for (let x = 0; x < nx; x++) { const t = zm[x] - svM[i]; em[x] = Math.exp(-gamma * t * t); }
          for (let y = 0; y < ny; y++) { const t = zp[y] - svP[i]; ep[y] = peso[i] * Math.exp(-gamma * t * t); }
          for (let y = 0; y < ny; y++) { const w = ep[y], o = y * nx; for (let x = 0; x < nx; x++) S[o + x] += w * em[x]; }
        }
        return S;
      }
      // Si falta el monto o el plazo, el pipeline usa la mediana de entrenamiento
      const dato = (k, col) => { const v = valorNum(fila, col); return Number.isNaN(v) ? pre.mediana_num[k] : v; };
      const r = perfil.numericas;
      const corte = {
        puntajeEn, malla, umbral: M.meta.umbral,
        monto: dato(K_MONTO, "monto_credito_cop"), plazo: dato(K_PLAZO, "plazo_meses"),
        montoImputado: Number.isNaN(valorNum(fila, "monto_credito_cop")), plazoImputado: Number.isNaN(valorNum(fila, "plazo_meses")),
        montoMin: r.monto_credito_cop.min, montoMax: r.monto_credito_cop.max, plazoMin: r.plazo_meses.min, plazoMax: r.plazo_meses.max,
      };
      corte.aprobado = puntajeEn(corte.monto, corte.plazo) < corte.umbral;
      corte.caminos = () => ({ aprobado: corte.aprobado, monto: limiteMonto(corte), plazo: limitePlazo(corte) });
      return corte;
    }

    /** Monto más alto con el que se aprueba, buscando hacia abajo (rechazo) o hacia arriba (aprobación).
     *  null: no existe en el rango de entrenamiento (rechazo) o se aprueba en todo él (aprobación). */
    function limiteMonto(k) {
      const hacia = k.aprobado ? k.montoMax : k.montoMin;
      if (k.aprobado ? k.monto >= hacia : k.monto <= hacia) return null;
      const l0 = Math.log(k.monto), l1 = Math.log(hacia), paso = (l1 - l0) / N_BARRIDO;
      const cambia = (s) => (k.aprobado ? s >= k.umbral : s < k.umbral);
      let j = -1;
      for (let i = 1; i <= N_BARRIDO; i++) if (cambia(k.puntajeEn(Math.exp(l0 + i * paso), k.plazo))) { j = i; break; }
      if (j < 0) return null;
      let lo = l0 + (j - 1) * paso, hi = l0 + j * paso;           // lo: misma decisión; hi: decisión cambiada
      for (let i = 0; i < N_BISECCION; i++) {
        const medio = (lo + hi) / 2;
        if (cambia(k.puntajeEn(Math.exp(medio), k.plazo))) hi = medio; else lo = medio;
      }
      let valor = Math.floor(Math.exp(k.aprobado ? lo : hi) / REDONDEO) * REDONDEO;
      for (let i = 0; i < 5 && valor > REDONDEO && k.puntajeEn(valor, k.plazo) >= k.umbral; i++) valor -= REDONDEO;
      return k.aprobado ? Math.max(valor, k.monto) : valor;
    }

    /** Plazo más largo (múltiplo de 3) con el que se aprueba; mismo significado de null. */
    function limitePlazo(k) {
      if (k.aprobado) {
        let ultimo = k.plazo;
        for (const p of PLAZOS.filter((p) => p > k.plazo)) {
          if (k.puntajeEn(k.monto, p) >= k.umbral) return ultimo;
          ultimo = p;
        }
        return null;
      }
      for (const p of PLAZOS.filter((p) => p < k.plazo).reverse()) if (k.puntajeEn(k.monto, p) < k.umbral) return p;
      return null;
    }

    // -----------------------------------------------------------------------
    // Verificación de coherencia (interpretacion.verificar_coherencia)
    // -----------------------------------------------------------------------
    function reglasDeError(filas) {
      const reglas = [];
      CAT.forEach((c, k) => {
        const desconocidas = filas.map((f) => !esNulo(f[c]) && !perfil.categorias[c].includes(String(f[c])));
        const ejemplos = [...new Set(filas.filter((f, i) => desconocidas[i]).map((f) => String(f[c])))].slice(0, 3).join(", ");
        reglas.push([c, desconocidas, `Categoría no vista en entrenamiento (${ejemplos}).`]);
      });
      NUM.forEach((c) => {
        const v = filas.map((f) => valorNum(f, c));
        reglas.push([c, filas.map((f, i) => !esNulo(f[c]) && Number.isNaN(v[i])), "Valor no numérico."]);
        if (c !== "saldo_cuenta_cop") reglas.push([c, v.map((x) => x <= 0), "Valor menor o igual a cero."]);
        else reglas.push([c, v.map((x) => x < 0), "Saldo negativo."]);
      });
      const edad = filas.map((f) => valorNum(f, "edad"));
      reglas.push(["edad", edad.map((x) => x < 18), "Menor de edad: no es sujeto de crédito."]);
      const saldo = filas.map((f) => valorNum(f, "saldo_cuenta_cop"));
      const sinCuenta = filas.map((f) => f.saldo_categoria === "sin_cuenta");
      reglas.push(["saldo_cuenta_cop", filas.map((f, i) => sinCuenta[i] && saldo[i] > 0),
        "Contradicción: categoría 'sin cuenta' con saldo mayor que cero."]);
      reglas.push(["saldo_cuenta_cop", filas.map((f, i) => !sinCuenta[i] && !esNulo(f.saldo_categoria) && saldo[i] === 0),
        "Contradicción: tiene cuenta pero el saldo es cero."]);
      return reglas;
    }

    function filasConError(filas, columnas) {
      if (COLUMNAS.some((c) => !columnas.includes(c))) return filas.map(() => true);
      const mascara = filas.map(() => false);
      for (const [, m] of reglasDeError(filas)) m.forEach((v, i) => { if (v) mascara[i] = true; });
      return mascara;
    }

    const psi = (esperado, observado, eps = 1e-4) => esperado.reduce((s, e, i) => {
      const ee = Math.max(e, eps), oo = Math.max(observado[i], eps);
      return s + (oo - ee) * Math.log(oo / ee);
    }, 0);

    /** np.histogram con bordes extremos en ±infinito (el último tramo incluye su borde derecho). */
    function histograma(valores, cortes) {
      const bordes = cortes.slice(); bordes[0] = -Infinity; bordes[bordes.length - 1] = Infinity;
      const cuenta = new Array(bordes.length - 1).fill(0);
      for (const v of valores) {
        let i = cuenta.length - 1;
        for (let t = 0; t < cuenta.length; t++) if (v < bordes[t + 1]) { i = t; break; }
        cuenta[i] += 1;
      }
      return cuenta;
    }

    function verificarCoherencia(filas, columnas, deriva = true, minFilasDeriva = 100) {
      const h = [];
      const agregar = (nivel, variable, mascaraON, detalle) => {
        const n = Array.isArray(mascaraON) ? mascaraON.filter(Boolean).length : mascaraON;
        if (n > 0) h.push({ nivel, variable, filas: n, detalle });
      };
      const ausentes = COLUMNAS.filter((c) => !columnas.includes(c));
      ausentes.forEach((c) => agregar("error", c, filas.length, "La columna no está en los datos; el modelo la necesita."));
      const extra = columnas.filter((c) => !COLUMNAS.includes(c) && c !== "id" && c !== "riesgo");
      if (extra.length) h.push({ nivel: "info", variable: extra.join(", "), filas: filas.length, detalle: "Columnas adicionales que el modelo ignora." });
      if (ausentes.length) return h;

      for (const [variable, mascara, detalle] of reglasDeError(filas)) agregar("error", variable, mascara, detalle);

      NUM.forEach((c) => {
        const p = perfil.numericas[c];
        agregar("advertencia", c, filas.map((f) => { const v = valorNum(f, c); return v < p.min || v > p.max; }),
          `Fuera del rango de entrenamiento [${miles(p.min)}, ${miles(p.max)}]; el modelo extrapola.`);
      });
      Object.entries(perfil.saldo_por_categoria).forEach(([cat, r]) => {
        if (cat === "sin_cuenta") return;
        agregar("advertencia", "saldo_cuenta_cop", filas.map((f) => {
          const s = valorNum(f, "saldo_cuenta_cop");
          return f.saldo_categoria === cat && (s < r.min || s > r.max);
        }), `Saldo fuera del rango visto para la categoría '${cat}' [${miles(r.min)}, ${miles(r.max)}].`);
      });
      COLUMNAS.forEach((c) => {
        const nFalt = filas.filter((f) => esNulo(f[c])).length;
        const influyente = c === "estado_pago_previo" || c === "saldo_categoria";
        agregar(influyente ? "advertencia" : "info", c, nFalt,
          "Valores faltantes: el pipeline los imputa con la mediana o la moda de entrenamiento" +
          (influyente ? " (variable muy influyente: la imputación pesa en el resultado)." : "."));
      });

      if (deriva && filas.length >= minFilasDeriva) {
        const hallazgoPSI = (c, valor) => {
          if (valor >= 0.25) h.push({ nivel: "advertencia", variable: c, filas: 0, detalle: `Deriva importante frente a entrenamiento (PSI = ${valor.toFixed(2)} ≥ 0.25).` });
          else if (valor >= 0.1) h.push({ nivel: "info", variable: c, filas: 0, detalle: `Deriva moderada frente a entrenamiento (PSI = ${valor.toFixed(2)}).` });
        };
        CAT.forEach((c) => {
          const esperado = perfil.proporciones[c];
          const presentes = filas.filter((f) => !esNulo(f[c])).map((f) => String(f[c]));
          const cats = Object.keys(esperado);
          const obs = cats.map((v) => (presentes.length ? presentes.filter((x) => x === v).length / presentes.length : 0));
          hallazgoPSI(c, psi(cats.map((v) => esperado[v]), obs));
        });
        NUM.forEach((c) => {
          const p = perfil.numericas[c];
          const v = filas.map((f) => valorNum(f, c)).filter((x) => !Number.isNaN(x));
          const cuenta = histograma(v, p.cortes);
          hallazgoPSI(c, psi(p.proporciones, cuenta.map((n) => n / Math.max(v.length, 1))));
        });
      }
      return h;
    }

    return {
      COLUMNAS, NUM, CAT, transformar, puntaje, probabilidad, probabilidadDePuntaje, umbralProbabilidad,
      crearExplicacion, explicar, crearCorte, verificarCoherencia, filasConError, valorNum,
    };
  }

  return { crearMotor, leerCSV, escribirCSV, decodificar, aNumero, esNulo, miles };
})();

if (typeof module !== "undefined") module.exports = Motor;
