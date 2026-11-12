/*
 * Interfaz de la versión web. Misma estructura que app/app.py (Streamlit):
 * barra lateral con la ficha del modelo y tres pestañas: "Evaluar una solicitud",
 * "Evaluar un lote (CSV)" y "Acerca del modelo". Los componentes imitan los de
 * shadcn/ui (Select, Tabs, Card, Badge, Alert, Slider, Checkbox, Tooltip, Sonner,
 * Data Table, Sidebar) en JavaScript sin dependencias.
 */
(() => {
  "use strict";

  const M = JSON.parse(document.getElementById("modelo-datos").textContent);
  const motor = Motor.crearMotor(M);
  let motorPuntoComa = null;
  const motorPara = (sep) => (sep === ";" ? (motorPuntoComa ||= Motor.crearMotor(M, { comaDecimal: true })) : motor);
  const META = M.meta, PERFIL = M.perfil, NOMBRES = M.nombres, ETIQUETAS = M.etiquetas, COLUMNAS = M.columnas;
  const P_UMBRAL = motor.umbralProbabilidad;
  const NO_INFORMADO = "__no_informado__";
  const $ = (sel, raiz = document) => raiz.querySelector(sel);

  // ---------------------------------------------------------------------
  // Íconos (Lucide, la librería de íconos de shadcn/ui)
  // ---------------------------------------------------------------------
  const ICONOS = {
    shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    panel: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/>',
    chevronRight: '<path d="m9 18 6-6-6-6"/>',
    chevronLeft: '<path d="m15 18-6-6 6-6"/>',
    chevronsLeft: '<path d="m11 17-5-5 5-5"/><path d="m18 17-5-5 5-5"/>',
    chevronsRight: '<path d="m6 17 5-5-5-5"/><path d="m13 17 5-5-5-5"/>',
    chevronDown: '<path d="m6 9 6 6 6-6"/>',
    upDown: '<path d="m7 15 5 5 5-5"/><path d="m7 9 5-5 5 5"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    github: '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
    sparkles: '<path d="M9.94 14.06 4 20"/><path d="m14 4 1.5 3.5L19 9l-3.5 1.5L14 14l-1.5-3.5L9 9l3.5-1.5Z"/><path d="M5 3v4"/><path d="M3 5h4"/>',
    user: '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    wallet: '<path d="M19 7V4a1 1 0 0 0-1-1H5a2 2 0 0 0 0 4h15a1 1 0 0 1 1 1v4h-3a2 2 0 0 0 0 4h3a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1"/><path d="M3 5v14a2 2 0 0 0 2 2h15a1 1 0 0 0 1-1v-4"/>',
    banknote: '<rect width="20" height="12" x="2" y="6" rx="2"/><circle cx="12" cy="12" r="2"/><path d="M6 12h.01M18 12h.01"/>',
    table: '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>',
    book: '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',
    help: '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    alert: '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    triangle: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    circleCheck: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    circleX: '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    minus: '<path d="M5 12h14"/>',
    plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
    upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M8 13h2"/><path d="M14 13h2"/><path d="M8 17h2"/><path d="M14 17h2"/>',
    loader: '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
    gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    chart: '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M7 16h8"/><path d="M7 11h12"/><path d="M7 6h3"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    rotate: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    arrowUp: '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    arrowDown: '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    arrowUpDown: '<path d="m21 16-4 4-4-4"/><path d="M17 20V4"/><path d="m3 8 4-4 4 4"/><path d="M7 4v16"/>',
    scale: '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    cpu: '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    lock: '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    target: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  };
  const icono = (n, clase = "") => `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"${clase ? ` class="${clase}"` : ""}>${ICONOS[n]}</svg>`;
  function colocarIconos(raiz = document) {
    raiz.querySelectorAll("[data-icono]").forEach((e) => { e.innerHTML = icono(e.dataset.icono); e.removeAttribute("data-icono"); });
    raiz.querySelectorAll("[data-icono-izq]").forEach((e) => { e.insertAdjacentHTML("afterbegin", icono(e.dataset.iconoIzq)); e.removeAttribute("data-icono-izq"); });
  }

  // ---------------------------------------------------------------------
  // Utilidades de formato
  // ---------------------------------------------------------------------
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct1 = (p) => `${(p * 100).toFixed(1)}%`;
  const cop = (v) => `$ ${Motor.miles(v)}`;
  const etiqueta = (col, v) => (ETIQUETAS[col] && ETIQUETAS[col][v]) || String(v).replace(/_/g, " ");
  const enMinuscula = (t) => t.slice(0, 1).toLowerCase() + t.slice(1);
  function mostrar(col, v) {
    if (Motor.esNulo(v)) return "(faltante)";
    if (M.num_cols.includes(col)) {
      const n = Motor.aNumero(v);
      return Number.isNaN(n) ? String(v) : Motor.miles(n);
    }
    return etiqueta(col, v);
  }
  const plural = (n, uno, varios) => `${n} ${n === 1 ? uno : varios}`;
  const esperar = (ms) => new Promise((r) => setTimeout(r, ms));
  const siguienteCuadro = () => new Promise((r) => { requestAnimationFrame(() => r()); setTimeout(r, 50); });
  const animacionReducida = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function contar(el, hasta, formato, duracion = 900) {
    if (animacionReducida()) { el.textContent = formato(hasta); return; }
    let inicio = null, terminado = false;
    const paso = (t) => {
      if (terminado) return;
      if (inicio === null) inicio = t;
      const x = Math.min(Math.max((t - inicio) / duracion, 0), 1), e = 1 - Math.pow(1 - x, 3);
      el.textContent = formato(hasta * e);
      if (x < 1) requestAnimationFrame(paso); else terminado = true;
    };
    requestAnimationFrame(paso);
    // Si el navegador pausa los cuadros (pestaña en segundo plano), igual queda el valor final
    setTimeout(() => { if (!terminado) { terminado = true; el.textContent = formato(hasta); } }, duracion + 150);
  }

  // ---------------------------------------------------------------------
  // Tooltip (Radix Tooltip)
  // ---------------------------------------------------------------------
  let tooltipEl = null, tooltipDe = null;
  function mostrarTooltip(objetivo) {
    ocultarTooltip();
    const texto = objetivo.getAttribute("data-tooltip");
    if (!texto) return;
    tooltipDe = objetivo;
    tooltipEl = document.createElement("div");
    tooltipEl.className = "tooltip";
    tooltipEl.setAttribute("role", "tooltip");
    tooltipEl.innerHTML = esc(texto) + (objetivo.dataset.atajo ? `<kbd>${esc(objetivo.dataset.atajo)}</kbd>` : "");
    document.body.appendChild(tooltipEl);
    const r = objetivo.getBoundingClientRect(), t = tooltipEl.getBoundingClientRect();
    let lado = "arriba", top = r.top - t.height - 8;
    if (top < 8) { lado = "abajo"; top = r.bottom + 8; }
    let left = Math.min(Math.max(r.left + r.width / 2 - t.width / 2, 8), window.innerWidth - t.width - 8);
    tooltipEl.dataset.lado = lado;
    tooltipEl.style.top = `${top}px`;
    tooltipEl.style.left = `${left}px`;
    tooltipEl.style.setProperty("--flecha", `${r.left + r.width / 2 - left}px`);
  }
  function ocultarTooltip() { if (tooltipEl) tooltipEl.remove(); tooltipEl = null; tooltipDe = null; }
  let temporizadorTooltip = null;
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest && e.target.closest("[data-tooltip]");
    if (t === tooltipDe) return;
    clearTimeout(temporizadorTooltip);
    if (!t) { ocultarTooltip(); return; }
    temporizadorTooltip = setTimeout(() => mostrarTooltip(t), 350);
  });
  document.addEventListener("focusin", (e) => { const t = e.target.closest && e.target.closest("[data-tooltip]"); if (t && e.target.matches(":focus-visible")) mostrarTooltip(t); });
  document.addEventListener("focusout", ocultarTooltip);
  window.addEventListener("scroll", ocultarTooltip, true);

  // ---------------------------------------------------------------------
  // Toast (Sonner)
  // ---------------------------------------------------------------------
  function toast(tipo, titulo, desc = "") {
    const el = document.createElement("div");
    el.className = "toast";
    el.dataset.tipo = tipo;
    el.setAttribute("role", tipo === "error" ? "alert" : "status");
    el.innerHTML = `${icono({ exito: "circleCheck", error: "circleX", info: "info" }[tipo])}<div class="toast-titulo">${esc(titulo)}</div>
      <button class="toast-cerrar" aria-label="Cerrar">${icono("x")}</button>${desc ? `<div class="toast-desc">${esc(desc)}</div>` : ""}`;
    const cerrar = () => { el.classList.add("saliendo"); setTimeout(() => el.remove(), 250); };
    el.querySelector(".toast-cerrar").addEventListener("click", cerrar);
    $("#toaster").appendChild(el);
    while ($("#toaster").children.length > 3) $("#toaster").firstElementChild.remove();
    setTimeout(cerrar, tipo === "error" ? 7000 : 4200);
  }

  // ---------------------------------------------------------------------
  // Select (Radix Select): disparador + lista flotante, con teclado
  // ---------------------------------------------------------------------
  let selectAbierto = null;
  function crearSelect({ id, opciones, valor, alCambiar }) {
    const boton = document.createElement("button");
    boton.type = "button"; boton.id = id; boton.className = "select-disparador";
    boton.setAttribute("aria-haspopup", "listbox"); boton.setAttribute("aria-expanded", "false");
    let actual = valor, lista = null, activa = 0, busqueda = "", tBusqueda = null;
    const pintar = () => {
      const o = opciones.find((x) => x.valor === actual);
      boton.innerHTML = `<span class="valor${o && o.especial ? " vacio" : ""}">${esc(o ? o.texto : "")}</span>${icono("upDown")}`;
    };
    function cerrar(devolverFoco = true) {
      if (!lista) return;
      const l = lista; lista = null; selectAbierto = null;
      boton.setAttribute("aria-expanded", "false");
      l.classList.add("cerrando");
      setTimeout(() => l.remove(), 100);
      if (devolverFoco) boton.focus();
    }
    function marcar(i) {
      activa = (i + opciones.length) % opciones.length;
      lista.querySelectorAll(".opcion").forEach((el, j) => el.dataset.activa = j === activa ? "true" : "false");
      const el = lista.querySelectorAll(".opcion")[activa];
      el.scrollIntoView({ block: "nearest" });
      lista.setAttribute("aria-activedescendant", el.id);
    }
    function elegir(i) {
      const nuevo = opciones[i].valor;
      cerrar();
      if (nuevo !== actual) { actual = nuevo; pintar(); alCambiar && alCambiar(actual); }
    }
    function abrir() {
      if (selectAbierto) selectAbierto();
      selectAbierto = () => cerrar(false);
      lista = document.createElement("div");
      lista.className = "popover"; lista.setAttribute("role", "listbox"); lista.tabIndex = -1;
      lista.innerHTML = opciones.map((o, i) => `${o.especial && i > 0 ? '<div class="opcion-sep" role="separator"></div>' : ""}
        <div class="opcion${o.especial ? " especial" : ""}" role="option" id="${id}-op-${i}" aria-selected="${o.valor === actual}" data-i="${i}">
        ${esc(o.texto)}<span class="check">${o.valor === actual ? icono("check") : ""}</span></div>`).join("");
      document.body.appendChild(lista);
      const r = boton.getBoundingClientRect();
      lista.style.minWidth = `${r.width}px`;
      const alto = lista.getBoundingClientRect().height;
      if (r.bottom + alto + 8 > window.innerHeight && r.top - alto - 8 > 0) { lista.dataset.lado = "arriba"; lista.style.top = `${r.top - alto - 4}px`; }
      else lista.style.top = `${r.bottom + 4}px`;
      lista.style.left = `${Math.min(r.left, window.innerWidth - lista.getBoundingClientRect().width - 8)}px`;
      boton.setAttribute("aria-expanded", "true");
      marcar(Math.max(0, opciones.findIndex((o) => o.valor === actual)));
      lista.addEventListener("pointermove", (e) => { const o = e.target.closest(".opcion"); if (o) marcar(+o.dataset.i); });
      lista.addEventListener("click", (e) => { const o = e.target.closest(".opcion"); if (o) elegir(+o.dataset.i); });
      lista.addEventListener("keydown", teclas);
      lista.focus({ preventScroll: true });
    }
    function teclas(e) {
      if (e.key === "ArrowDown") { e.preventDefault(); marcar(activa + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); marcar(activa - 1); }
      else if (e.key === "Home") { e.preventDefault(); marcar(0); }
      else if (e.key === "End") { e.preventDefault(); marcar(opciones.length - 1); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); elegir(activa); }
      else if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); cerrar(); }
      else if (e.key.length === 1) {
        busqueda += e.key.toLowerCase(); clearTimeout(tBusqueda); tBusqueda = setTimeout(() => (busqueda = ""), 600);
        const i = opciones.findIndex((o) => o.texto.toLowerCase().startsWith(busqueda));
        if (i >= 0) marcar(i);
      }
    }
    boton.addEventListener("click", () => (lista ? cerrar() : abrir()));
    boton.addEventListener("keydown", (e) => { if (["ArrowDown", "ArrowUp"].includes(e.key)) { e.preventDefault(); abrir(); } });
    pintar();
    return {
      el: boton,
      get: () => actual,
      set: (v) => { actual = v; pintar(); },
    };
  }
  document.addEventListener("pointerdown", (e) => {
    if (selectAbierto && !e.target.closest(".popover") && !e.target.closest(".select-disparador")) selectAbierto();
  });
  window.addEventListener("resize", () => selectAbierto && selectAbierto());
  document.addEventListener("scroll", (e) => { if (selectAbierto && !(e.target.closest && e.target.closest(".popover"))) selectAbierto(); }, true);

  // ---------------------------------------------------------------------
  // Campo numérico con pasos (como st.number_input)
  // ---------------------------------------------------------------------
  function crearNumero({ id, valor, min, max, paso, miles = false, prefijo = "", alCambiar }) {
    const caja = document.createElement("div");
    caja.className = `numero${prefijo ? " con-prefijo" : ""}`;
    caja.innerHTML = `${prefijo ? `<span class="prefijo">${esc(prefijo)}</span>` : ""}
      <input class="input" id="${id}" inputmode="numeric" autocomplete="off" spellcheck="false">
      <div class="pasos"><button type="button" tabindex="-1" aria-label="Disminuir">${icono("minus")}</button>
      <button type="button" tabindex="-1" aria-label="Aumentar">${icono("plus")}</button></div>`;
    const input = $("input", caja), [menos, mas] = caja.querySelectorAll(".pasos button");
    let actual = valor;
    const formatear = (v) => (Number.isNaN(v) ? "" : miles ? Motor.miles(v) : String(v));
    const leer = () => { const s = input.value.replace(/\./g, "").replace(/\s/g, "").replace(",", "."); return s === "" ? NaN : Number(s); };
    const validar = () => input.setAttribute("aria-invalid", Number.isNaN(actual) || actual < min || actual > max ? "true" : "false");
    const fijar = (v, avisar = true) => {
      actual = Number.isNaN(v) ? NaN : Math.min(max, Math.max(min, v));
      input.value = formatear(actual); validar();
      if (avisar && alCambiar) alCambiar(actual);
    };
    input.addEventListener("input", () => {
      const v = leer();
      actual = v; validar();
      if (miles && !Number.isNaN(v)) {                     // formato en vivo conservando el cursor al final
        const fin = input.selectionStart === input.value.length;
        input.value = Motor.miles(v);
        if (fin) input.setSelectionRange(input.value.length, input.value.length);
      }
      alCambiar && alCambiar(actual);
    });
    input.addEventListener("blur", () => fijar(leer()));
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowUp") { e.preventDefault(); fijar((Number.isNaN(actual) ? min : actual) + paso); }
      if (e.key === "ArrowDown") { e.preventDefault(); fijar((Number.isNaN(actual) ? min : actual) - paso); }
    });
    menos.addEventListener("click", () => fijar((Number.isNaN(actual) ? min : actual) - paso));
    mas.addEventListener("click", () => fijar((Number.isNaN(actual) ? min : actual) + paso));
    fijar(valor, false);
    return {
      el: caja, input,
      get: () => actual,
      set: (v) => fijar(v, false),
      deshabilitar: (si) => { input.disabled = si; menos.disabled = si; mas.disabled = si; },
    };
  }

  function crearCheckbox({ id, marcado, texto, alCambiar }) {
    const fila = document.createElement("label");
    fila.className = "check-fila";
    fila.innerHTML = `<button type="button" class="checkbox" role="checkbox" id="${id}" aria-checked="${marcado}">${icono("check")}</button><span>${esc(texto)}</span>`;
    const caja = $(".checkbox", fila);
    let valor = marcado;
    const fijar = (v, avisar) => { valor = v; caja.setAttribute("aria-checked", String(v)); if (avisar && alCambiar) alCambiar(v); };
    fila.addEventListener("click", (e) => { e.preventDefault(); fijar(!valor, true); caja.focus(); });
    return { el: fila, get: () => valor, set: (v) => fijar(v, false) };
  }

  // ---------------------------------------------------------------------
  // Barra lateral, encabezado y pestañas
  // ---------------------------------------------------------------------
  const app = $("#app");
  const esMovil = () => window.matchMedia("(max-width: 767px)").matches;
  function alternarBarra() {
    if (esMovil()) app.dataset.movil = app.dataset.movil === "abierta" ? "cerrada" : "abierta";
    else {
      app.dataset.barra = app.dataset.barra === "cerrada" ? "abierta" : "cerrada";
      try { localStorage.setItem("barra", app.dataset.barra); } catch (e) { /* almacenamiento no disponible */ }
    }
  }
  try { if (localStorage.getItem("barra") === "cerrada") app.dataset.barra = "cerrada"; } catch (e) { /* idem */ }
  $("#btn-barra").addEventListener("click", alternarBarra);
  $("#velo").addEventListener("click", () => (app.dataset.movil = "cerrada"));
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") { e.preventDefault(); alternarBarra(); }
    if (e.key === "Escape" && app.dataset.movil === "abierta") app.dataset.movil = "cerrada";
  });
  if (/Mac|iPhone|iPad/.test(navigator.platform)) $("#btn-barra").dataset.atajo = "⌘ B";
  $("#enlace-repo").href = META.repositorio;

  function pintarBarra() {
    const hp = META.hiperparametros, m = META.metricas_test;
    const metrica = (nombre, valor) => `<div class="metrica-barra"><div class="fila"><span>${nombre}</span><span>${valor.toFixed(3)}</span></div>
      <div class="mini-progreso"><span style="width:${(valor * 100).toFixed(1)}%"></span></div></div>`;
    $("#barra-contenido").innerHTML = `
      <div class="grupo"><div class="grupo-titulo">Modelo</div>
        <div class="kv"><span class="kv-clave">${icono("cpu")}Algoritmo</span><span class="kv-valor">${esc(META.nombre)}</span></div>
        <div class="kv"><span class="kv-clave">${icono("scale")}C</span><span class="kv-valor">${hp.C}</span></div>
        <div class="kv"><span class="kv-clave">${icono("target")}γ</span><span class="kv-valor">${hp.gamma}</span></div>
        <div class="kv"><span class="kv-clave">${icono("sparkles")}Vectores de soporte</span><span class="kv-valor">${Motor.miles(META.n_vectores_soporte)}</span></div>
      </div>
      <div class="grupo"><div class="grupo-titulo">Desempeño en prueba · 1.300 clientes</div>
        ${metrica("ROC-AUC", m["ROC-AUC"])}
        ${metrica('Recall de "malo"', m["Recall (malo)"])}
        <div class="kv"><span class="kv-clave">Costo medio 5:1</span><span class="kv-valor">${m["Costo medio (5:1)"].toFixed(3)}</span></div>
      </div>
      <div class="grupo"><div class="grupo-titulo">Umbral de decisión</div>
        <div class="umbral-caja"><span class="grande">${pct1(P_UMBRAL)}</span>
          <span class="pequeno">Se rechaza o revisa si P(malo) ≥ ${pct1(P_UMBRAL)}. Rechazar a un buen cliente cuesta 1; aprobar a uno malo, 5.</span></div>
      </div>`;
    $("#barra-pie").innerHTML = `<span>Entrenado el ${esc(META.fecha_entrenamiento)} con scikit-learn ${esc(META.versiones["scikit-learn"])}.</span>
      <span>${icono("lock").replace("<svg", '<svg style="display:inline;width:.75rem;height:.75rem;vertical-align:-1px;margin-right:.25rem"')}Todo se calcula en su navegador.</span>`;
  }

  const PESTANAS = { solicitud: "Evaluar una solicitud", lote: "Evaluar un lote (CSV)", acerca: "Acerca del modelo" };
  function activarPestana(nombre, foco = false) {
    Object.keys(PESTANAS).forEach((n) => {
      const tab = $(`#tab-${n}`), panel = $(`#panel-${n}`), activa = n === nombre;
      tab.setAttribute("aria-selected", String(activa)); tab.tabIndex = activa ? 0 : -1;
      if (activa && panel.hidden) { panel.hidden = false; panel.style.animation = "none"; panel.offsetHeight; panel.style.animation = ""; }
      else if (!activa) panel.hidden = true;
      if (activa && foco) tab.focus();
    });
    const migas = $("#migas-actual");
    migas.textContent = PESTANAS[nombre]; migas.classList.remove("cambio"); migas.offsetHeight; migas.classList.add("cambio");
    if (esMovil()) app.dataset.movil = "cerrada";
  }
  Object.keys(PESTANAS).forEach((n, i, todas) => {
    const tab = $(`#tab-${n}`);
    tab.addEventListener("click", () => activarPestana(n));
    tab.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (d) { e.preventDefault(); activarPestana(todas[(i + d + todas.length) % todas.length], true); }
    });
  });

  // ---------------------------------------------------------------------
  // Resultado de una solicitud (decisión + explicación SHAP)
  // ---------------------------------------------------------------------
  const tokens = new WeakMap();

  function alertaHallazgo(h) {
    const clase = { error: "alerta-error", advertencia: "alerta-aviso", info: "alerta-info" }[h.nivel];
    const ic = { error: "alert", advertencia: "triangle", info: "info" }[h.nivel];
    return `<div class="alerta ${clase}">${icono(ic)}<div class="alerta-titulo">${esc(NOMBRES[h.variable] || h.variable)}</div>
      <div class="alerta-desc">${esc(h.detalle)}${h.filas > 1 ? ` (${h.filas} filas)` : ""}</div></div>`;
  }

  function graficoShap(tabla) {
    const filas = tabla.slice(0, 8);
    const v = filas.map((f) => f.contrib);
    let lo = Math.min(0, ...v), hi = Math.max(0, ...v);
    const rango = hi - lo || 1;
    if (lo < 0) lo -= rango * 0.03;
    if (hi > 0) hi += rango * 0.03;
    const pos = (x) => ((x - lo) / (hi - lo)) * 100, cero = pos(0);
    // Marcas "redondas" del eje
    const crudo = (hi - lo) / 4, mag = Math.pow(10, Math.floor(Math.log10(crudo)));
    const paso = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= crudo);
    const ticks = [];
    for (let t = Math.ceil(lo / paso) * paso; t <= hi + 1e-12; t += paso) ticks.push(Math.abs(t) < 1e-12 ? 0 : t);
    const dec = paso < 0.1 ? 2 : 1;
    const filasHtml = filas.map((f, i) => {
      const p = pos(f.contrib), izq = Math.min(p, cero), ancho = Math.abs(p - cero), sube = f.contrib > 0;
      const num = `${f.contrib > 0 ? "+" : ""}${f.contrib.toFixed(3)}`;
      return `<div class="shap-fila" data-tooltip="${esc(`${f.variable}: ${f.valor} · ${sube ? "aumenta" : "reduce"} el puntaje en ${Math.abs(f.contrib).toFixed(3)}`)}">
        <div class="shap-etiqueta"><span class="var">${esc(f.variable)}</span><span class="val">${esc(f.valor)}</span></div>
        <div class="shap-pista"><div class="shap-eje" style="left:${cero}%"></div>
          <div class="shap-barra ${sube ? "sube" : "baja"}" style="left:${izq}%;width:${ancho}%;--origen:${sube ? "left" : "right"};animation-delay:${i * 55}ms"></div>
          <span class="shap-num" style="${sube ? `left:calc(${p}% + .375rem)` : `right:calc(${100 - p}% + .375rem)`}">${num}</span>
        </div></div>`;
    }).join("");
    return `<div class="shap" role="img" aria-label="Contribución de cada variable al puntaje de riesgo">${filasHtml}</div>
      <div class="shap-ejes"><span></span><div class="shap-ticks">${ticks.map((t) => `<span style="left:${pos(t)}%">${t.toFixed(dec)}</span>`).join("")}</div></div>
      <div class="shap-leyenda"><span><i class="cuadro baja"></i>Reduce el riesgo</span><span>Contribución al puntaje del SVM (SHAP)</span><span><i class="cuadro sube"></i>Aumenta el riesgo</span></div>`;
  }

  /** Muestra la evaluación de `fila` en `cont`. Devuelve una promesa que termina con la explicación. */
  async function renderizarResultado(cont, fila, m = motor) {
    const token = (tokens.get(cont) || 0) + 1;
    tokens.set(cont, token);
    const hallazgos = m.verificarCoherencia([fila], COLUMNAS, false);
    const errores = hallazgos.filter((h) => h.nivel === "error");
    if (errores.length) {
      cont.innerHTML = `<div class="card aparecer"><div class="card-header">
          <div class="card-titulo" style="color:var(--destructive)">${icono("circleX")}Los datos no son coherentes</div>
          <div class="card-desc">Corrija los datos: el modelo no puntúa solicitudes con errores de captura.</div></div>
        <div class="card-contenido"><div class="lista-alertas">${errores.map(alertaHallazgo).join("")}</div></div></div>`;
      cont.scrollIntoView({ behavior: animacionReducida() ? "auto" : "smooth", block: "nearest" });
      return;
    }
    const p = m.probabilidad(fila), rechazar = p >= P_UMBRAL, tipo = rechazar ? "rechazar" : "aprobar";
    const delta = (p - P_UMBRAL) * 100;
    const ancho = Math.min(p / 0.8, 1) * 100, marca = (P_UMBRAL / 0.8) * 100;
    const observaciones = hallazgos.length ? `<div class="colapsable" data-abierto="false">
        <button class="colapsable-boton" type="button" aria-expanded="false">Observaciones sobre los datos (${hallazgos.length})${icono("chevronDown")}</button>
        <div class="colapsable-cuerpo"><div><div class="lista-alertas">${hallazgos.map(alertaHallazgo).join("")}</div></div></div></div>` : "";

    cont.innerHTML = `<div class="resultado">
      <div class="card aparecer">
        <div class="card-header"><div class="card-titulo">${icono("gauge")}Decisión</div>
          <div class="card-desc">Umbral de costo mínimo 5:1, elegido por validación cruzada en entrenamiento</div></div>
        <div class="card-contenido" style="display:grid;gap:1.375rem">
          <div class="decision" data-tipo="${tipo}"><div class="decision-icono">${icono(rechazar ? "circleX" : "circleCheck")}</div>
            <div><div class="decision-titulo">${rechazar ? "Rechazar o revisar" : "Aprobar"}</div>
            <div class="decision-texto">Riesgo por ${rechazar ? "encima" : "debajo"} del umbral de costo mínimo.</div></div></div>
          <div class="metrica"><span class="metrica-etiqueta">Probabilidad estimada de incumplimiento</span>
            <div class="metrica-fila"><span class="metrica-valor" data-r="prob">0.0%</span>
            <span class="delta ${delta >= 0 ? "sube" : "baja"}">${icono(delta >= 0 ? "arrowUp" : "arrowDown")}${delta >= 0 ? "+" : ""}${delta.toFixed(1)} p.p. frente al umbral</span></div></div>
          <div class="riesgo" aria-label="Probabilidad ${pct1(p)} frente a un umbral de ${pct1(P_UMBRAL)}">
            <div class="riesgo-pista"><div class="riesgo-relleno" data-tipo="${tipo}"></div><div class="riesgo-marca" style="left:${marca}%"></div></div>
            <div class="riesgo-escala"><span style="left:0">0%</span><span class="centro" style="left:${marca}%">umbral ${pct1(P_UMBRAL)}</span><span style="right:0">80%</span></div>
          </div>
          <p class="texto-pequeno">Rechazar a un buen cliente cuesta 1 y aprobar a uno malo cuesta 5, así que el modelo rechaza
            desde un riesgo de ${pct1(P_UMBRAL)}, muy por debajo del 50%. Tasa media de incumplimiento en entrenamiento: 22.5%.</p>
          ${observaciones}
        </div>
      </div>
      <div class="card aparecer" style="animation-delay:80ms">
        <div class="card-header"><div class="card-titulo">${icono("chart")}¿Por qué?</div>
          <div class="card-desc">Lo que cada variable suma o resta al puntaje del SVM frente a un solicitante promedio</div></div>
        <div class="card-contenido" data-r="explicacion">
          <div class="cargando-shap"><div class="texto-pequeno" style="display:flex;align-items:center;gap:.5rem">${icono("loader", "girando")}
            <span>Calculando la explicación (SHAP)… <span class="mono" data-r="avance">0%</span></span></div>
            <div class="progreso"><span data-r="barra-avance"></span></div>
            ${Array.from({ length: 8 }, (_, i) => `<div class="skeleton" style="height:1.5rem;width:${92 - ((i * 37) % 45)}%;margin-left:auto"></div>`).join("")}
          </div></div>
      </div></div>`;
    const col = cont.querySelector(".colapsable");
    if (col) col.querySelector(".colapsable-boton").addEventListener("click", (e) => {
      const abierto = col.dataset.abierto !== "true";
      col.dataset.abierto = String(abierto); e.currentTarget.setAttribute("aria-expanded", String(abierto));
    });
    contar($('[data-r="prob"]', cont), p, pct1);
    await siguienteCuadro();
    $(".riesgo-relleno", cont).style.width = `${ancho}%`;
    cont.scrollIntoView({ behavior: animacionReducida() ? "auto" : "smooth", block: "nearest" });

    // SHAP exacto, por partes para no bloquear la interfaz
    const calculo = m.crearExplicacion(fila);
    await esperar(60);
    while (!calculo.terminado()) {
      if (tokens.get(cont) !== token) return;                   // hubo otra evaluación
      const avance = calculo.paso(2);
      const a = $('[data-r="avance"]', cont), barra = $('[data-r="barra-avance"]', cont);
      if (a) a.textContent = `${Math.round(avance * 100)}%`;
      if (barra) barra.style.width = `${avance * 100}%`;
      await esperar(0);
    }
    if (tokens.get(cont) !== token) return;
    const exp = calculo.resultado();
    const tabla = COLUMNAS.map((c, j) => ({ variable: NOMBRES[c], valor: mostrar(c, fila[c]), contrib: exp.valores[j] }))
      .sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
    const sube = tabla.filter((t) => t.contrib > 0).slice(0, 3), baja = tabla.filter((t) => t.contrib < 0).slice(0, 3);
    const frase = (t) => t.map((x) => `${esc(enMinuscula(x.variable))} = ${esc(x.valor)}`).join("; ");
    $('[data-r="explicacion"]', cont).innerHTML = `<div class="aparecer" style="display:grid;gap:1.25rem">${graficoShap(tabla)}
      <div class="frases">${sube.length ? `<div><strong>Aumentan el riesgo:</strong> ${frase(sube)}.</div>` : ""}
        ${baja.length ? `<div><strong>Lo reducen:</strong> ${frase(baja)}.</div>` : ""}</div>
      <p class="texto-pequeno">Cada barra es lo que esa variable suma o resta al puntaje del SVM frente a un solicitante promedio
        de entrenamiento (valores SHAP exactos sobre 50 clientes de referencia). Las barras suman la diferencia entre el
        puntaje de esta solicitud (${exp.puntaje.toFixed(3)}) y el promedio (${exp.base.toFixed(3)}).</p></div>`;
  }

  // ---------------------------------------------------------------------
  // Pestaña 1: evaluar una solicitud
  // ---------------------------------------------------------------------
  const CATS = PERFIL.categorias;
  const opcionesDe = (col, faltante = false) => [
    ...CATS[col].map((v) => ({ valor: v, texto: etiqueta(col, v) })),
    ...(faltante ? [{ valor: NO_INFORMADO, texto: "(no informado)", especial: true }] : []),
  ];
  const rangoSaldo = (cat) => PERFIL.saldo_por_categoria[cat] || { min: 0, max: 1e7 };
  const saldoSugerido = (cat) => { const r = rangoSaldo(cat); return r.min > 0 ? Math.round(Math.sqrt(r.min * r.max) / 1000) * 1000 : 500000; };
  const PREDETERMINADO = {
    edad: 35, genero: CATS.genero[0], tipo_ocupacion: CATS.tipo_ocupacion[3], tipo_vivienda: CATS.tipo_vivienda[2],
    saldo_categoria: CATS.saldo_categoria[2], nivel_ahorro: CATS.nivel_ahorro[3], estado_pago_previo: CATS.estado_pago_previo[0],
    monto_credito_cop: 4000000, plazo_meses: 18, proposito_credito: CATS.proposito_credito[5],
  };
  const formulario = {};

  function campo(id, texto, control, ayuda = "") {
    const div = document.createElement("div");
    div.className = "campo";
    div.innerHTML = `<label class="etiqueta" for="${id}">${esc(texto)}${ayuda ? `<span class="ayuda" tabindex="0" data-tooltip="${esc(ayuda)}" aria-label="${esc(ayuda)}">${icono("help")}</span>` : ""}</label>`;
    div.appendChild(control);
    return div;
  }

  function construirSolicitud() {
    const panel = $("#panel-solicitud");
    panel.innerHTML = `<div class="card">
        <div class="card-header borde"><div class="card-titulo">Datos de la solicitud</div>
          <div class="card-desc">Ingrese los datos de la solicitud. Los campos marcados como <em>(no informado)</em> se imputan con la moda de entrenamiento, igual que en el pipeline.</div>
          <div class="card-accion"><button class="btn btn-ghost btn-sm" type="button" id="btn-restablecer">${icono("rotate")}Restablecer</button></div></div>
        <div class="card-contenido"><div class="form-grid">
          <div class="form-seccion" id="sec-solicitante"><div class="form-seccion-titulo"><span class="icono-caja">${icono("user")}</span>Solicitante</div></div>
          <div class="form-seccion" id="sec-financiera"><div class="form-seccion-titulo"><span class="icono-caja">${icono("wallet")}</span>Situación financiera</div></div>
          <div class="form-seccion" id="sec-credito"><div class="form-seccion-titulo"><span class="icono-caja">${icono("banknote")}</span>Crédito solicitado</div></div>
        </div></div>
        <div class="card-pie borde"><button class="btn btn-primario" type="button" id="btn-evaluar">${icono("sparkles")}Evaluar solicitud</button>
          <span class="texto-pequeno">El cálculo se hace en su navegador: ningún dato sale de este equipo.</span></div>
      </div>
      <div id="resultado-solicitud"></div>`;
    const cambio = () => marcarDesactualizado();
    const sel = (col, faltante, valor) => (formulario[col] = crearSelect({ id: `f-${col}`, opciones: opcionesDe(col, faltante), valor, alCambiar: cambio }));

    formulario.edad = crearNumero({ id: "f-edad", valor: PREDETERMINADO.edad, min: 0, max: 100, paso: 1, alCambiar: cambio });
    const solicitante = $("#sec-solicitante");
    solicitante.append(campo("f-edad", NOMBRES.edad, formulario.edad.el),
      campo("f-genero", NOMBRES.genero, sel("genero", false, PREDETERMINADO.genero).el,
        "No interviene en el riesgo de la simulación; se incluye para verificar que el modelo no lo use."),
      campo("f-tipo_ocupacion", NOMBRES.tipo_ocupacion, sel("tipo_ocupacion", true, PREDETERMINADO.tipo_ocupacion).el),
      campo("f-tipo_vivienda", NOMBRES.tipo_vivienda, sel("tipo_vivienda", false, PREDETERMINADO.tipo_vivienda).el));

    formulario.saldo = crearNumero({ id: "f-saldo", valor: saldoSugerido(PREDETERMINADO.saldo_categoria), min: 0, max: 1e10, paso: 50000, miles: true, prefijo: "$", alCambiar: cambio });
    const campoSaldo = campo("f-saldo", NOMBRES.saldo_cuenta_cop, formulario.saldo.el, " ");
    formulario.saldo_categoria = crearSelect({
      id: "f-saldo_categoria", opciones: opcionesDe("saldo_categoria"), valor: PREDETERMINADO.saldo_categoria,
      alCambiar: (v) => { actualizarSaldo(v); cambio(); },
    });
    function actualizarSaldo(cat) {
      const ayuda = $(".ayuda", campoSaldo);
      if (cat === "sin_cuenta") {
        formulario.saldo.set(0); formulario.saldo.deshabilitar(true);
        ayuda.dataset.tooltip = "Sin cuenta: el saldo es cero por definición.";
      } else {
        const r = rangoSaldo(cat);
        formulario.saldo.deshabilitar(false); formulario.saldo.set(saldoSugerido(cat));
        ayuda.dataset.tooltip = `En entrenamiento, la categoría va de ${cop(r.min)} a ${cop(r.max)}.`;
      }
      ayuda.setAttribute("aria-label", ayuda.dataset.tooltip);
    }
    $("#sec-financiera").append(campo("f-saldo_categoria", NOMBRES.saldo_categoria, formulario.saldo_categoria.el), campoSaldo,
      campo("f-nivel_ahorro", NOMBRES.nivel_ahorro, sel("nivel_ahorro", true, PREDETERMINADO.nivel_ahorro).el),
      campo("f-estado_pago_previo", NOMBRES.estado_pago_previo, sel("estado_pago_previo", false, PREDETERMINADO.estado_pago_previo).el,
        "La variable más influyente del modelo."));
    actualizarSaldo(PREDETERMINADO.saldo_categoria);

    formulario.monto = crearNumero({ id: "f-monto", valor: PREDETERMINADO.monto_credito_cop, min: 0, max: 1e10, paso: 250000, miles: true, prefijo: "$", alCambiar: cambio });
    const deslizador = document.createElement("div");
    deslizador.className = "deslizador";
    deslizador.innerHTML = `<div class="deslizador-cabecera"><label class="etiqueta" for="f-plazo">${esc(NOMBRES.plazo_meses)}</label>
        <span class="deslizador-valor" id="plazo-valor"></span></div>
      <input type="range" id="f-plazo" min="3" max="72" step="3" value="${PREDETERMINADO.plazo_meses}">
      <div class="marcas"><span>3</span><span>18</span><span>36</span><span>54</span><span>72</span></div>`;
    const rango = $("input", deslizador);
    const pintarPlazo = () => {
      rango.style.setProperty("--pct", `${((rango.value - 3) / 69) * 100}%`);
      $("#plazo-valor", deslizador).textContent = `${rango.value} meses`;
    };
    rango.addEventListener("input", () => { pintarPlazo(); cambio(); });
    pintarPlazo();
    formulario.plazo = { get: () => Number(rango.value), set: (v) => { rango.value = v; pintarPlazo(); } };
    $("#sec-credito").append(campo("f-monto", NOMBRES.monto_credito_cop, formulario.monto.el), deslizador,
      campo("f-proposito_credito", NOMBRES.proposito_credito, sel("proposito_credito", false, PREDETERMINADO.proposito_credito).el));

    $("#btn-evaluar").addEventListener("click", evaluarSolicitud);
    $("#btn-restablecer").addEventListener("click", () => {
      ["genero", "tipo_ocupacion", "tipo_vivienda", "saldo_categoria", "nivel_ahorro", "estado_pago_previo", "proposito_credito"]
        .forEach((c) => formulario[c].set(PREDETERMINADO[c]));
      formulario.edad.set(PREDETERMINADO.edad); formulario.monto.set(PREDETERMINADO.monto_credito_cop);
      formulario.plazo.set(PREDETERMINADO.plazo_meses); actualizarSaldo(PREDETERMINADO.saldo_categoria);
      $("#resultado-solicitud").innerHTML = ""; tokens.set($("#resultado-solicitud"), -1);
      toast("info", "Formulario restablecido", "Se recuperaron los valores iniciales.");
    });
    panel.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") evaluarSolicitud(); });
  }

  function filaFormulario() {
    const v = (c) => { const x = formulario[c].get(); return x === NO_INFORMADO ? null : x; };
    const cat = v("saldo_categoria");
    return {
      edad: formulario.edad.get(), genero: v("genero"), tipo_ocupacion: v("tipo_ocupacion"), tipo_vivienda: v("tipo_vivienda"),
      nivel_ahorro: v("nivel_ahorro"), saldo_categoria: cat, saldo_cuenta_cop: cat === "sin_cuenta" ? 0 : formulario.saldo.get(),
      monto_credito_cop: formulario.monto.get(), plazo_meses: formulario.plazo.get(), proposito_credito: v("proposito_credito"),
      estado_pago_previo: v("estado_pago_previo"),
    };
  }

  function evaluarSolicitud() {
    // Como st.number_input: los valores escritos se ajustan al rango permitido
    [formulario.edad, formulario.saldo, formulario.monto].forEach((f) => f.set(f.get()));
    const fila = filaFormulario();
    const vacios = [["edad", formulario.edad], ["saldo_cuenta_cop", formulario.saldo], ["monto_credito_cop", formulario.monto]]
      .filter(([c]) => Number.isNaN(fila[c]));
    if (vacios.length) {
      vacios[0][1].input.focus();
      toast("error", "Faltan datos", `Complete: ${vacios.map(([c]) => NOMBRES[c]).join(", ")}.`);
      return;
    }
    const cont = $("#resultado-solicitud");
    cont.dataset.vigente = "true";
    renderizarResultado(cont, fila);
  }

  function marcarDesactualizado() {
    const cont = $("#resultado-solicitud");
    const res = cont && cont.querySelector(".resultado");
    if (!res || cont.dataset.vigente !== "true") return;
    cont.dataset.vigente = "false";
    res.classList.add("desactualizado");
    res.insertAdjacentHTML("afterbegin", `<div class="alerta alerta-info aviso-desactualizado aparecer">${icono("info")}
      <div class="alerta-titulo">Los datos cambiaron desde la última evaluación</div>
      <div class="alerta-desc">Este resultado ya no corresponde al formulario. <button class="btn btn-outline btn-sm" type="button" style="margin-left:.5rem">Evaluar de nuevo</button></div></div>`);
    res.querySelector(".aviso-desactualizado button").addEventListener("click", evaluarSolicitud);
  }

  // ---------------------------------------------------------------------
  // Pestaña 2: evaluar un lote (CSV)
  // ---------------------------------------------------------------------
  const lote = { datos: null, archivo: null, error: null, usarEjemplo: true, res: [], filtro: "todas", busqueda: "", orden: null, pagina: 0, porPagina: 10 };
  let casillaEjemplo;

  function construirLote() {
    const panel = $("#panel-lote");
    panel.innerHTML = `<div class="card">
        <div class="card-header"><div class="card-titulo">${icono("file")}Archivo CSV</div>
          <div class="card-desc">Suba un CSV con las 11 columnas del modelo. Antes de puntuar se verifica el esquema, las categorías,
            los rangos, las reglas de negocio y la deriva frente a entrenamiento (PSI).</div></div>
        <div class="card-contenido" style="display:grid;gap:1rem">
          <div class="columnas-lista">${COLUMNAS.map((c) => `<code>${esc(c)}</code>`).join("")}</div>
          <div id="zona-archivo"></div>
          <div id="casilla-ejemplo"></div>
        </div></div>
      <div id="lote-salida" style="display:flex;flex-direction:column;gap:1.5rem"></div>`;
    casillaEjemplo = crearCheckbox({
      id: "usar-ejemplo", marcado: true, texto: "Usar el archivo de ejemplo (200 solicitudes de prueba y 4 con errores de captura)",
      alCambiar: (v) => { lote.usarEjemplo = v; procesarLote(); },
    });
    $("#casilla-ejemplo").appendChild(casillaEjemplo.el);
    pintarZona();
    procesarLote();
  }

  function pintarZona() {
    const zona = $("#zona-archivo");
    if (lote.archivo) {
      zona.innerHTML = `<div class="archivo"><span class="archivo-icono">${icono("file")}</span><div>
          <div class="archivo-nombre">${esc(lote.archivo.nombre)}</div>
          <div class="archivo-info">${(lote.archivo.tamano / 1024).toFixed(1)} KB${lote.datos ? ` · ${plural(lote.datos.filas.length, "fila", "filas")} · separador «${lote.datos.separador === "\t" ? "tab" : esc(lote.datos.separador)}»` : ""}</div></div>
        <button class="btn btn-ghost btn-icono-sm" type="button" id="quitar-archivo" aria-label="Quitar archivo" data-tooltip="Quitar archivo">${icono("x")}</button></div>`;
      $("#quitar-archivo").addEventListener("click", () => {
        lote.archivo = null; lote.datos = null; lote.error = null; lote.usarEjemplo = true; casillaEjemplo.set(true);
        pintarZona(); procesarLote();
      });
      return;
    }
    zona.innerHTML = `<div class="zona" tabindex="0" role="button" aria-label="Subir archivo CSV">
        <span class="zona-icono">${icono("upload")}</span>
        <div><strong>Arrastre un archivo CSV</strong> <span class="muted">o haga clic para seleccionarlo</span></div>
        <span class="texto-pequeno">Separado por comas o punto y coma · UTF-8 o Latin-1 (como lo exporta Excel)</span>
        <input type="file" accept=".csv,text/csv" hidden></div>`;
    const area = $(".zona", zona), input = $("input", zona);
    area.addEventListener("click", () => input.click());
    area.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", () => input.files[0] && cargarArchivo(input.files[0]));
    ["dragenter", "dragover"].forEach((t) => area.addEventListener(t, (e) => { e.preventDefault(); area.dataset.arrastrando = "true"; }));
    ["dragleave", "drop"].forEach((t) => area.addEventListener(t, (e) => { e.preventDefault(); area.dataset.arrastrando = "false"; }));
    area.addEventListener("drop", (e) => e.dataTransfer.files[0] && cargarArchivo(e.dataTransfer.files[0]));
  }

  async function cargarArchivo(archivo) {
    lote.archivo = { nombre: archivo.name, tamano: archivo.size };
    lote.usarEjemplo = false; casillaEjemplo.set(false);
    const texto = Motor.decodificar(await archivo.arrayBuffer());
    const r = Motor.leerCSV(texto);
    if (r.error) {
      lote.datos = null; lote.error = `No se pudo leer el archivo como CSV: ${r.error}`;
      toast("error", "No se pudo leer el archivo", r.error);
    } else {
      lote.datos = r; lote.error = null;
      toast("exito", "Archivo cargado", `${plural(r.filas.length, "fila", "filas")} y ${plural(r.columnas.length, "columna", "columnas")}.`);
    }
    pintarZona(); procesarLote();
  }

  function datosActivos() {
    if (lote.archivo) return lote.datos;
    if (lote.usarEjemplo) return { ...Motor.leerCSV(M.ejemplo_csv), ejemplo: true };
    return null;
  }

  function procesarLote() {
    const salida = $("#lote-salida");
    lote.pagina = 0;
    if (lote.error) { salida.innerHTML = `<div class="alerta alerta-error aparecer">${icono("alert")}<div class="alerta-titulo">No se pudo leer el archivo</div><div class="alerta-desc">${esc(lote.error)}</div></div>`; return; }
    const datos = datosActivos();
    if (!datos) { salida.innerHTML = ""; lote.res = []; return; }
    const m = motorPara(datos.separador);
    const { columnas, filas } = datos;
    const hall = m.verificarCoherencia(filas, columnas);
    const cuenta = { error: 0, advertencia: 0, info: 0 };
    hall.forEach((h) => (cuenta[h.nivel] += 1));
    const ICONO_NIVEL = { error: "alert", advertencia: "triangle", info: "info" };
    const BADGE = { error: "badge-error", advertencia: "badge-aviso", info: "badge-info" };
    const ETQ = { error: "errores", advertencia: "advertencias", info: "observaciones informativas" };
    let html = `<div class="card aparecer">
      <div class="card-header"><div class="card-titulo">${icono("shield")}Verificación de coherencia con entrenamiento</div>
        <div class="card-desc">${datos.ejemplo ? "Archivo de ejemplo" : esc(lote.archivo.nombre)} · ${plural(filas.length, "fila", "filas")}</div></div>
      <div class="card-contenido" style="display:grid;gap:1rem">${hall.length ? `
        <div class="conteos">${Object.keys(ETQ).map((n) => `<span class="badge ${BADGE[n]}">${icono(ICONO_NIVEL[n])}<strong>${cuenta[n]}</strong> ${ETQ[n]}</span>`).join("")}</div>
        <div class="tabla-marco"><div class="tabla-scroll"><table><thead><tr><th>Nivel</th><th>Variable</th><th class="num">Filas</th><th>Detalle</th></tr></thead><tbody>
          ${hall.map((h) => `<tr><td><span class="badge ${BADGE[h.nivel]}">${icono(ICONO_NIVEL[h.nivel])}${h.nivel}</span></td>
            <td>${esc(NOMBRES[h.variable] || h.variable)}</td><td class="num">${h.filas}</td><td style="white-space:normal;min-width:18rem">${esc(h.detalle)}</td></tr>`).join("")}
        </tbody></table></div></div>` : `<div class="alerta">${icono("circleCheck")}<div class="alerta-titulo">Sin hallazgos</div><div class="alerta-desc">El lote es coherente con los datos de entrenamiento.</div></div>`}
      </div></div>`;

    if (COLUMNAS.some((c) => !columnas.includes(c))) {
      salida.innerHTML = html + `<div class="alerta alerta-error aparecer">${icono("alert")}<div class="alerta-titulo">Faltan columnas</div>
        <div class="alerta-desc">El archivo no tiene todas las columnas que necesita el modelo; no se puntúa.</div></div>`;
      lote.res = [];
      return;
    }
    const conError = m.filasConError(filas, columnas);
    lote.res = filas.map((f, i) => ({ f, i })).filter(({ i }) => !conError[i]).map(({ f }, pos) => {
      const p = m.probabilidad(f);
      return { pos: pos + 1, fila: f, p, rechazar: p >= P_UMBRAL };
    });
    lote.columnas = columnas; lote.motor = m;
    const res = lote.res, n = res.length;
    if (!n) {
      salida.innerHTML = html + `<div class="alerta alerta-aviso aparecer">${icono("triangle")}<div class="alerta-titulo">Nada que puntuar</div>
        <div class="alerta-desc">Ninguna fila del archivo se puede puntuar: todas tienen errores de captura.</div></div>`;
      return;
    }
    const aprobadas = res.filter((r) => !r.rechazar).length, media = res.reduce((s, r) => s + r.p, 0) / n;
    let etiquetaReal = "";
    if (columnas.includes("riesgo")) {
      const conEtiqueta = res.filter((r) => r.fila.riesgo === "bueno" || r.fila.riesgo === "malo");
      const malos = conEtiqueta.filter((r) => r.fila.riesgo === "malo");
      if (malos.length) {
        const recall = malos.filter((r) => r.rechazar).length / malos.length;
        const costo = (5 * malos.filter((r) => !r.rechazar).length + conEtiqueta.filter((r) => r.fila.riesgo === "bueno" && r.rechazar).length) / conEtiqueta.length;
        etiquetaReal = `El archivo trae la etiqueta real (${conEtiqueta.length} filas): recall de 'malo' = ${recall.toFixed(3)}, costo medio 5:1 = ${costo.toFixed(3)}.`;
      }
    }
    const kpi = (titulo, valor, pie, badge = "") => `<div class="card kpi aparecer"><div class="card-header"><div class="card-desc">${titulo}</div>
      ${badge ? `<div class="card-accion">${badge}</div>` : ""}<div class="kpi-valor" style="grid-column:1">${valor}</div></div><div class="card-pie">${pie}</div></div>`;
    html += `<div style="display:grid;gap:1rem"><h2 style="font-size:1.125rem;font-weight:600;letter-spacing:-.01em">Resultados</h2>
      <div class="kpis">
        ${kpi("Solicitudes puntuadas", n, `de ${plural(filas.length, "fila", "filas")} del archivo`)}
        ${kpi("Excluidas por errores", conError.filter(Boolean).length, "errores de captura: no se puntúan", conError.some(Boolean) ? `<span class="badge badge-error">${icono("alert")}revisar</span>` : "")}
        ${kpi("Aprobadas", pct1(aprobadas / n), `${plural(aprobadas, "solicitud", "solicitudes")} bajo el umbral`, `<span class="badge badge-exito">${icono("circleCheck")}${aprobadas}</span>`)}
        ${kpi("P(malo) media", pct1(media), `umbral de decisión: ${pct1(P_UMBRAL)}`)}
      </div>${etiquetaReal ? `<p class="texto-pequeno">${esc(etiquetaReal)}</p>` : ""}</div>
      <div class="card aparecer">
        <div class="card-header"><div class="card-titulo">${icono("table")}Solicitudes puntuadas</div>
          <div class="card-desc">Probabilidad calibrada de incumplimiento y decisión con el umbral de costo mínimo</div>
          <div class="card-accion"><button class="btn btn-outline btn-sm" type="button" id="descargar">${icono("download")}Descargar resultados (CSV)</button></div></div>
        <div class="card-contenido" style="display:grid;gap:.875rem">
          <div class="tabla-barra">
            <div class="segmentado" role="group" aria-label="Filtrar por decisión">
              <button type="button" data-filtro="todas" aria-pressed="true">Todas <span class="cuenta">${n}</span></button>
              <button type="button" data-filtro="aprobar" aria-pressed="false">Aprobar <span class="cuenta">${aprobadas}</span></button>
              <button type="button" data-filtro="rechazar" aria-pressed="false">Rechazar o revisar <span class="cuenta">${n - aprobadas}</span></button>
            </div>
            <div class="buscador">${icono("search")}<input class="input" id="buscar" placeholder="Buscar por id o valor…" autocomplete="off"></div>
          </div>
          <div class="tabla-marco"><div class="tabla-scroll"><table id="tabla-res"></table></div></div>
          <div class="paginacion" id="paginacion"></div>
        </div></div>
      <div class="card aparecer">
        <div class="card-header"><div class="card-titulo">${icono("chart")}Explicar una solicitud del lote</div>
          <div class="card-desc">Elija una fila por su posición en la tabla de resultados (columna #) o use el botón de cada fila.</div></div>
        <div class="card-contenido"><div class="explicar-fila"><div class="campo" id="campo-fila"></div>
          <button class="btn btn-primario" type="button" id="btn-explicar">${icono("sparkles")}Explicar</button></div></div>
      </div>
      <div id="resultado-lote"></div>`;
    salida.innerHTML = html;

    lote.filtro = "todas"; lote.busqueda = ""; lote.orden = null;
    salida.querySelectorAll("[data-filtro]").forEach((b) => b.addEventListener("click", () => {
      lote.filtro = b.dataset.filtro; lote.pagina = 0;
      salida.querySelectorAll("[data-filtro]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      pintarTabla();
    }));
    $("#buscar").addEventListener("input", (e) => { lote.busqueda = e.target.value.trim().toLowerCase(); lote.pagina = 0; pintarTabla(); });
    $("#descargar").addEventListener("click", descargar);
    const numeroFila = crearNumero({ id: "fila-explicar", valor: 1, min: 1, max: n, paso: 1 });
    lote.numeroFila = numeroFila;
    $("#campo-fila").innerHTML = `<label class="etiqueta" for="fila-explicar">Fila (posición en la tabla de resultados)</label>`;
    $("#campo-fila").appendChild(numeroFila.el);
    $("#btn-explicar").addEventListener("click", () => {
      const pos = numeroFila.get();
      if (Number.isNaN(pos)) { toast("error", "Indique una fila", `Entre 1 y ${n}.`); return; }
      explicarFila(pos);
    });
    pintarTabla();
  }

  function explicarFila(pos) {
    const r = lote.res[pos - 1];
    if (!r) return;
    lote.numeroFila.set(pos);
    const fila = Object.fromEntries(COLUMNAS.map((c) => [c, r.fila[c]]));
    renderizarResultado($("#resultado-lote"), fila, lote.motor);
  }

  const COLUMNAS_TABLA = [
    ["edad", "Edad", "num"], ["genero", "Género"], ["tipo_ocupacion", "Ocupación"], ["tipo_vivienda", "Vivienda"],
    ["nivel_ahorro", "Ahorro"], ["saldo_categoria", "Saldo (cat.)"], ["saldo_cuenta_cop", "Saldo (COP)", "num"],
    ["monto_credito_cop", "Monto (COP)", "num"], ["plazo_meses", "Plazo", "num"], ["proposito_credito", "Propósito"],
    ["estado_pago_previo", "Historial"],
  ];

  function pintarTabla() {
    const tieneId = lote.columnas.includes("id"), tieneRiesgo = lote.columnas.includes("riesgo");
    let filas = lote.res.filter((r) => lote.filtro === "todas" || (lote.filtro === "aprobar" ? !r.rechazar : r.rechazar));
    if (lote.busqueda) filas = filas.filter((r) => Object.values(r.fila).some((v) => v !== null && String(v).toLowerCase().includes(lote.busqueda)) || String(r.pos) === lote.busqueda);
    if (lote.orden) filas = filas.slice().sort((a, b) => (lote.orden === "asc" ? a.p - b.p : b.p - a.p));
    const total = filas.length, paginas = Math.max(1, Math.ceil(total / lote.porPagina));
    lote.pagina = Math.min(lote.pagina, paginas - 1);
    const vista = filas.slice(lote.pagina * lote.porPagina, (lote.pagina + 1) * lote.porPagina);
    const iconoOrden = lote.orden === "asc" ? "arrowUp" : lote.orden === "desc" ? "arrowDown" : "arrowUpDown";
    $("#tabla-res").innerHTML = `<thead><tr><th class="num">#</th>${tieneId ? "<th>id</th>" : ""}
        ${COLUMNAS_TABLA.map(([, t, n]) => `<th${n ? ' class="num"' : ""}>${t}</th>`).join("")}${tieneRiesgo ? "<th>Riesgo real</th>" : ""}
        <th><button class="th-orden" type="button" id="ordenar">P(malo)${icono(iconoOrden)}</button></th><th>Decisión</th><th><span class="sr-only">Acciones</span></th></tr></thead>
      <tbody>${vista.length ? vista.map((r) => `<tr class="fila-nueva">
        <td class="num mono">${r.pos}</td>${tieneId ? `<td class="mono">${esc(r.fila.id ?? "")}</td>` : ""}
        ${COLUMNAS_TABLA.map(([c, , n]) => `<td${n ? ' class="num"' : ""}>${Motor.esNulo(r.fila[c]) ? '<span class="muted">—</span>' : esc(mostrar(c, r.fila[c]))}</td>`).join("")}
        ${tieneRiesgo ? `<td>${r.fila.riesgo ? `<span class="badge ${r.fila.riesgo === "malo" ? "badge-error" : "badge-outline"}">${esc(r.fila.riesgo)}</span>` : '<span class="muted">—</span>'}</td>` : ""}
        <td><div class="celda-prob"><div class="pista"><span class="${r.rechazar ? "sube" : "baja"}" style="width:${(r.p * 100).toFixed(1)}%"></span></div><span class="num">${r.p.toFixed(3)}</span></div></td>
        <td><span class="badge ${r.rechazar ? "badge-error" : "badge-exito"}">${r.rechazar ? "rechazar / revisar" : "aprobar"}</span></td>
        <td><button class="btn btn-ghost btn-sm" type="button" data-explicar="${r.pos}">Explicar${icono("chevronRight")}</button></td></tr>`).join("")
        : `<tr><td colspan="20" class="vacio-tabla">No hay solicitudes que coincidan con el filtro.</td></tr>`}</tbody>`;
    $("#ordenar").addEventListener("click", () => { lote.orden = lote.orden === "desc" ? "asc" : lote.orden === "asc" ? null : "desc"; pintarTabla(); });
    $("#tabla-res").querySelectorAll("[data-explicar]").forEach((b) => b.addEventListener("click", () => explicarFila(+b.dataset.explicar)));
    const desde = total ? lote.pagina * lote.porPagina + 1 : 0, hasta = Math.min(total, (lote.pagina + 1) * lote.porPagina);
    $("#paginacion").innerHTML = `<span>${desde}–${hasta} de ${plural(total, "solicitud", "solicitudes")}</span>
      <div class="controles"><span>Filas por página</span><span id="por-pagina" style="width:4.75rem"></span>
        <span class="pag-texto">Página ${lote.pagina + 1} de ${paginas}</span>
        <button class="btn btn-outline btn-icono-sm" type="button" data-ir="0" aria-label="Primera página" ${lote.pagina === 0 ? "disabled" : ""}>${icono("chevronsLeft")}</button>
        <button class="btn btn-outline btn-icono-sm" type="button" data-ir="${lote.pagina - 1}" aria-label="Página anterior" ${lote.pagina === 0 ? "disabled" : ""}>${icono("chevronLeft")}</button>
        <button class="btn btn-outline btn-icono-sm" type="button" data-ir="${lote.pagina + 1}" aria-label="Página siguiente" ${lote.pagina >= paginas - 1 ? "disabled" : ""}>${icono("chevronRight")}</button>
        <button class="btn btn-outline btn-icono-sm" type="button" data-ir="${paginas - 1}" aria-label="Última página" ${lote.pagina >= paginas - 1 ? "disabled" : ""}>${icono("chevronsRight")}</button></div>`;
    $("#paginacion").querySelectorAll("[data-ir]").forEach((b) => b.addEventListener("click", () => { lote.pagina = +b.dataset.ir; pintarTabla(); }));
    const selector = crearSelect({ id: "sel-por-pagina", opciones: [10, 25, 50].map((x) => ({ valor: x, texto: String(x) })), valor: lote.porPagina,
      alCambiar: (v) => { lote.porPagina = v; lote.pagina = 0; pintarTabla(); } });
    selector.el.style.height = "2rem";
    $("#por-pagina").appendChild(selector.el);
  }

  function descargar() {
    const columnas = [...lote.columnas, "prob_malo", "decision"];
    const filas = lote.res.map((r) => ({ ...r.fila, prob_malo: Number(r.p.toFixed(4)), decision: r.rechazar ? "rechazar / revisar" : "aprobar" }));
    const blob = new Blob(["﻿" + Motor.escribirCSV(columnas, filas)], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "resultados_riesgo.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast("exito", "Resultados descargados", `resultados_riesgo.csv · ${plural(filas.length, "fila", "filas")}`);
  }

  // ---------------------------------------------------------------------
  // Pestaña 3: acerca del modelo
  // ---------------------------------------------------------------------
  function construirAcerca() {
    const hp = META.hiperparametros, m = META.metricas_test;
    $("#panel-acerca").innerHTML = `<div class="acerca">
      <div class="card ancho"><div class="card-header"><div class="card-titulo">${icono("gauge")}El modelo en cifras</div>
        <div class="card-desc">Desempeño en la partición de prueba (1.300 clientes que el modelo no vio al entrenar)</div></div>
        <div class="card-contenido"><div class="estadisticas">
          <div><span class="n">${m["ROC-AUC"].toFixed(3)}</span><span class="t">ROC-AUC</span></div>
          <div><span class="n">${m["Recall (malo)"].toFixed(3)}</span><span class="t">Recall de "malo"</span></div>
          <div><span class="n">${m["Costo medio (5:1)"].toFixed(3)}</span><span class="t">Costo medio por cliente (5:1)</span></div>
          <div><span class="n">${Motor.miles(META.n_vectores_soporte)}</span><span class="t">Vectores de soporte</span></div>
        </div></div></div>
      <div class="card"><div class="card-header"><div class="card-titulo">${icono("book")}Qué hace</div></div>
        <div class="card-contenido prosa">
          <p>Clasifica solicitudes de crédito de consumo en riesgo <strong>bueno</strong> o <strong>malo</strong> con una máquina de vectores de
          soporte (kernel gaussiano), el método de Xu y Cheng (<em>Credit Scoring Models Enhancement Using Support Vector Machines</em>),
          adaptado a una base simulada del contexto colombiano.</p>
          <p>Esta versión web reproduce la aplicación de Streamlit sin servidor: el modelo entrenado viaja dentro de la página y cada
          predicción se calcula en su navegador, con los mismos parámetros y los mismos resultados.</p>
        </div></div>
      <div class="card"><div class="card-header"><div class="card-titulo">${icono("sparkles")}Flujo de cada predicción</div></div>
        <div class="card-contenido"><ol class="pasos-flujo">
          <li><div><strong>Verificación</strong><span>De los datos frente al perfil de entrenamiento (esquema, categorías, rangos, reglas).</span></div></li>
          <li><div><strong>Pipeline</strong><span>Ajustado solo con entrenamiento: imputación del saldo por categoría y del resto por mediana o moda, escalamiento, codificación one-hot.</span></div></li>
          <li><div><strong>SVM RBF</strong><span><span class="mono">C = ${hp.C}</span>, <span class="mono">γ = ${hp.gamma}</span>, pesos de clase balanceados: puntaje de riesgo.</span></div></li>
          <li><div><strong>Calibración de Platt</strong><span>Convierte el puntaje en probabilidad sin cambiar el orden de los clientes.</span></div></li>
          <li><div><strong>Decisión</strong><span>Con el umbral que minimiza el costo 5:1 (elegido por validación cruzada en entrenamiento).</span></div></li>
          <li><div><strong>Explicación SHAP</strong><span>A nivel de las 11 variables del formulario; aquí se calcula de forma exacta.</span></div></li>
        </ol></div></div>
      <div class="card ancho"><div class="card-header"><div class="card-titulo">${icono("triangle")}Límites de uso</div></div>
        <div class="card-contenido"><ul class="limites">
          <li>${icono("triangle")}<span>Entrenado con <strong>datos simulados</strong>: mide qué tan bien se recupera una estructura de riesgo diseñada, no el comportamiento de clientes reales. No debe usarse para decisiones reales.</span></li>
          <li>${icono("triangle")}<span>El umbral depende de la matriz de costos 5:1; con otra relación de costos cambia la política.</span></li>
          <li>${icono("triangle")}<span>Con 22.5% de malos, rechaza a cerca de la mitad de los buenos clientes: es la consecuencia de la matriz de costos, y tiene implicaciones de inclusión financiera (sobre todo para no bancarizados y personas con mora previa) que una entidad debe decidir explícitamente.</span></li>
          <li>${icono("triangle")}<span>El género está en los datos, pero el modelo no lo usa (contribución SHAP ≈ 0). En un despliegue real convendría retirarlo y monitorear la equidad por grupo.</span></li>
        </ul></div></div>
    </div>`;
  }

  // ---------------------------------------------------------------------
  // Inicio
  // ---------------------------------------------------------------------
  colocarIconos();
  pintarBarra();
  construirSolicitud();
  construirLote();
  construirAcerca();
  document.documentElement.dataset.listo = "true";
})();
