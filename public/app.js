(() => {
  "use strict";
  const M = window.MODEL;
  const P = M.params;
  const META = M.meta;
  const K = META.K;

  /* ---------------- language ---------------- */
  let LANG = "en";
  try {
    const saved = localStorage.getItem("lang");
    if (saved === "ta" || saved === "en") LANG = saved;
  } catch (_) { /* storage blocked */ }
  const T = (k) => window.I18N[LANG][k];

  const SAMPLES = [
    { hyp: 0, heart: 0, work: 0, glu: 88, bmi: 23.5, smoke: 0 },
    { hyp: 1, heart: 0, work: 0, glu: 150, bmi: 31, smoke: 2 },
    { hyp: 1, heart: 1, work: 1, glu: 228.7, bmi: 36.6, smoke: 1 },
  ];

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmt = (x, d = 1) => Number(x).toFixed(d);
  const workLabels = () => [0, 1, 2, 3, 4].map((i) => T("work" + i));
  const smokeLabels = () => [0, 1, 2, 3].map((i) => T("smokeFull" + i));
  const bandKey = { Low: "Low", Moderate: "Mod", High: "High" };
  const bandLabel = (n) => T("band" + bandKey[n]);
  const bandWord = (n) => T("bandWord" + bandKey[n]);

  /* ---------------- model: attention-based neural network ---------------- */
  function forward(inp) {
    const g = clamp((inp.glu - META.G_MIN) / (META.G_MAX - META.G_MIN), 0, 1);
    const b = clamp((inp.bmi - META.B_MIN) / (META.B_MAX - META.B_MIN), 0, 1);
    const lin = (s, U, C) => U.map((u, k) => s * u + C[k]);
    // per-factor representations (6 x K)
    const V = [
      P.E_hyp[inp.hyp],
      P.E_heart[inp.heart],
      P.E_work[inp.work],
      lin(g, P.U_glu, P.C_glu),
      lin(b, P.U_bmi, P.C_bmi),
      P.E_smoke[inp.smoke],
    ];
    // attention scores e_i = q . tanh(Wa^T v_i + ba)
    const e = V.map((v) => {
      let s = 0;
      for (let j = 0; j < K; j++) {
        let u = P.ba[j];
        for (let i = 0; i < K; i++) u += v[i] * P.Wa[i][j];
        s += Math.tanh(u) * P.q[j];
      }
      return s;
    });
    const mx = Math.max(...e);
    const ex = e.map((x) => Math.exp(x - mx));
    const sum = ex.reduce((a, c) => a + c, 0);
    const alpha = ex.map((x) => x / sum);
    // weighted features Z (6K)
    const Z = [];
    V.forEach((v, gi) => v.forEach((x) => Z.push(6 * alpha[gi] * x)));
    // hidden ReLU layer
    const H = P.b1.length;
    const h = new Array(H).fill(0).map((_, j) => {
      let s = P.b1[j];
      for (let i = 0; i < Z.length; i++) s += Z[i] * P.W1[i][j];
      return Math.max(0, s);
    });
    const logit = h.reduce((a, x, j) => a + x * P.w2[j], P.b2[0]);
    const y = 1 / (1 + Math.exp(-logit));
    return { y, alpha, h, logit, g, b };
  }
  window.__forward = forward; // exposed for testing

  /* ---------------- state + form ---------------- */
  const state = { hyp: 0, heart: 0, work: 0, smoke: 0, glu: 95, bmi: 24, bmiUnknown: false };

  function setGroup(field, v) {
    state[field] = v;
    $$(`[data-field="${field}"] button`).forEach((b) => {
      const on = Number(b.dataset.v) === v;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on);
    });
  }
  $$("[data-field]").forEach((grp) => {
    grp.addEventListener("click", (ev) => {
      const b = ev.target.closest("button");
      if (b) setGroup(grp.dataset.field, Number(b.dataset.v));
    });
  });

  const gluNum = $("#gluNum"), gluRange = $("#gluRange"), gluTag = $("#gluTag");
  const bmiNum = $("#bmiNum"), bmiRange = $("#bmiRange"), bmiTag = $("#bmiTag");
  const bmiUnknown = $("#bmiUnknown");

  function paintRange(r) {
    const p = ((r.value - r.min) / (r.max - r.min)) * 100;
    r.style.setProperty("--p", p + "%");
  }
  function tag(el, text, cls) { el.textContent = text; el.className = "tag" + (cls ? " " + cls : ""); }

  function setGlu(v, from) {
    state.glu = v;
    if (from !== "num") gluNum.value = fmt(v, v % 1 ? 1 : 0);
    if (from !== "range") gluRange.value = clamp(v, 55, 272);
    paintRange(gluRange);
    if (!isFinite(v)) return tag(gluTag, "—", "info");
    if (v < 140) tag(gluTag, T("tagTypical"));
    else if (v < 200) tag(gluTag, T("tagElevated"), "mod");
    else tag(gluTag, T("tagHigh"), "high");
  }
  function setBmi(v, from) {
    state.bmi = v;
    if (from !== "num") bmiNum.value = fmt(v, 1);
    if (from !== "range") bmiRange.value = clamp(v, 12, 60);
    paintRange(bmiRange);
    if (state.bmiUnknown) return tag(bmiTag, T("tagImputed"), "info");
    if (!isFinite(v)) return tag(bmiTag, "—", "info");
    if (v < 18.5) tag(bmiTag, T("tagUnderweight"), "info");
    else if (v < 25) tag(bmiTag, T("tagHealthy"));
    else if (v < 30) tag(bmiTag, T("tagOverweight"), "mod");
    else tag(bmiTag, T("tagObese"), "high");
  }
  gluNum.addEventListener("input", () => setGlu(parseFloat(gluNum.value), "num"));
  gluRange.addEventListener("input", () => setGlu(parseFloat(gluRange.value), "range"));
  bmiNum.addEventListener("input", () => setBmi(parseFloat(bmiNum.value), "num"));
  bmiRange.addEventListener("input", () => setBmi(parseFloat(bmiRange.value), "range"));

  let bmiBeforeImpute = 24;
  bmiUnknown.addEventListener("change", () => {
    state.bmiUnknown = bmiUnknown.checked;
    bmiNum.disabled = bmiRange.disabled = bmiUnknown.checked;
    if (bmiUnknown.checked) { bmiBeforeImpute = state.bmi; setBmi(META.BMI_MEAN); }
    else setBmi(isFinite(bmiBeforeImpute) ? bmiBeforeImpute : 24);
  });

  const calcToggle = $("#bmiCalcToggle"), calc = $("#bmiCalc");
  calcToggle.addEventListener("click", () => {
    calc.hidden = !calc.hidden;
    calcToggle.setAttribute("aria-expanded", String(!calc.hidden));
  });
  const hCm = $("#hCm"), wKg = $("#wKg");
  function fromHW() {
    const h = parseFloat(hCm.value) / 100, w = parseFloat(wKg.value);
    if (h > 0.5 && h < 2.5 && w > 10 && w < 300) {
      if (bmiUnknown.checked) { bmiUnknown.checked = false; bmiUnknown.dispatchEvent(new Event("change")); }
      setBmi(Math.round((w / (h * h)) * 10) / 10);
    }
  }
  hCm.addEventListener("input", fromHW);
  wKg.addEventListener("input", fromHW);

  function loadSample(i) {
    const s = SAMPLES[i];
    ["hyp", "heart", "work", "smoke"].forEach((f) => setGroup(f, s[f]));
    if (bmiUnknown.checked) { bmiUnknown.checked = false; bmiUnknown.dispatchEvent(new Event("change")); }
    setGlu(s.glu); setBmi(s.bmi);
    toast(T("toastSample"));
  }
  $$(".sample").forEach((b) => b.addEventListener("click", () => loadSample(Number(b.dataset.sample))));

  /* ---------------- navigation ---------------- */
  function show(id) {
    $("#toast").classList.remove("show");
    $$(".screen").forEach((s) => s.classList.toggle("active", s.id === "screen-" + id));
    const tabId = id === "result" ? "assess" : id;
    $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.go === tabId));
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => show(t.dataset.go)));
  $("#backBtn").addEventListener("click", () => show("assess"));

  /* ---------------- predict + render ---------------- */
  function bandFor(y) {
    return M.bands.find((b) => y >= b.lo && y < b.hi) || M.bands[M.bands.length - 1];
  }
  const bandCls = { Low: "low", Moderate: "mod", High: "high" };

  function valueText(i, inp) {
    switch (i) {
      case 0: return inp.hyp ? T("yes") : T("no");
      case 1: return inp.heart ? T("yes") : T("no");
      case 2: return workLabels()[inp.work];
      case 3: return fmt(inp.glu, inp.glu % 1 ? 1 : 0) + " mg/dL";
      case 4: return fmt(inp.bmi, 1) + (inp.bmiUnknown ? T("imputed") : "");
      case 5: return smokeLabels()[inp.smoke];
    }
  }

  let last = null;
  $("#form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    if (!isFinite(state.glu)) { toast(T("toastGlu")); gluNum.focus(); return; }
    if (!isFinite(state.bmi)) { toast(T("toastBmi")); bmiNum.focus(); return; }
    const inp = {
      ...state,
      glu: clamp(state.glu, META.G_MIN, META.G_MAX),
      bmi: clamp(state.bmi, META.B_MIN, META.B_MAX),
    };
    const out = forward(inp);
    last = { inp, out };
    render(inp, out, true);
    show("result");
  });

  function drawTicks() {
    const g = $("#gTicks");
    g.innerHTML = "";
    M.bands.slice(1).forEach((b) => {
      const th = ((135 + b.lo * 270) * Math.PI) / 180;
      const x1 = 120 + 91 * Math.cos(th), y1 = 120 + 91 * Math.sin(th);
      const x2 = 120 + 109 * Math.cos(th), y2 = 120 + 109 * Math.sin(th);
      const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      g.appendChild(l);
    });
  }
  drawTicks();

  function render(inp, out, animate) {
    const y = out.y;
    const band = bandFor(y);
    const score = Math.round(y * 100);
    const FACTORS = T("factors");

    // gauge
    const fill = $("#gFill");
    if (animate) {
      fill.style.transition = "none";
      fill.style.strokeDasharray = "0 100";
      fill.getBoundingClientRect();
      fill.style.transition = "";
      requestAnimationFrame(() => requestAnimationFrame(() => {
        fill.style.strokeDasharray = `${(100 * y).toFixed(2)} 100`;
      }));
    } else {
      fill.style.strokeDasharray = `${(100 * y).toFixed(2)} 100`;
    }
    const scoreEl = $("#score");
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !animate) scoreEl.textContent = score;
    else {
      const t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / 1100);
        scoreEl.textContent = Math.round(score * (1 - Math.pow(1 - k, 3)));
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
    const bEl = $("#band");
    bEl.textContent = bandLabel(band.name);
    bEl.className = "band " + bandCls[band.name];

    const rate = (band.rate * 100).toFixed(1), base = (M.base_rate * 100).toFixed(1);
    const ratio = band.rate / M.base_rate;
    let cmp;
    if (ratio >= 1.1) cmp = T("cmpAbove")(ratio.toFixed(1), base);
    else if (ratio <= 0.9) cmp = T("cmpBelow")(base);
    else cmp = T("cmpNear")(base);
    $("#bandCopy").innerHTML = T("bandCopy")(bandWord(band.name), rate, cmp);

    // attention
    const order = out.alpha.map((a, i) => [a, i]).sort((p, q) => q[0] - p[0]);
    const list = $("#attList");
    list.className = "att " + bandCls[band.name];
    list.innerHTML = order.map(([a, i], r) => `
      <li class="${r === 0 ? "top" : ""}">
        <span><span class="nm">${FACTORS[i]}</span><span class="val">${valueText(i, inp)}</span></span>
        <span class="pct">${(a * 100).toFixed(1)}%</span>
        <span class="bar"><i data-w="${(a / order[0][0]) * 100}"></i></span>
      </li>`).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$("#attList .bar i").forEach((el) => (el.style.width = el.dataset.w + "%"));
    }));

    // trace
    const active = out.h.filter((x) => x > 0).length;
    const thr = M.metrics.threshold;
    $("#trace").innerHTML = `
      <div class="tstep"><h4>${T("tStep2")}</h4>
        <table>
          <tr><td>${T("tGlu")(fmt(inp.glu, 1))}</td><td>${fmt(out.g, 4)}</td></tr>
          <tr><td>${T("tBmi")(fmt(inp.bmi, 1), inp.bmiUnknown ? T("meanImputed") : "")}</td><td>${fmt(out.b, 4)}</td></tr>
          <tr><td>${T("fHyp")}</td><td>${inp.hyp}</td></tr>
          <tr><td>${T("fHeart")}</td><td>${inp.heart}</td></tr>
          <tr><td>${T("fWork")}</td><td>${workLabels()[inp.work]}</td></tr>
          <tr><td>${T("fSmoke")}</td><td>${smokeLabels()[inp.smoke]}</td></tr>
        </table>
      </div>
      <div class="tstep"><h4>${T("tStep4")}</h4>
        <table>${out.alpha.map((a, i) => `<tr><td>α ${FACTORS[i]}</td><td>${fmt(a, 4)}</td></tr>`).join("")}</table>
      </div>
      <div class="tstep"><h4>${T("tStep5")}</h4>
        <div class="neurons">${out.h.map((x) => `<i class="${x > 0 ? "on" : ""}" title="${fmt(x, 3)}"></i>`).join("")}</div>
        <p>${T("tNeurons")(active, out.h.length)}</p>
      </div>
      <div class="tstep"><h4>${T("tStep6")}</h4>
        <table>
          <tr><td>W₂·h + b₂</td><td>${fmt(out.logit, 4)}</td></tr>
          <tr><td>y = σ(·)</td><td>${fmt(y, 4)}</td></tr>
          <tr><td>${T("tThreshold")}</td><td>${fmt(thr, 3)}</td></tr>
        </table>
        <p>${T("tVerdict")(y >= thr, y >= thr ? T("clsHigh") : T("clsNot"))}</p>
      </div>`;
  }

  /* ---------------- share ---------------- */
  $("#shareBtn").addEventListener("click", async () => {
    if (!last) return;
    const { inp, out } = last;
    const band = bandLabel(bandFor(out.y).name);
    const top = out.alpha.map((a, i) => [a, i]).sort((p, q) => q[0] - p[0])[0];
    const text = T("shareText")(Math.round(out.y * 100), band, T("factors")[top[1]], valueText(top[1], inp));
    try {
      if (navigator.share) await navigator.share({ title: T("shareTitle"), text });
      else { await navigator.clipboard.writeText(text); toast(T("toastCopied")); }
    } catch (_) { /* dismissed */ }
  });

  /* ---------------- research metrics ---------------- */
  function renderMetrics() {
    const m = M.metrics, c = m.confusion;
    const pct = (x) => Math.round(x * 100) + "%";
    const nf = (x) => x.toLocaleString("en-IN");
    $("#metricsCard").innerHTML = `
      <p class="small" style="margin-top:0">${T("mIntro")(nf(m.n_total), m.n_stroke, (M.base_rate * 100).toFixed(1), nf(m.n_test))}</p>
      <div class="mgrid">
        <div class="m"><b>${fmt(m.test_auc, 2)}</b><span>${T("mAuc")}</span></div>
        <div class="m"><b>${pct(m.accuracy)}</b><span>${T("mAcc")}</span></div>
        <div class="m"><b>${pct(m.sensitivity)}</b><span>${T("mSens")}</span></div>
        <div class="m"><b>${pct(m.specificity)}</b><span>${T("mSpec")}</span></div>
      </div>
      <div class="cm">
        <div class="h"></div><div class="h">${T("cmPredStroke")}</div><div class="h">${T("cmPredNo")}</div>
        <div class="h" style="text-align:left">${T("cmActStroke")}</div><div class="g">${c.tp} TP</div><div class="b">${c.fn} FN</div>
        <div class="h" style="text-align:left">${T("cmActNo")}</div><div class="b">${c.fp} FP</div><div class="g">${c.tn} TN</div>
      </div>
      <p class="small"><b>${T("mWhyLabel")}</b>${T("mWhy")}</p>`;
  }

  /* ---------------- Tamil Nadu trend chart ---------------- */
  const TN_SERIES = [
    { year: 2021, cases: 108000 }, { year: 2022, cases: 110000 },
    { year: 2023, cases: 112000 }, { year: 2024, cases: 113500 },
    { year: 2025, cases: 114500 }, { year: 2026, cases: 115450 },
  ];
  function drawTN() {
    const host = $("#tnChart");
    if (!host) return;
    const L = 40, R = 326, TOP = 16, BOT = 162;      // plot box inside a 340 x 190 frame
    const LO = 100000, HI = 120000;                   // axis floor is not zero; every tick is labelled
    const x = (i) => L + (i * (R - L)) / (TN_SERIES.length - 1);
    const y = (v) => BOT - ((v - LO) / (HI - LO)) * (BOT - TOP);
    const inr = (n) => n.toLocaleString("en-IN");
    const ticks = [100000, 105000, 110000, 115000, 120000];

    const grid = ticks.map((t) =>
      `<line class="grid" x1="${L}" x2="${R}" y1="${y(t).toFixed(1)}" y2="${y(t).toFixed(1)}"/>` +
      `<text class="axis" x="${L - 6}" y="${(y(t) + 3).toFixed(1)}" text-anchor="end">${(t / 100000).toFixed(2)}</text>`
    ).join("");
    const path = TN_SERIES.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(d.cases).toFixed(1)}`).join(" ");
    const dots = TN_SERIES.map((d, i) =>
      `<circle class="dot" cx="${x(i).toFixed(1)}" cy="${y(d.cases).toFixed(1)}" r="4"><title>${d.year}: ${inr(d.cases)}</title></circle>`
    ).join("");
    const years = TN_SERIES.map((d, i) =>
      `<text class="axis" x="${x(i).toFixed(1)}" y="${BOT + 18}" text-anchor="middle">${d.year}</text>`
    ).join("");
    const first = TN_SERIES[0], lastPt = TN_SERIES[TN_SERIES.length - 1];
    const labels =
      `<text class="val" x="${L}" y="${(y(first.cases) - 12).toFixed(1)}" text-anchor="start">${inr(first.cases)}</text>` +
      `<text class="val" x="${R}" y="${(y(lastPt.cases) - 12).toFixed(1)}" text-anchor="end">${inr(lastPt.cases)}</text>`;
    const rows = TN_SERIES.map((d) => `<tr><td>${d.year}</td><td>${inr(d.cases)}</td></tr>`).join("");

    host.innerHTML =
      `<svg class="tn-svg" viewBox="0 0 340 190" role="img" aria-label="${T("tnChartTitle")}">` +
      `<text class="cap" x="0" y="10">${T("tnLakh")}</text>` +
      grid + years +
      `<path class="ln" d="${path}"/>` + dots + labels +
      `</svg>` +
      `<div class="sr-only"><table><caption>${T("tnChartTitle")}</caption><tbody>${rows}</tbody></table></div>`;
  }

  /* ---------------- apply a language ---------------- */
  function applyLang(next) {
    LANG = next;
    try { localStorage.setItem("lang", next); } catch (_) { /* storage blocked */ }
    document.documentElement.lang = next;
    $$("[data-t]").forEach((el) => { el.textContent = T(el.dataset.t); });
    $$("[data-th]").forEach((el) => { el.innerHTML = T(el.dataset.th); });
    $$(".lang button").forEach((b) => b.classList.toggle("on", b.dataset.lang === next));
    setGlu(state.glu); setBmi(state.bmi);
    renderMetrics();
    drawTN();
    if (last) render(last.inp, last.out, false);
  }
  $$(".lang button").forEach((b) => b.addEventListener("click", () => applyLang(b.dataset.lang)));

  /* ---------------- toast ---------------- */
  let tt;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(tt);
    tt = setTimeout(() => t.classList.remove("show"), 2200);
  }

  /* ---------------- PWA ---------------- */
  let deferred;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e;
    $("#installBtn").hidden = false;
  });
  $("#installBtn").addEventListener("click", async () => {
    if (!deferred) return;
    deferred.prompt();
    await deferred.userChoice;
    deferred = null;
    $("#installBtn").hidden = true;
  });
  if ("serviceWorker" in navigator && location.protocol !== "file:") {
    window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }

  // init
  setGlu(95); setBmi(24);
  ["hyp", "heart", "work", "smoke"].forEach((f) => setGroup(f, state[f]));
  applyLang(LANG);
})();
