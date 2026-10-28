/* Shared loader and interaction policy for the Vega-Lite report charts (from the author's temporal-lm project). */
(() => {
  "use strict";

  const mounted = new WeakMap();
  const generations = new WeakMap();
  let tooltipHideTimer = null;
  let tooltipOwner = null;
  const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';  // report.py's body font
  const reportConfig = {
    background: "#fffbfd",  // the cards' --surface (report.py)
    font: FONT,
    view: {continuousWidth: 300, continuousHeight: 300, stroke: null},
    // as the blog post's LineChart draws its axes and legend (brunborg-com's src/lib/charts/LineChart.svelte)
    axis: {
      domainColor: "#d8c4ce", domainWidth: 1, gridColor: "#f0e5eb", gridOpacity: 1,
      labelColor: "#75666e", labelFont: FONT, labelFontSize: 14, labelPadding: 8,
      tickColor: "#d8c4ce", tickSize: 6, titleColor: "#75666e", titleFont: FONT,
      titleFontSize: 15, titleFontWeight: "normal", titlePadding: 14,
    },
    axisY: {ticks: false},
    legend: {
      columns: 4, direction: "horizontal", labelColor: "#4f4249", labelFont: FONT,
      labelFontSize: 15, orient: "top", symbolStrokeWidth: 3, symbolSize: 240, columnPadding: 24, title: null,
    },
  };
  const style = document.createElement("style");
  style.textContent = `
    .altair-chart { width: 100%; min-height: 240px; }
    .altair-chart .vega-embed, .altair-chart .vega-embed > div { width: 100%; }
    .altair-chart .vega-actions { display: none; }
    #floating-chart-tip { position: fixed; inset: 0 auto auto 0; z-index: 100; min-width: 210px; max-width: 330px; padding: 8px 10px; border: 1px solid #383835; border-radius: 6px; background: #1a1a19; box-shadow: 0 2px 8px rgba(0,0,0,.5); pointer-events: none; visibility: hidden; font: 12px/1.55 "Helvetica Neue", Helvetica, Arial, sans-serif; font-variant-numeric: tabular-nums; }
    #floating-chart-tip .tip-title { color: #fff; font-weight: 700; padding: 1px 3px 6px; }
    #floating-chart-tip .tip-row { display: grid; grid-template-columns: 14px minmax(0,1fr) auto; gap: 6px; align-items: center; min-height: 19px; padding: 1px 3px; border-radius: 3px; }
    #floating-chart-tip .tip-row.nearest { background: #383835; }
    #floating-chart-tip .tip-row.nearest .tip-name { color: #fff; font-weight: 700; }
    #floating-chart-tip .tip-key { width: 12px; height: 2px; }
    #floating-chart-tip .tip-name { color: #898781; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #floating-chart-tip .tip-value { color: #fff; font-weight: 700; }
    #floating-chart-tip.card-tip { max-width: 400px; }
    #floating-chart-tip .tip-text { color: #c3c2b7; padding: 0 3px 6px; white-space: normal; }
    #floating-chart-tip .tip-text:last-child { color: #fff; font-weight: 650; padding-bottom: 2px; }
  `;
  document.head.append(style);

  const tip = document.createElement("div");
  tip.id = "floating-chart-tip";
  tip.setAttribute("role", "status");
  tip.setAttribute("aria-live", "polite");
  document.body.append(tip);

  function moveTip(event) {
    const rect = tip.getBoundingClientRect();
    let x = event.clientX + 14;
    let y = event.clientY - rect.height / 2;
    if (x + rect.width + 8 > innerWidth) x = event.clientX - rect.width - 14;
    y = Math.max(6, Math.min(innerHeight - rect.height - 6, y));
    tip.style.transform = `translate(${x}px, ${y}px)`;
  }

  function formatLearningRate(value) {
    return value.toExponential(3).replace(/\.0+(?=e)/, "").replace(/(\.\d*?)0+(?=e)/, "$1").replace("e-0", "e-").replace("e+0", "e+");
  }

  function formatX(value, meta) {
    if (meta.xLabels) return meta.xLabels[value] ?? `${value}`;
    if (meta.xKind === "tokens") return `${Math.round(value)}M tokens`;
    if (meta.xKind === "buffer") return `buffer ${Math.round(value)}`;
    if (meta.xKind === "revisions") return `${value}k revisions`;
    if (meta.xKind === "articles") return `${value.toFixed(1)}k articles`;
    if (meta.xKind === "timestamp-rank") return `${value.toFixed(1)}k timestamp-ranked articles`;
    if (meta.xKind === "learning-rate") return `learning rate ${formatLearningRate(value)}`;
    if (meta.xKind === "year") return `end ${value}`;
    // decimal year, e.g. 2010.0833 -> "Feb 2010"; rounding guards against float drift in the 1/12 steps
    if (meta.xKind === "month") {
      const months = Math.round(value * 12);
      const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      return `${names[((months % 12) + 12) % 12]} ${Math.floor(months / 12)}`;
    }
    const suffix = meta.xSuffix ? ` ${meta.xSuffix}` : "";
    return `${meta.xPrefix ?? ""}${value.toLocaleString(undefined, {maximumFractionDigits: 3})}${suffix}`;
  }

  function seriesOf(path, meta, index) {
    // Vega binds the scenegraph item to each SVG element; a line path's item carries the
    // first datum of its series. Fall back to draw order when that binding is missing.
    const datum = path.__data__?.datum;
    const field = meta.seriesField ?? "series";
    return datum && field in datum ? datum[field] : meta.series[index];
  }

  function setLineHighlight(element, meta, nearest) {
    element.querySelectorAll("svg g.mark-line path").forEach((path, index) => {
      path.style.strokeWidth = seriesOf(path, meta, index) === nearest ? "2.6px" : "";
    });
  }

  function sourceRow(item) {
    // Vega-Lite's nearest hit area is a voronoi path whose datum is the rule item it was built
    // from; that rule's datum is the pivoted data row with the raw numbers for every series.
    const source = item?.isVoronoi ? item.datum : item;
    return source?.datum ?? {};
  }

  function parseNumber(text) {
    // Formatted tooltip strings use a Unicode minus sign and thousands separators.
    return Number(String(text ?? "").replace(/\u2212/g, "-").replace(/,/g, ""));
  }

  function nearestSeries(element, rows, event, view) {
    // Compare the cursor against each series' plotted y at the shared x, in plot pixels.
    const yScale = view?.scale("y");
    const frame = element.querySelector("svg g.role-frame.root path.background");
    if (!yScale || !frame) return null;
    const rect = frame.getBoundingClientRect();
    const localY = (event.clientY - rect.top) * view.height() / Math.max(1, rect.height);
    return rows.reduce((best, row) => Math.abs(yScale(row.value) - localY) < Math.abs(yScale(best.value) - localY) ? row : best).name;
  }

  function renderTip(title, rows, nearest, decimals, valueSuffix) {
    const heading = document.createElement("div");
    heading.className = "tip-title";
    heading.textContent = title;
    tip.replaceChildren(heading);
    for (const row of rows) {
      const container = document.createElement("div");
      container.className = `tip-row${row.name === nearest ? " nearest" : ""}`;
      const key = document.createElement("i");
      key.className = "tip-key";
      key.style.background = row.color;
      const name = document.createElement("span");
      name.className = "tip-name";
      name.textContent = row.name;
      const value = document.createElement("span");
      value.className = "tip-value";
      value.textContent = `${row.value.toFixed(decimals)}${valueSuffix}`;
      container.append(key, name, value);
      tip.append(container);
    }
  }

  function lineTooltip(element, meta, getView) {
    return (_handler, event, item, value) => {
      clearTimeout(tooltipHideTimer);
      if (!value) {
        tooltipHideTimer = setTimeout(() => {
          if (tooltipOwner !== element) return;
          tip.style.visibility = "hidden";
          setLineHighlight(element, meta, null);
          tooltipOwner = null;
        }, 80);
        return;
      }
      tooltipOwner = element;
      const row = sourceRow(item);
      const number = (field, formatted) => (Number.isFinite(row[field]) ? row[field] : parseNumber(formatted));
      const x = number(meta.xField ?? "x", value.__x);
      const rows = meta.series
        .filter((name) => value[name] !== undefined && value[name] !== null)
        .map((name) => ({name, value: number(name, value[name]), color: meta.colors[name]}))
        .filter((entry) => Number.isFinite(entry.value))
        .sort((a, b) => b.value - a.value);
      if (!rows.length) return;
      const nearest = nearestSeries(element, rows, event, getView());
      setLineHighlight(element, meta, nearest);
      renderTip(formatX(x, meta), rows, nearest, Number(meta.decimals ?? 4), meta.valueSuffix ?? "");
      tip.style.visibility = "visible";
      moveTip(event);
    };
  }

  function barTooltip(element) {
    // A bar's tooltip fields tip0, tip1, ... are its card: a title, then paragraphs (report.py's bar_chart).
    return (_handler, event, _item, value) => {
      clearTimeout(tooltipHideTimer);
      if (!value) {
        tooltipHideTimer = setTimeout(() => {
          if (tooltipOwner !== element) return;
          tip.style.visibility = "hidden";
          tip.classList.remove("card-tip");
          tooltipOwner = null;
        }, 80);
        return;
      }
      tooltipOwner = element;
      const [title, ...lines] = Object.keys(value).filter((key) => /^tip\d+$/.test(key)).sort((a, b) => a.slice(3) - b.slice(3)).map((key) => value[key]);
      const heading = Object.assign(document.createElement("div"), {className: "tip-title", textContent: title});
      tip.replaceChildren(heading, ...lines.filter(Boolean).map((line) => Object.assign(document.createElement("div"), {className: "tip-text", textContent: line})));
      tip.classList.add("card-tip");
      tip.style.visibility = "visible";
      moveTip(event);
    };
  }

  // An svg_card's elements with data-tip-title (report.py's svg_tip) show the same card tooltip as bar_chart's bars:
  // the title, then a paragraph per line of data-tip.
  let svgTipOwner = null;
  document.addEventListener("pointermove", (event) => {
    const target = event.target.closest?.(".svg-figure [data-tip-title]");
    if (!target) {
      if (svgTipOwner) {
        tip.style.visibility = "hidden";
        tip.classList.remove("card-tip");
        svgTipOwner = null;
      }
      return;
    }
    if (svgTipOwner !== target) {
      const heading = Object.assign(document.createElement("div"), {className: "tip-title", textContent: target.dataset.tipTitle});
      const lines = (target.dataset.tip ?? "").split("\n").filter(Boolean);
      tip.replaceChildren(heading, ...lines.map((line) => Object.assign(document.createElement("div"), {className: "tip-text", textContent: line})));
      tip.classList.add("card-tip");
      svgTipOwner = target;
    }
    tip.style.visibility = "visible";
    moveTip(event);
  });

  // The legends' line symbols with the lines' round caps (Vega-Lite has no legend option for it), so that a dashed or
  // dotted line's symbol is dashed or dotted as the line is: square-capped, a dot is a short dash.
  function roundLegendStrokes(spec) {
    for (const legend of spec.legends ?? []) {
      const symbols = (legend.encode ??= {}).symbols ??= {};
      (symbols.update ??= {}).strokeCap = {value: "round"};
    }
    return spec;
  }

  async function load(element, source = element.dataset.spec) {
    if (!source) return;
    const generation = (generations.get(element) ?? 0) + 1;
    generations.set(element, generation);
    element.classList.add("is-loading");
    try {
      const response = await fetch(source);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText} while loading ${source}`);
      const spec = await response.json();
      if (generations.get(element) !== generation) return;
      mounted.get(element)?.finalize();
      let view = null;
      const meta = spec.usermeta?.report ?? {};
      const options = {renderer: "svg", actions: false, config: reportConfig, patch: roundLegendStrokes};
      options.tooltip = meta.kind === "line" ? lineTooltip(element, meta, () => view) : meta.kind === "bar-card" ? barTooltip(element) : {theme: "dark"};
      const result = await vegaEmbed(element, spec, options);
      if (generations.get(element) !== generation) {
        result.finalize();
        return;
      }
      view = result.view;
      mounted.set(element, result);
      element.dataset.spec = source;
      element.dispatchEvent(new CustomEvent("report-chart-loaded", {detail: {source}}));
    } catch (error) {
      if (generations.get(element) !== generation) return;
      const message = document.createElement("pre");
      message.className = "chart-error";
      message.textContent = error instanceof Error ? error.stack : String(error);
      element.replaceChildren(message);
    } finally {
      if (generations.get(element) === generation) element.classList.remove("is-loading");
    }
  }

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      load(entry.target);
    }
  }, {rootMargin: "1000px 0px"});

  document.querySelectorAll(".altair-chart[data-spec]").forEach((element) => observer.observe(element));
  addEventListener("beforeprint", () => document.querySelectorAll(".altair-chart[data-spec]").forEach((element) => load(element)));
  // A card's view buttons (report.py's figure_card) swap its chart to another spec. The chart's data-spec changes
  // first, so a chart not yet scrolled into view loads the chosen spec when it is.
  document.addEventListener("click", (event) => {
    const button = event.target.closest(".view-toggle button[data-spec]");
    if (!button || button.getAttribute("aria-pressed") === "true") return;
    const card = button.closest(".chart-card");
    const chart = card.querySelector(".altair-chart");
    for (const other of card.querySelectorAll(".view-toggle button")) other.setAttribute("aria-pressed", String(other === button));
    chart.dataset.spec = button.dataset.spec;
    if (mounted.has(chart) || chart.classList.contains("is-loading")) load(chart);
  });
  window.ReportPlots = {load};
})();
