/* The world record over time, animated (overview.py's records_animation_card): the records since Jan '25 drawn as a
   blue line, then the new record's pink drop, the line labelled as the chart zooms out, on a 1200×675 canvas (16:9,
   the shape Twitter shows in full) that loops.
   Each figure.animation-card carries its data as JSON: the README's records from the first one shown, the new record
   and the previous one re-timed alongside it, and the new record's date; with data-layout="phone", it is drawn laid out
   for a phone (PHONE, as the blog post's record-animation.js draws it below md): narrower and taller, its text larger
   for the smaller screen; with data-layout="tweet", as the phone's but shorter (TWEET). */
(() => {
  "use strict";

  const C = { bg: "#fffbfd", ink: "#2b2025", muted: "#75666e", grid: "#f0e5eb", blue: "#3a7be6", pink: "#f7329a" };  // plum.py's PLUM_ANIMATION
  const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';  // report.py's body font
  const LABEL = "all improvements since 2025";  // the blue line's label, centered over the plot
  const HEADER = "NanoGPT speedrun since 2025";  // in the label's place until the drop lands
  // The layout: the canvas (W×H), the plot, the top of the y axis (s), which is the label's top too, the top of the plot
  // before the drop and after the zoom (lower, the zoomed-out line from Apr to Jul '25 runs into the label), and the
  // sizes (px) and places of the text.
  const DESKTOP = {
    W: 1200, H: 675, plot: { X0: 120, X1: 905, YT: 128, YB: 600 }, top: 240,
    ticks: { size: 18, dx: 12, dy: 6, gap: 48, max: Infinity },  // the y axis's, right-aligned dx left of it
    months: { size: 18, dy: 30, gap: 84 },  // the x axis's, dy under it
    label: { x: 512.5, size: 26, cap: 19 },  // its centre (over the plot's), its cap height: its baseline is cap px under
                                             // the top of the y axis
    time: { x: 148, y: 548, size: 64 },  // the time, its baseline's left end
    brace: { x: 952, text: 984, size: 40, dy: 14 },  // the braces' left ends, and their reductions'
    note: { at: 17, dx: 18, dy: [-2, 28], size: 25 },  // the drop's label, its lines' baselines dy under at s, dx left of it
  };
  // On a phone (the blog post's): shown at about 0.55× (350 px at a 390 px viewport), so that its smallest text, the
  // ticks, is about 11.5 px. The plot is narrower, and runs up to 300 s, over the line's start (214 s), for the label
  // on one line above it, wider than the plot, centred over the drawing; no tick is drawn above 200 s, beside it.
  const PHONE = {
    W: 640, H: 470, plot: { X0: 78, X1: 420, YT: 60, YB: 390 }, top: 300,
    ticks: { size: 21, dx: 10, dy: 7, gap: 34, max: 200 },
    months: { size: 21, dy: 30, gap: 72 },
    label: { x: 318, size: 24, cap: 17.5 },
    time: { x: 92, y: 372, size: 36 },
    brace: { x: 448, text: 478, size: 34, dy: 12 },
    note: { at: 15, dx: 14, dy: [-2, 25], size: 23 },
  };
  // In a tweet (twitter.py's mock): the phone's, shorter, to sit lower in the timeline: less room above the plot, the
  // plot shorter (its ticks closer), its y axis lower until the drop lands (start), and the reductions smaller; the canvas ends fit px right of the widest reduction
  // the font draws (fitted, when it is mounted), so that no empty space is left on the right; and it holds the finished
  // chart longer before it loops.
  const TWEET = {
    ...PHONE,
    W: 620, H: 350, plot: { X0: 78, X1: 420, YT: 36, YB: 286 }, top: 360,  // up to 360 s, for room under the label
    ticks: { ...PHONE.ticks, gap: 18 },  // the most that keeps 10, 20, 50, 100 and 200 s on the finished axis
    time: { x: 92, y: 268, size: 36 },
    brace: { x: 448, text: 472, size: 24, dy: 8.5 },
    note: { ...PHONE.note, at: 24 },
    label: { ...PHONE.label, x: (78 + 420) / 2 },  // centred over the plot's grid, not the drawing
    // (s) the top of the y axis until the drop lands, lower than the finished chart's: the axis keeps its height as it
    // pans down with the drop, so the blue line scrolls nearly all out of sight (high enough for the line to start under
    // the label)
    start: 280,
    fit: 16,
    hold: 6.7,  // (s) the finished chart, before it loops: as long as the new record
    // (s) the end of the hold, in which it goes back to its start, so that it loops without a jump: the axis zooms back
    // in as the curve and its text fade out; its label is there from the start, in the finished chart's place
    loop: .3,
    // (s) of the hold it starts with, so that its still before it plays (and a recorded video's first frame, X's
    // preview) is the finished chart, 39.9 s → 6.7 s, not an empty plot; before the loop, 2 s of it
    lead: 2.3,
  };
  const PINK_AT = 3, PINK_END = .4;  // the pink drop starts at 3 s and ends at 0.4× its unpaced speed
  const ZOOM = 2.5;  // (s) the zoom out once the drop has landed
  const HOLD = 4.9;  // (s) the finished chart, before it loops (a layout's hold, if it has one)

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const seg = (t, a, b) => clamp((t - a) / (b - a));
  const lerp = (a, b, x) => a + (b - a) * x;
  const easeOut = x => 1 - Math.pow(1 - x, 3);
  const easeInOut = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  const easeInOutSine = x => (1 - Math.cos(Math.PI * x)) / 2;
  const backOut = x => { const c1 = 1.9, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };
  const pct = r => `−${(100 * r).toFixed(1)}%`;
  const day = s => { const [m, d, y] = s.split("/").map(Number); return Date.UTC(2000 + y, m - 1, d) / 864e5; };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthLabel = d => { const t = new Date(d * 864e5); return `${MONTHS[t.getUTCMonth()]} '${String(t.getUTCFullYear()).slice(2)}`; };

  // ---------- drawing helpers ----------
  function txt(c, s, x, y, o = {}) {
    const { size = 18, weight = 400, color = C.ink, align = "left", alpha = 1, halo = false } = o;
    if (alpha <= 0) return;
    c.save();
    c.globalAlpha *= alpha;
    c.font = `${weight} ${size}px ${FONT}`;
    c.textAlign = align; c.textBaseline = "alphabetic";
    if (halo) { c.lineWidth = 6; c.strokeStyle = C.bg; c.lineJoin = "round"; c.strokeText(s, x, y); }
    c.fillStyle = color; c.fillText(s, x, y);
    c.restore();
  }
  // Bold text runs on one line: [[string, color], ...]
  function runs(c, parts, x, y, size, alpha) {
    c.save(); c.font = `700 ${size}px ${FONT}`;
    const widths = parts.map(([s]) => c.measureText(s).width);
    c.restore();
    parts.forEach(([s, color], i) => { txt(c, s, x, y, { size, weight: 700, color, alpha }); x += widths[i]; });
  }
  function line(c, x1, y1, x2, y2, color, width = 1, alpha = 1) {
    if (alpha <= 0) return;
    c.save(); c.globalAlpha *= alpha; c.strokeStyle = color; c.lineWidth = width;
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke(); c.restore();
  }
  function dot(c, x, y, r, fill) {
    if (r <= 0) return;
    c.save(); c.fillStyle = fill; c.strokeStyle = C.bg; c.lineWidth = 1.5;
    c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill(); c.stroke(); c.restore();
  }
  // A curly brace beside [y1, y2], its ends left at x, its point right at x + 18.
  function brace(c, x, y1, y2, alpha) {
    if (alpha <= 0) return;
    const m = (y1 + y2) / 2, r = Math.min(10, (y2 - y1) / 4);
    c.save(); c.globalAlpha *= alpha; c.strokeStyle = C.ink; c.lineWidth = 1.6; c.lineCap = "round";
    c.beginPath(); c.moveTo(x, y1); c.quadraticCurveTo(x + 9, y1, x + 9, y1 + r); c.lineTo(x + 9, m - r);
    c.quadraticCurveTo(x + 9, m, x + 18, m); c.quadraticCurveTo(x + 9, m, x + 9, m + r); c.lineTo(x + 9, y2 - r);
    c.quadraticCurveTo(x + 9, y2, x, y2); c.stroke(); c.restore();
  }
  // Candidate ticks: every 1- and 2-significant-digit value in [lo, hi], the roundest first (p).
  function niceCandidates(lo, hi) {
    const out = new Set();
    for (let k = Math.floor(Math.log10(lo)) - 1; k <= Math.floor(Math.log10(hi)); k++)
      for (let j = 10; j < 100; j++) { const v = +(j * 10 ** k).toPrecision(6); if (v >= lo && v <= hi) out.add(v); }
    return [...out].map(v => {
      const m = +(v / 10 ** Math.floor(Math.log10(v) + 1e-9)).toPrecision(6);
      const p = Number.isInteger(m) ? (m === 1 ? 0 : m === 2 || m === 5 ? 1 : 2) : Math.round(m * 10) % 5 === 0 ? 3 : 4;
      return { v, p };
    });
  }
  // Keep a tick if it is minGap px from every rounder tick kept; it fades in over the next 22 px, so ticks come and go
  // smoothly as the axis zooms.
  function pickTicks(cands, pos, minGap) {
    cands.sort((a, b) => a.p - b.p || a.v - b.v);
    const kept = [];
    for (const cd of cands) {
      const px = pos(cd.v);
      let d = Infinity;
      for (const k of kept) d = Math.min(d, Math.abs(k.px - px));
      if (d >= minGap) kept.push({ ...cd, px, alpha: clamp((d - minGap) / 22) });
    }
    return kept;
  }

  // ---------- the animation ----------
  // data: { rows: [[record number, seconds, "MM/DD/YY"], ...] (a repeated number is a re-timing), previousRetimed,
  // newSeconds, newDate }. Returns its draw(c, t) and its duration.
  function animation({ rows, previousRetimed, newSeconds: NEW, newDate }, L = DESKTOP) {
    const REC = rows.map(([n, s, d]) => ({ n, s, d: day(d) }));
    const START = REC[0].s, PREV = REC[REC.length - 1].s, D_NEW = day(newDate);
    const R_NEW = 1 - NEW / previousRetimed;
    const DA = REC[0].d - 7, DB = D_NEW + 21;  // the x axis
    const { W, H, top: TOP } = L, { X0, X1, YT, YB } = L.plot;
    const x = d => X0 + (X1 - X0) * (d - DA) / (DB - DA);

    // The curve as vertices (day, seconds): the blue steps, then the pink drop. The head moves along its length as the
    // finished chart lays it out, a month as far as a drop of the same length on screen: on the desktop's screen, in
    // either layout, so that the phone's plays in the same time.
    const V = [{ d: DA, s: START }];
    REC.slice(1).forEach((r, k) => V.push({ d: r.d, s: REC[k].s }, { d: r.d, s: r.s, r }));
    V.push({ d: D_NEW, s: PREV }, { d: D_NEW, s: NEW, plunge: true });
    const dp = DESKTOP.plot;
    const fx = (dp.X1 - dp.X0) / (DB - DA), fy = (dp.YB - dp.YT) / Math.log(DESKTOP.top / (NEW / 1.25));
    const cum = [0], mins = [START];
    for (let j = 1; j < V.length; j++) {
      const a = V[j - 1], b = V[j];
      cum.push(cum[j - 1] + Math.abs(b.d - a.d) * fx + Math.abs(Math.log(b.s / a.s)) * fy);
      mins.push(b.plunge ? mins[j - 1] : Math.min(mins[j - 1], b.s));
    }
    // The pace: from 0.5 s the head starts fast and slows down exponentially (v0·e^(−kτ)), reaching the top of the drop
    // at PINK_AT s and its bottom at PINK_END times the speed it would have at an even pace over 9.1 s (ve). With a the
    // speed at the top of the drop, k = (a − ve)/Lp and a·(e^(kT) − 1)/k = Lb; the drop takes P s.
    const total = cum[cum.length - 1], Lb = cum[V.length - 2], Lp = total - Lb;
    const T = PINK_AT - .5, ve = PINK_END * total / 9.1;
    const blueLen = a => { const k = (a - ve) / Lp; return a * (Math.exp(k * T) - 1) / k; };
    let lo = ve * (1 + 1e-9), hi = ve + 50 * Lp;
    for (let n = 0; n < 80; n++) { const m = (lo + hi) / 2; blueLen(m) < Lb ? lo = m : hi = m; }
    const a = lo, k = (a - ve) / Lp, v0 = a * Math.exp(k * T), P = Math.log(a / ve) / k;
    const end = PINK_AT + P;  // when the drop lands
    const lenAt = t => Math.min(total, v0 * (1 - Math.exp(-k * clamp(t - .5, 0, T + P))) / k);
    // When the head reaches each vertex.
    const times = cum.map(L => {
      let lo = 0, hi = end;
      for (let n = 0; n < 40; n++) { const m = (lo + hi) / 2; lenAt(m) < L ? lo = m : hi = m; }
      return hi;
    });
    const recs = V.map((v, j) => ({ j, r: v.r })).filter(v => v.r);
    // The head on segment i (V[i - 1] to V[i]): its day and seconds, how far down the pink drop it is (pl) and the record
    // so far (min, the drop not included).
    const at = t => {
      const L = lenAt(t);
      let i = 1;
      while (i < V.length - 1 && cum[i] < L) i++;
      const a = V[i - 1], b = V[i], f = cum[i] > cum[i - 1] ? clamp((L - cum[i - 1]) / (cum[i] - cum[i - 1])) : 1;
      const s = Math.exp(lerp(Math.log(a.s), Math.log(b.s), f)), pl = b.plunge ? f : 0;
      return { i, d: lerp(a.d, b.d, f), s, pl, min: pl ? mins[i - 1] : Math.min(mins[i - 1], s) };
    };
    // The y axis: the blue records' height, down to just below the previous record; during the drop it keeps that
    // height and pans down with the pink dot, and once the drop has landed it zooms out (easing in and out over ZOOM s)
    // to show the whole chart. It is still (fixed) except while it zooms.
    const START_TOP = L.start ?? TOP;  // the y axis's top before the drop
    const span = Math.log(START_TOP * 1.12 / PREV);
    // How far a loop has gone back to its start (L.loop), 0 to 1, eased gently.
    const DUR = end + (L.hold ?? HOLD), back = t => L.loop ? easeInOutSine(seg(t, DUR - L.loop, DUR)) : 0;
    const yDomain = (h, t) => {
      const lo = Math.min(PREV, h.s) / 1.12;
      if (t < end) return { lo, hi: lo * Math.exp(span), fixed: true };
      const u = back(t);  // the loop's zoom back in, to the start's axis
      if (u > 0) return { lo: Math.exp(lerp(Math.log(NEW / 1.25), Math.log(PREV / 1.12), u)),
                          hi: Math.exp(lerp(Math.log(TOP), Math.log(START_TOP), u)), fixed: u === 1 };
      const z = easeInOut(seg(t, end, end + ZOOM));
      return { lo: Math.exp(lerp(Math.log(lo), Math.log(NEW / 1.25), z)),
               hi: Math.exp(lerp(Math.log(lo) + span, Math.log(TOP), z)), fixed: z === 0 || z === 1 };
    };

    const scale = d => v => YB - (YB - YT) * Math.log(v / d.lo) / Math.log(d.hi / d.lo);
    // The ticks a still axis shows, and while the axis zooms (out after the drop, or back in for a loop), those of the
    // axes it zooms from and to and how far it has got (0 to 1, in time). A zoom shows only those: the ones it zooms to,
    // and the others it zooms from, fading out in its first 60%, not ticks that would come and go in between, nor ones
    // lasting, half faded, until its end.
    const shown = d => new Set(pickTicks(niceCandidates(d.lo, d.hi), scale(d), L.ticks.gap)
      .filter(k => k.alpha >= 1 && k.v <= L.ticks.max).map(k => k.v));
    const zoom = (h, t) => {
      if (t < end) return null;
      const a = DUR - (L.loop ?? 0);
      if (back(t) > 0) return { from: shown(yDomain(h, a)), to: shown(yDomain(h, DUR)), p: seg(t, a, DUR) };
      if (t < end + ZOOM) return { from: shown(yDomain(h, end)), to: shown(yDomain(h, end + ZOOM)), p: seg(t, end, end + ZOOM) };
      return null;
    };
    const tickAlpha = (z, v) => !z || z.to.has(v) ? 1 : z.from.has(v) ? 1 - easeInOut(clamp(z.p / .6)) : 0;

    function draw(c, t) {
      c.fillStyle = C.bg; c.fillRect(0, 0, W, H);
      const h = at(t), dm = yDomain(h, t), z = zoom(h, t);
      let y = scale(dm);
      const axA = L.loop ? 1 : easeOut(seg(t, 0, .5));  // a loop's start is its end: the axis is there
      for (const tk of pickTicks(niceCandidates(dm.lo, dm.hi), y, L.ticks.gap)) {
        if (dm.fixed && tk.alpha < 1) continue;  // a still axis has no fades: a tick is shown in full or not at all
        if (tk.v > L.ticks.max) continue;
        const a = axA * tk.alpha * tickAlpha(z, tk.v);
        line(c, X0, tk.px, X1, tk.px, C.grid, 1, a);
        txt(c, `${+tk.v.toPrecision(3)} s`, X0 - L.ticks.dx, tk.px + L.ticks.dy, { size: L.ticks.size, color: C.muted, align: "right", alpha: a });
      }
      const months = [];
      for (let i = -1; i < 24; i++) {
        const d = Date.UTC(2025, i, 1) / 864e5, mo = (i + 12) % 12;
        if (d >= DA && d <= DB) months.push({ v: d, p: mo === 0 ? 0 : mo === 6 ? 1 : mo % 3 === 0 ? 2 : 3 });
      }
      for (const tk of pickTicks(months, x, L.months.gap)) {
        line(c, tk.px, YT - 10, tk.px, YB, C.grid, 1, axA * tk.alpha);
        txt(c, monthLabel(tk.v), tk.px, YB + L.months.dy, { size: L.months.size, color: C.muted, align: "center", alpha: axA * tk.alpha });
      }
      line(c, X0 - 4, YB, X1, YB, C.ink, 1.4, axA);

      // The header, then the label, under the curve and moving with it. The header is there from the start, its
      // baseline L.header px under the top of the axis (START_TOP; by default its cap height, L.label.cap, under it: in
      // the plot), and the axis panning down with the drop carries it up out of sight: out of the plot, or with
      // L.header, above it, off the canvas. The label, its
      // cap height under TOP, is drawn from when the drop lands, out of sight above the plot then, so it comes down into
      // view with the zoom out, clipped to the plot's height (the curve, to the plot).
      // With L.loop, there is no header: the label is there from the start, where the finished chart has it (its cap
      // height under the top of the axis, in the plot), clipped to the plot as it pans; its top is START_TOP before the
      // drop, TOP after it and dm.hi as the axis zooms back in, so that it is in the same place at the end as at the start.
      const font = { size: L.label.size, weight: 700, color: C.blue, align: "center" };
      c.save();
      if (L.loop) {
        c.beginPath(); c.rect(0, YT - 14, W, YB - YT + 14); c.clip();
        txt(c, LABEL, L.label.x, y(t < end ? START_TOP : back(t) > 0 ? dm.hi : TOP) + L.label.cap, font);
      } else if (t < end) {
        const top = L.header == null ? YT - 14 : 0;
        c.beginPath(); c.rect(0, top, W, YB - top); c.clip();
        txt(c, HEADER, L.label.x, y(START_TOP) + (L.header ?? L.label.cap), font);
      } else {
        c.beginPath(); c.rect(0, YT - 14, W, YB - YT + 14); c.clip();
        txt(c, LABEL, L.label.x, y(TOP) + L.label.cap, font);
      }
      c.restore();

      // The rest fades out where it is as a loop goes back to its start, on the finished chart's axis: only the axis
      // moves.
      if (back(t) > 0) y = scale(yDomain(h, DUR - L.loop));
      c.save();
      c.globalAlpha = L.loop ? 1 - easeInOut(seg(t, DUR - L.loop, DUR)) : 1;  // eased in and out, as the ticks fade

      // The curve, clipped to the plot as the axis pans and zooms.
      c.save();
      c.beginPath(); c.rect(X0, YT - 14, X1 - X0 + 10, YB - YT + 14); c.clip();
      c.strokeStyle = C.blue; c.lineWidth = 3; c.lineJoin = "round";
      c.beginPath(); c.moveTo(x(V[0].d), y(V[0].s));
      for (let j = 1; j < h.i; j++) c.lineTo(x(V[j].d), y(V[j].s));
      if (!h.pl) c.lineTo(x(h.d), y(h.s));
      if (t > .5) c.stroke();
      for (const { j, r } of recs) {
        const g = clamp((t - times[j]) / .15);
        if (g > 0) dot(c, x(r.d), y(r.s), 4.2 * backOut(g), C.blue);
      }
      const dotS = h.pl > 0 ? h.s : PREV;
      if (h.pl > 0) {
        line(c, x(D_NEW), y(PREV), x(D_NEW), y(dotS), C.pink, 6);
        dot(c, x(D_NEW), y(dotS), 8, C.pink);
      }
      c.restore();
      const ring = seg(t, end - .15, end + .55);
      if (ring > 0 && ring < 1) {
        c.save(); c.globalAlpha = 1 - ring; c.strokeStyle = C.pink; c.lineWidth = 3;
        c.beginPath(); c.arc(x(D_NEW), y(NEW), 8 + 46 * easeOut(ring), 0, 2 * Math.PI); c.stroke(); c.restore();
      }

      // The time, in the bottom left: the record so far, then the previous record (blue) and the falling time (pink).
      const ra = seg(t, .5, .9);
      const { time: tm, brace: br, note } = L;
      if (h.pl > 0) runs(c, [[`${PREV.toFixed(1)} s`, C.blue], [` → ${h.s.toFixed(1)} s`, C.pink]], tm.x, tm.y, tm.size, ra);
      else txt(c, `${h.s.toFixed(1)} s`, tm.x, tm.y, { size: tm.size, weight: 700, color: C.blue, alpha: ra });

      // The braces on the right: the pink one grows with the drop, counting its reduction up; the black one (all the
      // blue records, its reduction in blue) fades in as the chart zooms out. A brace the axis pushes out of the plot
      // shrinks to the part still in it, and fades as that gets short.
      const bA = easeOut(seg(t, end + 1, end + 2)), bB = h.pl > 0 ? 1 : 0;
      const yS = y(START), yP = y(PREV), y1 = y(h.min), y2 = y(dotS);
      const inPlot = (a, b) => {
        const lo = clamp(Math.min(a, b), YT - 14, YB), hi = clamp(Math.max(a, b), YT - 14, YB);
        return [lo, hi, hi - lo < Math.abs(b - a) - .5 ? clamp((hi - lo - 30) / 60) : 1];
      };
      const b1 = inPlot(yS, y1 - 3), b2 = inPlot(yP + 3, y2);
      if (b1[1] - b1[0] > 4) brace(c, br.x, b1[0], b1[1], bA * b1[2]);
      if (b2[1] - b2[0] > 4) brace(c, br.x, b2[0], b2[1], bB * b2[2]);
      txt(c, pct(1 - h.min / START), br.text, (b1[0] + b1[1]) / 2 + br.dy, { size: br.size, weight: 700, color: C.blue, alpha: bA * b1[2] });
      const r2 = R_NEW * (1 - dotS / PREV) / (1 - NEW / PREV);
      txt(c, pct(r2), br.text, (b2[0] + b2[1]) / 2 + br.dy, { size: br.size, weight: 700, color: C.pink, alpha: bB * b2[2] });
      const lA = easeOut(seg(t, end - .2, end + .4)), ly = y(note.at);
      txt(c, "n-gram hinted", x(D_NEW) - note.dx, ly + note.dy[0], { size: note.size, weight: 700, color: C.pink, align: "right", alpha: lA, halo: true });
      txt(c, "language modeling", x(D_NEW) - note.dx, ly + note.dy[1], { size: note.size, weight: 700, color: C.pink, align: "right", alpha: lA, halo: true });
      c.restore();
    }
    // Where draw(c, t) puts each record's dot (null before it appears, or while the axis has it out of the plot) and the
    // new record's drop (null before it starts), for hover targets over the canvas (the blog's player).
    function marks(t) {
      const h = at(t), dm = yDomain(h, t);
      const y = v => YB - (YB - YT) * Math.log(v / dm.lo) / Math.log(dm.hi / dm.lo);
      const inPlot = py => py >= YT - 14 && py <= YB;
      const dots = recs.map(({ j, r }) => t > times[j] && inPlot(y(r.s)) ? { x: x(r.d), y: y(r.s) } : null);
      const drop = h.pl > 0 ? { x: x(D_NEW), y1: Math.max(YT - 14, y(PREV)), y2: y(h.s) } : null;
      return { dots, drop };
    }
    return { draw, marks, dur: DUR, L };
  }

  // ---------- the player ----------
  const style = document.createElement("style");
  style.textContent = `
    .animation-card canvas { display: block; width: 100%; aspect-ratio: 16 / 9; margin: 8px 0 4px; border-radius: 6px; }
    .animation-controls { display: flex; gap: 8px; align-items: center; margin: 4px 0 2px; font-size: 12px; }
    .animation-controls button { padding: 2px 9px; border: 1px solid var(--baseline); border-radius: 5px; background: var(--surface); color: var(--ink-2); font: inherit; cursor: pointer; }
    .animation-controls button:hover { border-color: var(--accent); color: var(--ink); }
    .animation-controls input { flex: 1; min-width: 80px; accent-color: ${C.pink}; }
    .animation-controls .clock { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 4em; }
    .animation-controls .clock.played { text-align: right; }
    .animation-controls .status { color: var(--muted); }
  `;
  document.head.append(style);

  // A layout with fit: its width, up to the right end of the widest reduction (−88.8%, its digits as wide as any) in
  // the page's font, and fit px more.
  function fitted(L) {
    if (!L.fit) return L;
    const c = document.createElement("canvas").getContext("2d");
    c.font = `700 ${L.brace.size}px ${FONT}`;
    return { ...L, W: Math.ceil(L.brace.text + c.measureText(pct(.888)).width + L.fit) };
  }

  function makeCanvas(el, scale, { W, H }) {
    el.width = W * scale; el.height = H * scale;
    const c = el.getContext("2d"); c.setTransform(scale, 0, 0, scale, 0, 0); return c;
  }

  // Records one loop in real time, at 1.6× its layout (1920×1080 for the desktop's, 1024×752 for the phone's): MP4
  // where the browser can encode it (as recent Chrome can), else WebM.
  function record(A, status) {
    const types = ["video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
    const mime = types.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
    if (!mime) { status.textContent = "This browser can't record canvas video."; return; }
    const cv = document.createElement("canvas");
    cv.style.cssText = "position:fixed;left:-99999px;top:0";
    document.body.append(cv);
    const ctx = makeCanvas(cv, 1.6, A.L);
    const from = start(A);
    A.draw(ctx, from);
    const rec = new MediaRecorder(cv.captureStream(60), { mimeType: mime, videoBitsPerSecond: 16e6 });
    const chunks = [];
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type: mime.split(";")[0] }));
      a.download = `nanogpt-speedrun-improvements.${mime.includes("mp4") ? "mp4" : "webm"}`;
      a.click(); cv.remove();
      status.textContent = mime.includes("mp4") ? "Saved MP4." : "Saved WebM (for MP4: ffmpeg -i in.webm -c:v libx264 -pix_fmt yuv420p out.mp4).";
    };
    const t0 = performance.now();
    rec.start();
    status.textContent = "Recording… keep this tab visible.";
    (function step() {
      const t = (performance.now() - t0) / 1000;
      // from the layout's start: one loop, the same one, begun there (the end of its hold, the drop, the hold to it)
      A.draw(ctx, from ? (from + Math.min(t, A.dur)) % A.dur : Math.min(t, A.dur));
      if (t < A.dur + .2) requestAnimationFrame(step); else rec.stop();
    })();
  }

  // Where a loop starts: the layout's lead before its end, in the finished chart's hold; else the start.
  const start = A => A.L.lead ? A.dur - A.L.lead : 0;

  // Plays (looping) while the card is on screen, from its start (start) the first time it is; with reduced motion it
  // shows the finished chart until played.
  function mount(figure) {
    const L = fitted({ phone: PHONE, tweet: TWEET }[figure.dataset.layout] ?? DESKTOP);
    const A = animation(JSON.parse(figure.querySelector("script[type='application/json']").textContent), L);
    const canvas = figure.querySelector("canvas");
    const controls = document.createElement("div");
    controls.className = "animation-controls";
    controls.innerHTML = `<button type="button" data-a="play"></button><button type="button" data-a="replay">↺ Replay</button>
      <span class="clock played" title="played"></span><input type="range" min="0" max="${A.dur}" step="0.01" aria-label="Time">
      <span class="clock remaining" title="remaining"></span><button type="button" data-a="video">⬇ Video</button><span class="status"></span>`;
    canvas.after(controls);
    const $ = s => controls.querySelector(s);
    const playBtn = $('[data-a="play"]'), scrub = $("input");
    canvas.style.aspectRatio = `${L.W} / ${L.H}`;
    const ctx = makeCanvas(canvas, Math.min(2, Math.max(1, devicePixelRatio || 1)), L);
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let t = still ? A.dur : start(A), playing = !still, visible = false, last = null;
    const setPlaying = p => { playing = p; playBtn.textContent = p ? "❚❚ Pause" : "▶ Play"; };
    setPlaying(playing);
    playBtn.onclick = () => { if (t >= A.dur) t = 0; setPlaying(!playing); };
    $('[data-a="replay"]').onclick = () => { t = 0; setPlaying(true); };
    scrub.oninput = () => { t = +scrub.value; setPlaying(false); };
    $('[data-a="video"]').onclick = () => record(A, $(".status"));
    function frame(now) {
      if (playing && visible && last !== null) { t += Math.min(.1, (now - last) / 1000); if (t > A.dur) t = 0; }
      last = now;
      A.draw(ctx, t);
      scrub.value = t;
      $(".played").textContent = `${t.toFixed(1)} s`;
      $(".remaining").textContent = `−${(A.dur - t).toFixed(1)} s`;
      if (visible) requestAnimationFrame(frame);
    }
    new IntersectionObserver(([e]) => {
      const was = visible;
      visible = e.isIntersecting;
      if (visible && !was) { last = null; requestAnimationFrame(frame); }
    }, { threshold: .25 }).observe(canvas);
    A.draw(ctx, t);
  }

  document.querySelectorAll("figure.animation-card").forEach(mount);
})();
