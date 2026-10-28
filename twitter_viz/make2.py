"""Second round, following the second sketch: context -> CPU hint box -> transformer box with hatched hint blocks."""
import base64
import math
import json
from pathlib import Path

W, H = 1600, 900
HERE = Path(__file__).parent
FONTS = HERE / "fonts"
WIDTHS = json.loads((FONTS / "widths.json").read_text())


def face(family, file, weight):
    data = base64.b64encode((FONTS / file).read_bytes()).decode()
    return f"@font-face{{font-family:{family};font-weight:{weight};src:url(data:font/woff2;base64,{data}) format('woff2')}}"


SANS = dict(family="'Helvetica Neue', HintSans, Helvetica, Arial, sans-serif", widths="", scale=1.0,
            css=face("HintSans", "NimbusSans-Regular.woff2", 400) + face("HintSans", "NimbusSans-Bold.woff2", 700))
HAND = dict(family="HintHand, 'Caveat', cursive", widths="caveat-", scale=1.32,
            css=face("HintHand", "Caveat-500.woff2", 400) + face("HintHand", "Caveat-700.woff2", 700))

THEMES = {
    "blog": dict(font=SANS, bg="#fffbfd", panel="#ffffff", panel_stroke="#ead9e1", panel_sw=3, ink="#1c1c1c",
                 muted="#75666e", chip="#ffffff", lm_fill="#3a7be6", lm_stroke="none", hint="#fb64b6",
                 hint_bg="#ffe6f3", trick="#fb923c", trick_bg="#ffedd5", arrow="#1c1c1c", rough=False),
    "ink": dict(font=SANS, bg="#ffffff", panel="#ffffff", panel_stroke="#1f2a7a", panel_sw=2.5, ink="#1f2a7a",
                muted="#5b6390", chip="#ffffff", lm_fill="#ffffff", lm_stroke="#1f2a7a", hint="#1f2a7a",
                hint_bg="#ffffff", arrow="#1f2a7a", rough=False),
    "hand": dict(font=HAND, bg="#fbf8f1", panel="none", panel_stroke="#1f2a7a", panel_sw=2.6, ink="#1f2a7a",
                 muted="#5b6390", chip="none", lm_fill="none", lm_stroke="#1f2a7a", hint="#f0509f",
                 hint_bg="#fde3ef", trick="#e8772e", trick_bg="#fdebdc", arrow="#1f2a7a", rough=True),
    "dark": dict(font=SANS, bg="#141117", panel="#1b171f", panel_stroke="#2c2530", panel_sw=2, ink="#f4eef1",
                 muted="#a99aa2", chip="#2a2430", lm_fill="#5b8ff0", lm_stroke="none", hint="#fb64b6",
                 hint_bg="#3a1f30", arrow="#f4eef1", rough=False),
}

LM = {"hint": 0.20, "cool": 0.45, "trick": 0.25}   # the LM's own guess
VOTES = {"hint": 2, "trick": 1}                     # what came next in the training data


class Canvas:
    def __init__(self, th):
        self.th, self.f, self.out, self.solid, self.patterns = th, th["font"], [], False, set()

    def add(self, s):
        self.out.append(s)

    def tw(self, s, fs, weight="normal"):
        table = WIDTHS[self.f["widths"] + weight]
        return sum(table.get(c, 0.6) for c in s) * fs * self.f["scale"]

    def text(self, x, y, s, fs, fill=None, anchor="start", weight="normal"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs * self.f["scale"]:.1f}" fill="{fill or self.th["ink"]}" '
                 f'text-anchor="{anchor}" font-weight="{weight}" font-family="{self.f["family"]}">{s}</text>')

    def rect(self, x, y, w, h, fill, stroke="none", sw=0, rx=10, extra=""):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def chip(self, x, y, label, fs, stroke=None, weight="normal", sw=2.5, pad=0.45, h=None, fill=None):
        w = self.tw(label, fs, weight) + 2 * pad * fs
        h = h or fs * 1.75
        self.rect(x, y, w, h, fill or self.th["chip"], stroke or self.th["ink"], sw, rx=min(12, h / 4))
        self.text(x + w / 2, y + h / 2 + fs * 0.36 * self.f["scale"], label, fs, None, "middle", weight)
        return w

    def arrow(self, d, color=None, sw=3.5):
        c = color or self.th["arrow"]
        self.add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}" stroke-linecap="round" '
                 f'marker-end="url(#ah-{c[1:]})"/>')

    def hatched(self, x, y, w, h, color=None):
        color = color or self.th["hint"]
        if self.solid == "tint":   # like the hint/trick chips: light fill, thick border, no hatching
            fill = {self.th["hint"]: self.th["hint_bg"], self.th.get("trick"): self.th.get("trick_bg")}[color]
            self.rect(x + 1.5, y + 1.5, w - 3, h - 3, fill, color, 3, rx=7)
        elif self.solid:
            self.rect(x, y, w, h, color, rx=6)
        else:
            self.patterns.add(color)
            # inset by half the stroke so the outer edge lines up with unstroked bars
            self.rect(x + 1.25, y + 1.25, w - 2.5, h - 2.5, f"url(#hatch-{color[1:]})", color, 2.5, rx=5)

    def cpu(self, x, y, s, color, fg):
        """CPU chip icon, s px square incl. pins."""
        u = s / 100
        self.rect(x + 17 * u, y + 17 * u, 66 * u, 66 * u, color, rx=9 * u)
        for p in (28.5, 42.5, 57.5, 71.5):
            for px, py, pw, ph in ((p - 3, 4, 6, 15), (p - 3, 81, 6, 15), (4, p - 3, 15, 6), (81, p - 3, 15, 6)):
                self.rect(x + px * u, y + py * u, pw * u, ph * u, color, rx=3 * u)
        fs = 22 * u / self.f["scale"]
        self.text(x + 50 * u, y + 50 * u + fs * 0.36 * self.f["scale"], "CPU", fs, fg, "middle", "bold")

    def gpu(self, x, y, s, color, fg):
        """Graphics card icon, s px tall, 1.3 s wide."""
        u = s / 100
        self.rect(x + 9 * u, y + 10 * u, 7 * u, 86 * u, color, rx=2 * u)        # bracket
        self.rect(x, y + 10 * u, 16 * u, 6 * u, color, rx=2 * u)                # bracket tab
        self.rect(x + 19 * u, y + 10 * u, 111 * u, 68 * u, color, rx=5 * u)     # board
        self.rect(x + 40 * u, y + 76 * u, 60 * u, 11 * u, color, rx=1.5 * u)    # PCIe fingers
        cx, cy = x + 82 * u, y + 44 * u
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{28 * u:.1f}" fill="{fg}"/>')
        def pt(r, deg):
            t = math.radians(deg)
            return f"{cx + r * u * math.cos(t):.1f},{cy + r * u * math.sin(t):.1f}"
        for k in range(6):
            a = k * 60
            self.add(f'<path d="M{pt(5, a)} Q{pt(17, a + 4)} {pt(23, a + 30)} A{23 * u:.1f},{23 * u:.1f} 0 0 1 {pt(23, a + 58)} '
                     f'Q{pt(14, a + 40)} {pt(5, a + 40)} Z" fill="{color}"/>')
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{6 * u:.1f}" fill="{fg}"/>')

    def svg(self):
        th = self.th
        colors = {th["arrow"], th["hint"], th["ink"]}
        marks = "".join(f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
                        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors)
        tint = {th["hint"]: th["hint_bg"], th.get("trick"): th.get("trick_bg")}
        hatch = "".join(f'<pattern id="hatch-{col[1:]}" patternUnits="userSpaceOnUse" width="14" height="14" '
                        f'patternTransform="rotate(45)"><rect width="14" height="14" fill="{tint[col]}"/>'
                        f'<line x1="0" y1="0" x2="0" y2="14" stroke="{col}" stroke-width="5"/></pattern>'
                        for col in sorted(self.patterns))
        rough = ('<filter id="rough" x="-5%" y="-5%" width="110%" height="110%"><feTurbulence type="fractalNoise" '
                 'baseFrequency="0.022" numOctaves="2" seed="7"/><feDisplacementMap in="SourceGraphic" scale="5"/></filter>')
        body = "".join(self.out)
        if th["rough"]:
            body = f'<g filter="url(#rough)">{body}</g>'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
                f'<defs><style>{self.f["css"]}</style>{marks}{hatch}{rough}</defs>'
                f'<rect width="{W}" height="{H}" fill="{th["bg"]}"/>{body}</svg>')


def draw(theme, steps=False, solid=False, chart=True, heading="cpu", two=False, icons=None, sub=True):
    th = THEMES[theme]
    c = Canvas(th)
    c.solid = solid
    ink, muted, hint = th["ink"], th["muted"], th["hint"]
    col = {"hint": hint, "trick": th["trick"] if two else hint}
    tint = {"hint": th["hint_bg"], "trick": th["trick_bg"] if two else th["hint_bg"]}
    hand = th["font"] is HAND
    fg = th["panel"] if th["panel"] != "none" else th["bg"]

    # context
    x, cy, fs = 60, 470, 52
    for tok in ["this", "is", "a"]:
        x += c.chip(x, cy - fs * 0.875, tok, fs) + 14
    ctx_r = x - 14

    # CPU hint box
    bx, by, bw, bh = 520, 115, 530, 510
    if not chart:
        by, bh = 165, 345
    if icons == "header":
        bx, bw = 500, 565
    c.rect(bx, by, bw, bh, th["panel"], th["panel_stroke"], th["panel_sw"], rx=22)
    hx = bx + 32 + (92 if icons == "header" else 0)
    ty = by + 60 if sub else by + (79 if icons == "header" else 72)   # centre on the CPU icon
    if heading == "cpu":
        c.text(hx, ty, "Hints from the CPU", 36, ink, weight="bold")
        if sub:
            c.text(hx, by + 98, "training data", 22, muted)
    else:
        c.text(hx, ty, "Hints from training data", 36, ink, weight="bold")
        if sub:
            c.text(hx, by + 98, "(on CPU)", 22, muted)
    if icons == "above":
        c.cpu(bx + bw / 2 - 52, by - 122, 104, ink, th["bg"])
    elif icons == "header":
        c.cpu(bx + 22, by + 26, 84, ink, fg)
    for i, nxt in enumerate(["hint", "trick", "hint"]):
        y = by + 158 + i * 58
        pre = ["… so", "… but,", "… and"][i] + " this is a "
        c.text(bx + 32, y, pre, 30, muted)
        tx = bx + 32 + c.tw(pre, 30) + 4
        nw = c.chip(tx, y - 33, nxt, 30, stroke=col[nxt], weight="bold", sw=3, pad=0.45 if hand else 0.3, h=46,
                    fill=tint[nxt] if two else None)
        c.text(tx + nw + 8, y, "…", 30, muted)
    base, unit, bwid = by + bh - 62, 52, 84
    for j, (n, v) in enumerate(VOTES.items() if chart else []):
        x0 = bx + bw / 2 - 128 + j * 172
        for k in range(v):
            c.hatched(x0, base - (k + 1) * unit, bwid, unit - 4, col[n])
        c.text(x0 + bwid / 2, base + 36, n, 28, ink, "middle", "bold")
        c.text(x0 + bwid / 2, base - v * unit - 12, f"×{v}", 26, hint, "middle", "bold")
    if chart:
        c.add(f'<line x1="{bx + bw / 2 - 160}" y1="{base}" x2="{bx + bw / 2 + 160}" y2="{base}" stroke="{ink}" stroke-width="2.5"/>')

    # transformer box
    lx, ly, lw, lh = 1130, 115, 415, 690
    if icons == "above":
        ly, lh = 165, 640
    c.rect(lx, ly, lw, lh, th["panel"], th["panel_stroke"], th["panel_sw"], rx=22)
    hx = lx + 30 + (88 if icons == "header" else 0)
    c.text(hx, ly + 70 if sub else ly + 82, "Transformer", 44, ink, weight="bold")
    if sub:
        c.text(hx, ly + 108, "on the GPU" if heading == "cpu" else "(on GPU)", 22, muted)
    if icons == "above":
        c.gpu(lx + lw / 2 - 59, ly - 116, 90, ink, th["bg"])
    elif icons == "header":
        c.gpu(lx + 26, ly + 38, 56, ink, fg)
    label = "gain from hints"
    lt = lx + lw - 28 - c.tw(label, 22)   # right-aligned with a 28 px margin
    if two:
        c.hatched(lt - 76, ly + 150, 30, 30, col["hint"])
    c.hatched(lt - 40, ly + 150, 30, 30, col["trick"])
    c.text(lt, ly + 174, label, 22, ink)
    lbase, U, V, w3, g3, G = ly + lh - 120, 500, 118, 92, 32, 6
    x0 = lx + (lw - (3 * w3 + 2 * g3)) / 2
    for i, n in enumerate(["hint", "cool", "trick"]):
        xx = x0 + i * (w3 + g3)
        h = LM[n] * U
        if th["lm_stroke"] != "none":
            c.rect(xx + 1.25, lbase - h + 1.25, w3 - 2.5, h - 2.5, th["lm_fill"], th["lm_stroke"], 2.5, rx=5)
        else:
            c.rect(xx, lbase - h, w3, h, th["lm_fill"], rx=6)
        for k in range(VOTES.get(n, 0)):
            c.hatched(xx, lbase - h - (k + 1) * V, w3, V - G, col[n])   # G gap below every block
        c.text(xx + w3 / 2, lbase + 44, n, 32, ink, "middle", "bold")
    c.add(f'<line x1="{x0 - 16}" y1="{lbase}" x2="{x0 + 3 * w3 + 2 * g3 + 16}" y2="{lbase}" stroke="{ink}" stroke-width="2.5"/>')

    # arrows
    c.arrow(f"M{ctx_r + 10},{cy - 40} C{ctx_r + 60},{cy - 170} {bx - 80},{by + 140} {bx - 12},{by + 140}")
    low = max(by + bh + 80, 705)
    c.arrow(f"M{ctx_r + 10},{cy + 40} C{ctx_r + 30},{cy + 230} {bx},{low} {bx + bw / 2},{low} "
            f"S{lx - 70},{ly + lh - 200} {lx - 12},{ly + lh - 200}")
    c.arrow(f"M{bx + bw + 10},{by + min(250, bh / 2)} L{lx - 12},{by + min(250, bh / 2)}")

    if steps:
        def num(xc, yc, n):
            c.add(f'<circle cx="{xc}" cy="{yc}" r="22" fill="{hint}"/>')
            c.text(xc, yc + 9, str(n), 26, "#fff", "middle", "bold")
        num(ctx_r - 5, by + 185, 1)
        num(bx + bw / 2 - 210, base - 50, 2)
        num(lx - 50, by + 210, 3)
        y = 862
        parts = [(1, "find “this is a” in the training data"), (2, "count what came next"), (3, "the transformer gets the counts as a hint")]
        xx = 70
        for n, s in parts:
            num(xx + 22, y - 9, n)
            c.text(xx + 56, y, s, 26, ink)
            xx += 56 + c.tw(s, 26) + 50
    return c.svg()


VARIANTS = [("b1", "blog", {}), ("b2", "ink", {}), ("b3", "hand", {}), ("b4", "dark", {}), ("b5", "blog", dict(steps=True)),
            ("c1", "blog", {}),
            ("c2", "blog", dict(solid=True)),
            ("c3", "blog", dict(chart=False)),
            ("c4", "blog", dict(heading="training")),
            ("c5", "blog", dict(solid=True, chart=False, heading="training")),
            ("d1", "blog", dict(solid=True, chart=False, heading="training", two=True)),
            ("d2", "blog", dict(chart=False, heading="training", two=True)),
            ("d3", "blog", dict(solid=True, chart=False, two=True)),
            ("d4", "blog", dict(chart=False, two=True)),
            ("e1", "blog", dict(chart=False, heading="training", two=True, icons="above")),
            ("e2", "blog", dict(chart=False, heading="training", two=True, icons="above", sub=False)),
            ("e3", "blog", dict(chart=False, heading="training", two=True, icons="header", sub=False)),
            ("e4", "blog", dict(solid=True, chart=False, heading="training", two=True, icons="header", sub=False)),
            ("e6", "blog", dict(solid="tint", chart=False, heading="training", two=True, icons="header", sub=False)),
            ("e5", "hand", dict(chart=False, heading="training", two=True, icons="above", sub=False))]
if __name__ == "__main__":
    for name, theme, kw in VARIANTS:
        (HERE / f"retrieval_hints_{name}.svg").write_text(draw(theme, **kw))
    print("ok")
