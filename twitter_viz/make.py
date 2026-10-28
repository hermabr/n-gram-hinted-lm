"""Five variants of the retrieval-hint sketch, 1600x900 SVGs for Twitter."""
import base64
import json
from pathlib import Path

W, H = 1600, 900
FONT = "HintSans, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "HintMono, 'DejaVu Sans Mono', Menlo, Consolas, monospace"
FONTS = Path(__file__).parent / "fonts"


def font_face(family, file, weight):
    data = base64.b64encode((FONTS / file).read_bytes()).decode()
    return (f"@font-face{{font-family:{family};font-weight:{weight};"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")


FONT_CSS = "".join([font_face("HintSans", "NimbusSans-Regular.woff2", 400),
                    font_face("HintSans", "NimbusSans-Bold.woff2", 700),
                    font_face("HintMono", "DejaVuSansMono.woff2", 400),
                    font_face("HintMono", "DejaVuSansMono-Bold.woff2", 700)])
BG, INK, MUTED, LINE = "#fffbfd", "#1c1c1c", "#75666e", "#ead9e1"
PINK, BLUE = "#fb64b6", "#3a7be6"
CONTEXT = ["this", "is", "a", "useful"]
LM = {"hint": 0.18, "trick": 0.24, "idea": 0.40}   # the LM's own guess
VOTES = {"hint": 2, "trick": 1, "idea": 0}           # next tokens seen in training data


WIDTHS = json.loads((FONTS / "widths.json").read_text())


def tw(s, fs, weight="normal"):  # Nimbus Sans advance widths
    return sum(WIDTHS[weight].get(c, 0.6) for c in s) * fs


def text(x, y, s, fs=32, fill=INK, anchor="start", weight="normal", family=FONT, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}" {extra}>{s}</text>')


def chip(x, y, label, fs=40, h=None, w=None, fill="#fff", stroke=INK, color=INK, sw=2.5, weight="normal", rx=12):
    h = h or fs * 1.8
    w = w or tw(label, fs, weight) + fs * 0.9
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>' + text(x + w / 2, y + h / 2 + fs * 0.35, label, fs, color, "middle", weight)), w


def chips(x, y, labels, fs=40, gap=14, **kw):
    out = []
    for lab in labels:
        s, w = chip(x, y, lab, fs, **kw)
        out.append(s)
        x += w + gap
    return "".join(out), x - gap


def arrow(d, color=INK, sw=3, dash=""):
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'{dash} marker-end="url(#ah-{color[1:]})"/>')


def svg(body, bg=BG):
    marks = "".join(
        f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c in (INK, PINK, BLUE, MUTED, "#ffffff", "#a99aa2"))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<defs><style>{FONT_CSS}</style>{marks}</defs><rect width="{W}" height="{H}" fill="{bg}"/>{body}</svg>')


def bars(x0, base, names, lm, votes, bw=100, gap=50, unit=380, vote_h=95, fs=30,
         lm_fill=BLUE, vote_fill=PINK, label_fill=INK, vote_text=True, outline=None):
    """LM bar with the retrieval votes stacked on top; returns svg and per-bar geometry."""
    out, geo = [], {}
    for i, n in enumerate(names):
        x = x0 + i * (bw + gap)
        h = lm.get(n, 0) * unit
        out.append(f'<rect x="{x}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" rx="6" fill="{lm_fill}"/>')
        blocks = []
        top = base - h
        for v in range(votes.get(n, 0)):
            y = top - vote_h
            out.append(f'<rect x="{x}" y="{y + 4:.1f}" width="{bw}" height="{vote_h - 4}" rx="6" fill="{vote_fill}"'
                       + (f' stroke="{outline}" stroke-width="2"' if outline else "") + "/>")
            if vote_text:
                out.append(text(x + bw / 2, y + vote_h / 2 + 13, "+1", 34, "#fff", "middle", "bold"))
            blocks.append((x, y + 4, bw, vote_h - 4))
            top = y
        out.append(text(x + bw / 2, base + fs + 14, n, fs, label_fill, "middle", "bold"))
        geo[n] = dict(x=x, top=top, blocks=blocks)
    out.append(f'<line x1="{x0 - 20}" y1="{base}" x2="{x0 + len(names) * (bw + gap) - gap + 20}" y2="{base}" '
               f'stroke="{label_fill}" stroke-width="2.5"/>')
    return "".join(out), geo


def legend(x, y, fs=26, fill=INK, items=((BLUE, "the LM's own guess"), (PINK, "votes from the training data"))):
    out = []
    for c, lab in items:
        out.append(f'<rect x="{x}" y="{y - fs * 0.8:.1f}" width="{fs}" height="{fs}" rx="5" fill="{c}"/>')
        out.append(text(x + fs * 1.5, y, lab, fs, fill))
        x += fs * 1.5 + tw(lab, fs) + 50
    return "".join(out)


# ---------------------------------------------------------------- 1: faithful redraw
def v1():
    b = []
    b.append(text(70, 95, "Retrieval hints: before the LM guesses, check what came next in the training data", 38, INK, weight="bold"))
    # CPU panel
    b.append(f'<rect x="60" y="150" width="860" height="690" rx="26" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
    b.append(text(100, 215, "CPU", 40, PINK, weight="bold") + text(110 + tw("CPU", 40, "bold"), 215, "· search the training data", 36, MUTED))
    row_y = 455
    s, end = chips(100, row_y, CONTEXT, 46, gap=16)
    b.append(s)
    b.append(arrow(f"M{end + 18},{row_y + 41} L{end + 70},{row_y + 41}", INK, 3.5))
    cx = end + 90
    nexts = [("trick", 285), ("hint", row_y), ("hint", 625)]
    cw = 190
    for lab, y in nexts:
        s, _ = chip(cx, y, lab, 46, w=cw, stroke=PINK, color=INK, sw=4, weight="bold")
        b.append(s)
    b.append(text(cx + cw / 2, 790, "3 matches → 3 votes", 28, MUTED, "middle"))
    # GPU panel
    b.append(f'<rect x="1000" y="150" width="540" height="690" rx="26" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
    b.append(text(1040, 215, "GPU", 40, BLUE, weight="bold") + text(1050 + tw("GPU", 40, "bold"), 215, "· the LM predicts", 36, MUTED))
    s, end2 = chips(1040, 255, CONTEXT, 26, gap=8, sw=2)
    b.append(s + text(end2 + 14, 255 + 33, "→ ?", 28, INK, weight="bold"))
    bs, g = bars(1085, 735, ["hint", "trick", "idea"], LM, VOTES, bw=100, gap=50, unit=330, vote_h=110)
    b.append(bs)
    b.append(legend(1060, 815, 24, items=((BLUE, "LM's guess"), (PINK, "training-data votes"))))
    # arrows: retrieved tokens -> vote blocks
    hb, tb = g["hint"]["blocks"], g["trick"]["blocks"]
    sx = cx + cw + 6
    ty = nexts[0][1] + 41
    tx, tyb = tb[0][0] + tb[0][2] / 2, tb[0][1]
    b.append(arrow(f"M{sx},{ty} C{sx + 200},{ty} {tx},{ty} {tx},{tyb - 10}", PINK, 4))
    for (lab, y), blk in zip(nexts[1:], [hb[1], hb[0]]):
        yy = y + 41
        ex, ey = blk[0] - 8, blk[1] + blk[3] / 2
        b.append(arrow(f"M{sx},{yy} C{sx + 110},{yy} {ex - 110},{ey} {ex},{ey}", PINK, 4))
    return svg("".join(b))


# ---------------------------------------------------------------- 2: three steps
def v2():
    b = []
    b.append(text(800, 100, "Retrieval hints in three steps", 52, INK, "middle", "bold"))
    cols = [(60, "1", "Find the context"),
            (660, "2", "Count what came next"),
            (1150, "3", "Nudge the LM")]
    for x, n, t in cols:
        b.append(f'<circle cx="{x + 24}" cy="{190}" r="24" fill="{INK}"/>' + text(x + 24, 200, n, 30, "#fff", "middle", "bold"))
        b.append(text(x + 62, 201, t, 30, INK, weight="bold"))
    # step 1: three snippets from different documents
    snips = [("… so", "hint", ""), ("… honestly", "trick", ""), ("… and", "hint", "")]
    for i, (pre, nxt, post) in enumerate(snips):
        y = 260 + i * 175
        b.append(f'<rect x="60" y="{y}" width="500" height="140" rx="18" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
        b.append(text(84, y + 46, f"training document {['#12,804', '#3,551,207', '#90,113'][i]}", 22, MUTED))
        b.append(f'<text x="84" y="{y + 100}" font-size="31" font-family="{FONT}" fill="#a99aa2">{pre} '
                 f'<tspan fill="{INK}" font-weight="bold">this is a useful</tspan> '
                 f'<tspan fill="{PINK}" font-weight="bold" text-decoration="underline">{nxt}</tspan></text>')
        b.append(arrow(f"M575,{y + 70} C600,{y + 70} 600,{450} 640,{450 + (i - 1) * 40}", "#a99aa2", 2.5))
    # step 2: tally
    b.append(f'<rect x="660" y="320" width="380" height="270" rx="18" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
    for i, (n, v) in enumerate([("hint", 2), ("trick", 1), ("idea", 0)]):
        y = 385 + i * 80
        b.append(text(700, y + 12, n, 38, INK, weight="bold"))
        for k in range(v):
            b.append(f'<circle cx="{850 + k * 56}" cy="{y}" r="20" fill="{PINK}"/>')
        b.append(text(1000, y + 12, f"×{v}", 36, PINK if v else "#c9b9c1", "end", "bold"))
    b.append(text(850, 650, "these counts are the hint", 26, MUTED, "middle"))
    b.append(arrow("M1055,455 L1120,455", INK, 3.5))
    # step 3: bars
    bs, _ = bars(1170, 690, ["hint", "trick", "idea"], LM, VOTES, bw=95, gap=45, unit=380, vote_h=110)
    b.append(bs)
    b.append(text(1355, 290, "this is a useful ___", 28, MUTED, "middle"))
    b.append(legend(470, 835, 26))
    return svg("".join(b))


# ---------------------------------------------------------------- 3: dark, corpus wall
FILLER = ("the model was trained on a large mix of web pages and books which covers many topics "
          "from cooking to physics and some of it repeats in ways that are easy to exploit if you "
          "can look things up fast enough during training without slowing down the gpu at all and "
          "most phrases have been seen before in some form by the time the model reads them again ").split()


def v3():
    D_BG, D_INK, D_DIM, D_LINE = "#141117", "#f4eef1", "#4a4148", "#2c2530"
    b = []
    b.append(text(70, 95, "The training data already knows what usually comes next", 42, D_INK, weight="bold"))
    b.append(text(70, 140, "so we look it up on the CPU and hand it to the LM as a hint", 30, "#a99aa2"))
    fs, cw = 25, 25 * 0.602
    x0, y0, lh = 70, 215, 46
    b.append(f'<rect x="45" y="175" width="905" height="680" rx="22" fill="#1b171f" stroke="{D_LINE}" stroke-width="2"/>')
    hits = {1: "trick", 6: "hint", 10: "hint"}
    ends, k = {}, 0
    ncols = 58
    for ln in range(13):
        y = y0 + 30 + ln * lh
        if ln in hits:
            pre_words = []
            while len(" ".join(pre_words)) < {1: 18, 6: 30, 10: 8}[ln]:
                pre_words.append(FILLER[k % len(FILLER)]); k += 1
            pre = " ".join(pre_words) + " "
            ctx = "this is a useful "
            nxt = hits[ln]
            pre_x = x0
            ctx_x = pre_x + len(pre) * cw
            nxt_x = ctx_x + len(ctx) * cw
            b.append(text(pre_x, y, pre.replace(" ", " "), fs, D_DIM, family=MONO, extra=f'textLength="{len(pre.rstrip()) * cw:.1f}"'))
            b.append(f'<rect x="{ctx_x - 4:.1f}" y="{y - fs + 1}" width="{(len(ctx) - 1) * cw + 8:.1f}" height="{fs + 10}" rx="5" fill="#3a3340"/>')
            b.append(text(ctx_x, y, ctx.strip(), fs, D_INK, family=MONO, extra=f'textLength="{(len(ctx) - 1) * cw:.1f}"'))
            nw = len(nxt) * cw
            b.append(f'<rect x="{nxt_x - 5:.1f}" y="{y - fs + 1}" width="{nw + 10:.1f}" height="{fs + 10}" rx="5" fill="{PINK}"/>')
            b.append(text(nxt_x, y, nxt, fs, "#1b171f", weight="bold", family=MONO, extra=f'textLength="{nw:.1f}"'))
            ends[ln] = (nxt_x + nw + 5, y - fs / 2 + 4)
        else:
            words, used = [], 0
            while used + len(FILLER[k % len(FILLER)]) + 1 <= ncols:
                words.append(FILLER[k % len(FILLER)]); used += len(words[-1]) + 1; k += 1
            line = " ".join(words)
            b.append(text(x0, y, line, fs, D_DIM, family=MONO, extra=f'textLength="{len(line) * cw:.1f}"'))
    b.append(text(70, 840, "CPU · training data", 24, "#a99aa2", weight="bold"))
    # GPU side
    b.append(f'<rect x="1030" y="175" width="525" height="680" rx="22" fill="#1b171f" stroke="{D_LINE}" stroke-width="2"/>')
    b.append(text(1060, 840, "GPU · language model", 24, "#a99aa2", weight="bold"))
    s, e = chips(1060, 215, CONTEXT, 26, gap=8, fill="#2a2430", stroke="#2a2430", color=D_INK, sw=1)
    b.append(s + text(e + 14, 215 + 33, "→ ?", 28, D_INK, weight="bold"))
    bs, g = bars(1105, 740, ["hint", "trick", "idea"], LM, VOTES, bw=100, gap=50, unit=330, vote_h=105,
                 lm_fill="#5b8ff0", label_fill=D_INK)
    b.append(bs)
    # arrows from the pink hits to the vote blocks
    hb, tb = g["hint"]["blocks"], g["trick"]["blocks"]
    targets = {1: (tb[0][0] + 50, tb[0][1] - 8, "top"), 6: (hb[1][0] - 8, hb[1][1] + 50, "side"), 10: (hb[0][0] - 8, hb[0][1] + 50, "side")}
    for ln, (sx, sy) in ends.items():
        ex, ey, kind = targets[ln]
        if kind == "top":
            d = f"M{sx + 6},{sy} C{sx + 300},{sy - 120} {ex - 60},{ey - 160} {ex},{ey}"
        else:
            d = f"M{sx + 6},{sy} L960,{sy} C1030,{sy} 1020,{ey} {ex},{ey}"
        b.append(arrow(d, PINK, 3.5))
    return svg("".join(b), D_BG)


# ---------------------------------------------------------------- 4: CPU / GPU lanes, closer to the real wiring
def v4():
    b = []
    b.append(text(70, 90, "Two processors, one prediction", 46, INK, weight="bold"))
    b.append(text(70, 135, "The CPU looks up the context in the training data while the GPU runs the model", 28, MUTED))
    # lanes
    b.append(f'<rect x="60" y="175" width="1480" height="270" rx="24" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
    b.append(f'<rect x="60" y="475" width="1480" height="380" rx="24" fill="#fff" stroke="{LINE}" stroke-width="3"/>')
    b.append(text(90, 220, "CPU", 32, PINK, weight="bold"))
    b.append(text(90, 520, "GPU", 32, BLUE, weight="bold"))
    # CPU lane: context -> lookup table -> counts
    s, e = chips(150, 275, CONTEXT, 34, gap=10)
    b.append(s)
    b.append(arrow(f"M{e + 20},305 L{e + 90},305", INK, 3))
    tx = e + 110
    b.append(f'<rect x="{tx}" y="235" width="430" height="150" rx="14" fill="{BG}" stroke="{INK}" stroke-width="2.5"/>')
    b.append(text(tx + 215, 275, "index of the training data", 22, MUTED, "middle"))
    rows = [("…", "…"), ("this is a useful", "hint ×2 · trick ×1"), ("…", "…")]
    for i, (k_, v_) in enumerate(rows):
        y = 310 + i * 30
        col = INK if i == 1 else "#c9b9c1"
        b.append(text(tx + 20, y, k_, 21, col, weight="bold" if i == 1 else "normal"))
        b.append(text(tx + 410, y, v_, 21, PINK if i == 1 else "#c9b9c1", "end", "bold" if i == 1 else "normal"))
    hx = tx + 520
    b.append(arrow(f"M{tx + 440},310 L{hx - 10},310", INK, 3))
    # hint pill
    b.append(f'<rect x="{hx}" y="270" width="200" height="80" rx="40" fill="{PINK}"/>')
    b.append(text(hx + 100, 320, "hint", 36, "#fff", "middle", "bold"))
    # GPU lane: tokens -> blocks -> output bars
    s, e = chips(150, 640, CONTEXT, 30, gap=10)
    b.append(s)
    lx = e + 60
    b.append(arrow(f"M{e + 12},667 L{lx - 8},667", INK, 3))
    nlay = 6
    for i in range(nlay):
        b.append(f'<rect x="{lx + i * 62}" y="560" width="46" height="215" rx="10" fill="{BLUE}" opacity="{0.55 + 0.07 * i:.2f}"/>')
    b.append(text(lx + (nlay * 62 - 16) / 2, 815, "transformer layers", 22, MUTED, "middle"))
    # hint enters the residual stream (three places)
    for xi in (lx - 22, lx + 3 * 62 - 8, lx + nlay * 62 - 2):
        b.append(arrow(f"M{hx + 100},352 C{hx + 100},450 {xi},450 {xi},548", PINK, 3))
    b.append(text(hx + 125, 402, "into the residual stream", 22, PINK, "start"))
    ox = lx + nlay * 62 + 30
    b.append(arrow(f"M{ox - 8},667 L{ox + 50},667", INK, 3))
    final = {"hint": 0.62, "trick": 0.24, "idea": 0.14}
    bs, _ = bars(ox + 90, 780, ["hint", "trick", "idea"], final, {}, bw=80, gap=40, unit=300, fs=26)
    b.append(bs)
    for i, n in enumerate(["hint", "trick", "idea"]):
        h = LM[n] / sum(LM.values()) * 300
        b.append(f'<rect x="{ox + 90 + i * 120}" y="{780 - h:.1f}" width="80" height="{h:.1f}" rx="6" fill="none" '
                 f'stroke="{INK}" stroke-width="2" stroke-dasharray="7 6"/>')
    b.append(f'<rect x="{ox + 440}" y="700" width="22" height="22" rx="4" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="5 4"/>')
    b.append(text(ox + 472, 718, "without the hint", 20, MUTED))
    b.append(f'<rect x="{ox + 440}" y="740" width="22" height="22" rx="4" fill="{BLUE}"/>')
    b.append(text(ox + 472, 758, "with the hint", 20, MUTED))
    b.append(text(ox + 90 + 140, 545, "next-token guess", 22, MUTED, "middle"))
    return svg("".join(b))


# ---------------------------------------------------------------- 5: equation
def v5():
    b = []
    b.append(text(800, 150, '<tspan>this is a useful </tspan><tspan fill="#c9b9c1">___</tspan>', 64, INK, "middle", "bold"))
    cols = [(110, "what the LM thinks", LM, {}, BLUE),
            (620, "what the training data says", {}, VOTES, PINK),
            (1130, "the prediction", LM, VOTES, None)]
    base = 710
    for x, title, lm, votes, c in cols:
        b.append(text(x + 180, 300, title, 32, c or INK, "middle", "bold"))
        bs, _ = bars(x + 20, base, ["hint", "trick", "idea"], lm, votes, bw=90, gap=45, unit=420, vote_h=125, fs=30,
                     vote_text=False)
        b.append(bs)
    for x, sym in ((560, "+"), (1070, "=")):
        b.append(text(x, 600, sym, 110, INK, "middle", "bold"))
    b.append(text(300, 800, "a small transformer on the GPU", 26, MUTED, "middle"))
    b.append(text(810, 800, "an n-gram lookup on the CPU", 26, MUTED, "middle"))
    b.append(text(1320, 800, "hint wins", 26, PINK, "middle", "bold"))
    return svg("".join(b))


out = Path(__file__).parent
for i, f in enumerate([v1, v2, v3, v4, v5], 1):
    (out / f"retrieval_hints_v{i}.svg").write_text(f())
print("ok")
