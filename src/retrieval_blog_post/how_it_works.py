"""A static figure of how retrieval hints work, at a glance: the context ("hill climbing is") goes to the training data
on the CPU, where the tokens that followed it before (fun, hard, fun) join into one hint; on the GPU, the hint is
added to the stream after the embedding and after each transformer layer, and it lifts those tokens' bars (the
hinted gains, pointed at from their label over the bars).

The hint's line stays pink with orange dashes, for the two tokens it carries. The context, the middle matched line
and the hint's line sit at the height of the middle plus sign, so the hint's line runs straight across. The bars are
made up. Stacked, the context sits above the training data instead, in two columns, not three (for a tweet's narrower
image)."""

import math

from retrieval_blog_post.slot_index import line, rect, text

INK, MUTED = "#1c1c1c", "#75666e"
WHITE, PANEL = "#ffffff", "#ead9e1"  # the boxes' fill and border, on the card's #fffbfd
BLUE = "#3a7be6"  # the model's own guess
CONTEXT = ("hill", "climbing", "is")
MATCHES = (("… so", "fun"), ("… but,", "hard"), ("… yet", "fun"))  # the training data's lines, and their next tokens
COLOR = {"fun": "#fb64b6", "hard": "#fb923c"}  # a matched token's line and border
TINT = {"fun": "#ffe6f3", "hard": "#ffedd5"}  # its fill
GUESS = {"fun": 96, "solved": 216, "hard": 120}  # the bars' heights without the hint
GAIN = {"fun": 176, "hard": 104}  # what the hint adds on top

# Helvetica's advance widths, in thousandths of the font size, of the characters measured here
REGULAR = {
    **dict.fromkeys(" ,ft", 278),
    **dict.fromkeys("abdeghnou", 556),
    **dict.fromkeys("ilj", 222),
    **dict.fromkeys("cksvxyz", 500),
    "m": 833,
    "r": 333,
    "s": 500,
    "…": 1000,
}
BOLD = {
    " ": 278,
    "g": 611,
    "m": 889,
    **dict.fromkeys("acekosv", 556),
    **dict.fromkeys("bdhnpqu", 611),
    **dict.fromkeys("ijl", 278),
    **dict.fromkeys("ft", 333),
    "r": 389,
}
CTX_SIZE, LINE_SIZE = 32, 26  # the context's font size, and the training data's lines'

FY = 470  # the middle plus sign: the height of the context, the middle matched line and the hint's line
LX, LY, LW, LH = 935, 70, 635, 760  # the GPU's box
SX, SW, BH = LX + 124, 230, 68  # the stack's left edge, its blocks' width and height
STACK = (("embedding", LY + 150), ("transformer", LY + 325), ("transformer", LY + 475), ("LM head", LY + 625))


def width(s, size, table=REGULAR):
    return sum(table[c] for c in s) * size / 1000


def ctx_width(size=CTX_SIZE):
    return sum(width(tok, size) + 0.9 * size for tok in CONTEXT) + 12 * (len(CONTEXT) - 1)


def line_width(lead, size=LINE_SIZE):
    """A training-data line's width up to its next token: its lead, then the context it matched in bold."""
    return width(f"{lead} ", size) + width(" ".join(CONTEXT), size, BOLD) + width(" ", size)


CHIP_GAP = 8  # between a line's text (and the space after it) and its next token's chip
# the CPU's box: right of the context's arrow, and wide enough for its longest line plus room for the lines to join
BX, BY, BH_CPU = 40 + ctx_width() + 76, FY - 214, 345
BW = 32 + max(line_width(lead) + CHIP_GAP + width(tok, LINE_SIZE, BOLD) + 0.6 * LINE_SIZE
              for lead, tok in MATCHES) + 86 + 42
# stacked: the matched tokens' lines' run to where they join (from the widest chip), and the hint's fan's to the stack
JOIN, FAN = 48, 60
# the context's font size, the training data's lines' and the gap between them, the bars' labels' and the gap between
# the bars; stacked, larger, as the tweet shows the figure smaller (the bars apart enough for solved's larger label)
SIZES = {False: (CTX_SIZE, LINE_SIZE, 64, 27, 30), True: (38, 30, 68, 30, 44)}


def arrow_defs() -> str:
    return (
        "<defs>"
        + "".join(
            f"<marker id='how-arrow-{c[1:]}' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='5' "
            f"markerHeight='5' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{c}'/></marker>"
            for c in (INK, COLOR["fun"])
        )
        + "</defs>"
    )


def path(d, color=INK, w=3.5, head=False, cap="round", extra=""):
    marker = f"marker-end='url(#how-arrow-{color[1:]})'" if head else ""
    return f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{w}' stroke-linecap='{cap}' {marker} {extra}/>"


def chip(x, y, s, size, h, pad, stroke=INK, fill=WHITE, weight="normal", sw=2.5):
    """A token in a rounded box with its top left at (x, y); returns its svg and width."""
    w = width(s, size, BOLD if weight == "bold" else REGULAR) + 2 * pad * size
    return (
        rect(x, y, w, h, fill, stroke, min(12, h / 4), f"stroke-width='{sw}'")
        + text(x + w / 2, y + h / 2 + size * 0.36, s, size, "middle", INK, weight)
    ), w


def cpu(x, y, s):
    """A CPU chip, s px square with its pins."""
    u = s / 100
    o = [rect(x + 17 * u, y + 17 * u, 66 * u, 66 * u, INK, rx=9 * u)]
    for p in (28.5, 42.5, 57.5, 71.5):
        for px, py, pw, ph in ((p - 3, 4, 6, 15), (p - 3, 81, 6, 15), (4, p - 3, 15, 6), (81, p - 3, 15, 6)):
            o.append(rect(x + px * u, y + py * u, pw * u, ph * u, INK, rx=3 * u))
    o.append(text(x + 50 * u, y + 50 * u + 22 * u * 0.36, "CPU", 22 * u, "middle", WHITE, "bold"))
    return "".join(o)


def gpu(x, y, s):
    """A graphics card, s px tall and 1.3 s wide: its bracket, board, connector and a six-bladed fan."""
    u = s / 100
    o = [
        rect(x + 9 * u, y + 10 * u, 7 * u, 86 * u, INK, rx=2 * u),
        rect(x, y + 10 * u, 16 * u, 6 * u, INK, rx=2 * u),
        rect(x + 19 * u, y + 10 * u, 111 * u, 68 * u, INK, rx=5 * u),
        rect(x + 40 * u, y + 76 * u, 60 * u, 11 * u, INK, rx=1.5 * u),
    ]
    cx, cy = x + 82 * u, y + 44 * u

    def pt(r, deg):
        t = math.radians(deg)
        return f"{cx + r * u * math.cos(t):.1f},{cy + r * u * math.sin(t):.1f}"

    o.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{28 * u:.1f}' fill='{WHITE}'/>")
    for k in range(6):
        a = k * 60
        o.append(
            f"<path d='M{pt(5, a)} Q{pt(17, a + 4)} {pt(23, a + 30)} A{23 * u:.1f},{23 * u:.1f} 0 0 1 {pt(23, a + 58)} "
            f"Q{pt(14, a + 40)} {pt(5, a + 40)} Z' fill='{INK}'/>"
        )
    o.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{6 * u:.1f}' fill='{WHITE}'/>")
    return "".join(o)


def gain_block(x, y, w, h, token):
    """A block of what the hint adds, drawn like its token's chip; its border's outer edge is (x, y, w, h)."""
    return rect(x + 1.5, y + 1.5, w - 3, h - 3, TINT[token], COLOR[token], 7, "stroke-width='3'")


def how_it_works_svg(stacked: bool = False) -> str:
    """The figure; `stacked`, with the context centred above the training data's box, an arrow down into it, and the
    drawing cropped to the two columns left (the training data and the model)."""
    o = [arrow_defs()]
    ctx_size, line_size, row, bar_size, g3 = SIZES[stacked]

    # the CPU's box's width, the GPU's box's left edge and width, the stack's left edge, and where the hint joins and
    # fans out; stacked, closer: the hint joins just right of the chips and fans out between the boxes, and the GPU's
    # box starts its card's margin left of the stack, so that its card sits over the embedding's left edge
    if stacked:
        mx = BX + 32 + max(line_width(lead, line_size) + CHIP_GAP + width(tok, line_size, BOLD) + 0.6 * line_size
                           for lead, tok in MATCHES) + 6 + JOIN
        bw = mx + 12 - BX
        fx = BX + bw + 6
        sx = fx + FAN + 10
        gpu_x, gpu_w = sx - 28, LX + LW - SX + 28 + 2 * (g3 - 30)
    else:
        bw, gpu_x, gpu_w, sx, mx, fx = BW, LX, LW, SX, BX + BW - 42, LX + 36

    # the context: left of the training data, at the height of its middle line; or stacked, above it
    size = ctx_size
    x0, cy = (BX + (bw - ctx_width(size)) / 2, BY - 110) if stacked else (40, FY)
    x = x0
    for tok in CONTEXT:
        svg, w = chip(x, cy - size * 0.875, tok, size, size * 1.75, 0.45)
        o.append(svg)
        x += w + 12
    ctx_mid, ctx_right, ctx_top = (x0 + x - 12) / 2, x - 12, cy - size * 0.875

    # the training data, on the CPU
    o.append(rect(BX, BY, bw, BH_CPU, WHITE, PANEL, 22, "stroke-width='3'"))
    o.append(cpu(BX + 22, BY + 20, 76))
    o.append(text(BX + 114, BY + 70, "Training data", 36, "start", INK, "bold"))
    if stacked:
        o.append(path(f"M{ctx_mid},{cy + size * 0.875 + 12} V{BY - 14}", head=True))
    else:
        o.append(path(f"M{ctx_right + 12},{FY} L{BX - 14},{FY}", head=True))
    ends = []  # where each matched token's line starts, and its token
    for i, (lead, token) in enumerate(MATCHES):
        y, h = BY + 224 + (i - 1) * row, round(46 * line_size / 30)  # the middle line's chip at the height FY
        # the lead in gray, the context it matched in bold black
        o.append(text(BX + 32, y, f"{lead} ", line_size, "start", MUTED))
        o.append(text(BX + 32 + width(f"{lead} ", line_size), y, " ".join(CONTEXT), line_size, "start", INK, "bold"))
        tx = BX + 32 + line_width(lead, line_size) + CHIP_GAP
        svg, w = chip(tx, y - 10 - h / 2, token, line_size, h, 0.3, COLOR[token], TINT[token], "bold", 3)
        o.append(svg)
        ends.append((tx + w, y - 10, token))

    # the model, on the GPU: the stack, with a plus sign between each pair of its blocks
    o.append(rect(gpu_x, LY, gpu_w, LH, WHITE, PANEL, 22, "stroke-width='3'"))
    o.append(gpu(gpu_x + 28, LY + 32, 60))
    mid = sx + SW / 2
    for label, y in STACK:
        o.append(rect(sx, y - BH / 2, SW, BH, WHITE, INK, 12, "stroke-width='2.5'"))
        o.append(text(mid, y + 29 * 0.36, label, 29, "middle", INK, "bold"))
    adds = []  # the plus signs' heights
    for (_, a), (_, b) in zip(STACK, STACK[1:]):
        m = (a + b) / 2
        o.append(line(mid, a + BH / 2, mid, m - 15, INK, 3))
        o.append(f"<circle cx='{mid}' cy='{m}' r='15' fill='{WHITE}' stroke='{INK}' stroke-width='3'/>")
        o.append(f"<path d='M{mid - 8},{m} H{mid + 8} M{mid},{m - 8} V{m + 8}' stroke='{INK}' stroke-width='3'/>")
        o.append(path(f"M{mid},{m + 15} L{mid},{b - BH / 2 - 6}", w=3, head=True))
        adds.append(m)
    assert adds[1] == FY

    # the matched tokens join in the CPU's box into one hint, which crosses to the GPU and fans out to the plus signs
    f = (sx - 10 - fx) / 78  # the fan's run, to the unstacked one's
    for ex, ey, token in ends:
        o.append(path(f"M{ex + 6},{ey} C{ex + 50},{ey} {mx - 50},{FY} {mx},{FY}", COLOR[token]))
    hint = [f"M{mx},{FY} H{fx}"] + [
        f"M{fx},{FY} C{fx + 50 * f:g},{FY} {sx - 10 - 60 * f:g},{y} {sx - 10},{y} H{mid - 23}"  # level before the stack
        for y in adds
    ]
    o.append(path(hint[0], COLOR["fun"], cap="butt"))
    o += [path(d, COLOR["fun"], head=True) for d in hint[1:]]
    o += [path(d, COLOR["hard"], cap="butt", extra="stroke-dasharray='22 22'") for d in hint]

    # the context also goes into the embedding: straight up, one bend, straight in; stacked, right from it and down to
    # the embedding's height before the GPU's box (clear of the GPU)
    ye, r = STACK[0][1], 40
    if stacked:
        o.append(path(f"M{ctx_right + 12},{cy} H{gpu_x - 90} C{gpu_x - 40},{cy} {gpu_x - 40},{ye} {gpu_x + 10},{ye} H{sx - 8}",
                      head=True))
    else:
        o.append(path(f"M{ctx_mid},{ctx_top - 12} V{ye + r} Q{ctx_mid},{ye} {ctx_mid + r},{ye} H{sx - 8}", head=True))

    # the bars, and over them "hinted gains", a line from it down to each gain block (in the block's color)
    w3, gap = 54, 6  # g3: wide gaps, so that solved's label clears its neighbours'
    x0, base = sx + 258, STACK[-1][1] + BH / 2
    gains = []  # each gain block's top middle, and its token
    for i, token in enumerate(GUESS):
        xx, h = x0 + i * (w3 + g3), GUESS[token]
        o.append(rect(xx, base - h, w3, h, BLUE, rx=6))
        if token in GAIN:
            top = base - h - gap - GAIN[token]
            o.append(gain_block(xx, top, w3, GAIN[token], token))
            gains.append((xx + w3 / 2, top, token))
        o.append(text(xx + w3 / 2, base + 38, token, bar_size, "middle", INK, "bold"))
    # the label's middle and its two lines' baselines
    size = 34
    lx, ly = x0 + 1.5 * w3 + g3, min(top for _, top, _ in gains) - 76
    o.append(text(lx, ly - 1.15 * size, "hinted", size, "middle", INK, "bold"))
    o.append(text(lx, ly, "gains", size, "middle", INK, "bold"))
    for gx, top, token in gains:
        sx = lx + (0.3 if gx > lx else -0.3) * width("hinted", size, BOLD)
        o.append(line(sx, ly + 16, gx + (sx - gx) * 0.15, top - 10, COLOR[token], 3))
    o.append(line(x0 - 10, base, x0 + 3 * w3 + 2 * g3 + 10, base, INK, 2.5))

    left = BX - 40 if stacked else 0  # stacked, the context's column is empty
    return (
        f"<svg viewBox='{20 + left:g} 50 {gpu_x + gpu_w - left:g} 800' xmlns='http://www.w3.org/2000/svg' role='img' "
        "aria-label='How retrieval hints work'>" + "".join(o) + "</svg>"
    )
