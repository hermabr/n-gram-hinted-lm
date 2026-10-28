"""A static figure of a simple SlotIndex that indexes 10-grams only: where one 10-gram's entry goes.

The 10 context tokens, hashed, and the next token make one bar: the hash's first 14 bits (the partition, not stored),
the next 14 (bucket 1), the 14 after those (bucket 2; a simplification: exact_match/src/slot.rs's second bucket is a
hash of its whole 30-bit check, the first's bits included) and the rest of its 62 bits, then the 16-bit next token.
Under the bar, hints[partition][bucket].append(checksum + next token): an arrow from each piece of the bar to its place
in the line, bucket 1's and bucket 2's dashed and merging into one, labelled that the entry goes in the emptier bucket.
The entry (64 bits) is everything but the partition: the rest of the checksum and the next token.

The page shows it without the pieces' sizes, and with them (in bits, and the entry's 64 stored bits) while the mouse is over it.

simple_index_svg draws the same 10-gram as the Python version stores it: hints[hash].append(next_token).

On phones (`phone`), both are narrower: only the last six tokens, "we train gpt-2 in just 6.7", the next token beside
them; the bar's pieces
are only as wide as their labels; and the hints[...] line breaks before .append(, its second line pushed right so that
the green arrow passes its first."""

from html import escape

FONT = "font-family=\"'Helvetica Neue', Helvetica, Arial, sans-serif\""
INK, GREY, MUTED = "#1c1c1c", "#8a8a8a", "#666"
PART, LPART = "#7a4fa0", "#e6dbf0"  # partition: dark, light
BUCK, LBUCK = "#1f5fa8", "#c9dbf0"  # bucket
NEXT, LNEXT = "#c86b12", "#f6dcc2"  # next token
REST, LREST = "#9a9a9a", "#ececec"  # the hash's other bits
STORE, LSTORE = "#2a8a4a", "#cfe8d6"  # the stored entry
SURFACE = "var(--surface, #ffffff)"  # the card's background, for what breaks the lines it covers

# the 10-gram and its next token, a word to a token (not GPT-2's tokens: those split NanoGPT and GPT-2), all lower
# case so that no word looks bigger than the rest for its capitals
CONTEXT = ("in", "the", "nanogpt", "speedrun", "we", "train", "gpt-2", "in", "just", "6.7")
NEXT_TOKEN = "seconds"

W = 960  # the drawing's width
Y, H = 130, 44  # the bar's top and height
# the bar's pieces' x ranges
SEG = {
    "part": (100, 290),
    "b1": (290, 430),
    "b2": (430, 570),
    "rest": (570, 690),
    "next": (700, 900),
}
# the centre of the hints[...] line's bucket box, under the boundary of bucket 1 and bucket 2, so that their arrows
# merge symmetrically; the line is laid out around it
EQ_BUCKET = SEG["b1"][1]
EQ_Y = 300  # the top of the hints[...] line's boxes
ENTRY_W = 204  # the width of its entry's box: "checksum + next token" (177 in Helvetica Bold 16) and a little room


def text(
    x, y, s, size=13, anchor="start", fill=INK, weight="normal", font=FONT, extra=""
):
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' fill='{fill}' "
        f"font-weight='{weight}' {font} {extra}>{escape(s, quote=False)}</text>"
    )


def rect(x, y, w, h, fill, stroke="none", rx=0, extra=""):
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' fill='{fill}' stroke='{stroke}' "
        f"rx='{rx}' {extra}/>"
    )


def line(x1, y1, x2, y2, stroke, width=1, extra=""):
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' "
        f"stroke-width='{width}' {extra}/>"
    )


def brace(x1, x2, y, color, up=False, h=7, stub=5, tick=None):
    """A square bracket under (or over) [x1, x2], its ends pointing at what it spans, a stub out of it at x = tick
    (the middle by default)."""
    s = -1 if up else 1
    m = (x1 + x2) / 2 if tick is None else tick
    return (
        f"<path d='M{x1:.1f},{y - s * h:.1f} V{y:.1f} H{x2:.1f} V{y - s * h:.1f} M{m:.1f},{y:.1f} "
        f"V{y + s * stub:.1f}' stroke='{color}' stroke-width='1.4' fill='none'/>"
    )


def defs() -> str:
    """Arrowheads in each colour, and the hatching of the bits we don't store."""
    marks = "".join(
        f"<marker id='slot-arrow-{c[1:]}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' "
        f"markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{c}'/></marker>"
        for c in (PART, BUCK, GREY, STORE, NEXT)
    )
    hatch = (
        "<pattern id='slot-hatch' width='7' height='7' patternUnits='userSpaceOnUse' "
        f"patternTransform='rotate(45)'><line x1='0' y1='0' x2='0' y2='7' stroke='{PART}' "
        "stroke-width='1.2' opacity='0.35'/></pattern>"
    )
    return f"<defs>{marks}{hatch}</defs>"


def arrow(d, color, width=1.6, dash=False):
    """A path ending in an arrowhead, on a band of the background, so the lines it crosses break around it."""
    da = " stroke-dasharray='5 4'" if dash else ""
    return (
        f"<path d='{d}' style='stroke: {SURFACE}' stroke-width='6' fill='none'/>"
        f"<path d='{d}' stroke='{color}' stroke-width='{width}' fill='none'{da} "
        f"marker-end='url(#slot-arrow-{color[1:]})'/>"
    )


def curve(x0, y0, x1, y1, tail=8):
    """An S-curve from (x0, y0) down to (x1, y1), leaving and arriving vertically, its last `tail` straight so that an
    arrowhead sits square on it."""
    ym = (y0 + y1 - tail) / 2
    return f"M{x0:.1f},{y0:.1f} C{x0:.1f},{ym:.1f} {x1:.1f},{ym:.1f} {x1:.1f},{y1 - tail:.1f} V{y1:.1f}"


def on_line(x, y, lines, color, size=12, lh=15):
    """A label sitting on a line, centred at (x, y), on a patch of the background that breaks the line."""
    w = max(len(s) for s in lines) * size * 0.58 + 8
    h = lh * len(lines) + 4
    return rect(
        x - w / 2, y - h / 2, w, h, "none", extra=f"style='fill: {SURFACE}'"
    ) + "".join(
        text(
            x,
            y - lh * (len(lines) - 1) / 2 + lh * k + size * 0.36,
            s,
            size,
            "middle",
            color,
            "bold",
        )
        for k, s in enumerate(lines)
    )


def mid(k):
    a, b = SEG[k]
    return (a + b) / 2


def tokens(y=14, h=32, size=16, gap=6) -> str:
    """The 10 tokens -> the next token: the ten filling the width of the hash under them, each as wide as its text and
    the same padding, and the next token over its block, so that the arrows into the bar go straight down."""
    o = []
    (x0, xe), nx = (SEG["part"][0], SEG["rest"][1]), mid("next")
    hx = (x0 + xe) / 2
    tw = [sum(ADVANCE[c] for c in t) * size / 1000 for t in CONTEXT]
    pad = (xe - x0 - gap * (len(tw) - 1) - sum(tw)) / len(tw)
    x = x0
    for token, t in zip(CONTEXT, tw):
        o.append(rect(x, y, t + pad, h, "#f2f2f2", "#9a9a9a", 4))
        o.append(text(x + (t + pad) / 2, y + h / 2 + size * 0.36, token, size, "middle", INK))
        x += t + pad + gap
    nw = 105  # "seconds" in Helvetica Bold 16 (65) and 20 on each side
    o.append(arrow(f"M{xe + 8:.1f},{y + h / 2:.1f} H{nx - nw / 2 - 6:.1f}", GREY, 1.3))
    o.append(rect(nx - nw / 2, y, nw, h, LNEXT, NEXT, 4))
    o.append(text(nx, y + h / 2 + size * 0.36, NEXT_TOKEN, size, "middle", NEXT, "bold"))
    o.append(brace(x0, xe, y + h + 7, GREY, h=4))  # all ten are hashed
    o.append(brace(SEG["part"][0], SEG["rest"][1], Y - 12, INK, up=True, h=5))
    o.append(arrow(f"M{hx:.1f},{y + h + 13} V{Y - 19}", GREY, 1.3))
    o.append(on_line(hx, (y + h + 13 + Y - 19) / 2, ["n-gram hash"], INK, 13))
    o.append(arrow(f"M{nx:.1f},{y + h + 2} V{Y - 3}", GREY, 1.3))
    return "".join(o)


# each piece of the bar's size, under its label with `sizes`
BITS = {"part": "(14 bits, not stored)", "b1": "(14 bits)", "b2": "(14 bits)", "rest": "(20 bits)", "next": "(16 bits)"}


def bar(sizes=False) -> str:
    """The five pieces of the hash and the next token, the partition hatched; with `sizes`, each one's size in bits
    under its label (the grey's alone in its middle), and the entry's, 64 stored bits, by the green arrow."""
    o = []
    for k, stroke, fill, label in (
        ("part", PART, LPART, "partition"),
        ("b1", BUCK, LBUCK, "bucket 1"),
        ("b2", BUCK, LBUCK, "bucket 2"),
        ("rest", REST, LREST, ""),
        ("next", NEXT, LNEXT, "next token"),
    ):
        a, b = SEG[k]
        o.append(rect(a, Y, b - a, H, fill, stroke, 0, "stroke-width='1.4'"))
        if k == "part":
            o.append(rect(a, Y, b - a, H, "url(#slot-hatch)"))
        if label:
            o.append(text(mid(k), Y + H / 2 + (0 if sizes else 5), label, 14, "middle", stroke, "bold"))
        if sizes:
            color, y = (MUTED, Y + H / 2 + 4) if k == "rest" else (stroke, Y + H / 2 + 15)
            o.append(text(mid(k), y, BITS[k], 11, "middle", color, extra="font-style='italic'"))
    # what we store: everything but the partition, the bracket's stub over its box
    sx = equation_line()[1][2]
    o.append(brace(SEG["b1"][0], SEG["next"][1], Y + H + 14, STORE, tick=sx))
    if sizes:
        o.append(text(sx + 12, Y + H + 34, "64 stored bits", 13, "start", STORE, "bold"))
    return "".join(o)


def box(x, s, color, fill, w=140, y=EQ_Y, h=34, size=20, dy=24):
    """A box in the hints[...] line, centred at x, h high, around s (its baseline dy under the top)."""
    return rect(x - w / 2, y, w, h, fill, color, 5, "stroke-width='1.5'") + text(
        x, y + dy, s, size, "middle", color, "bold"
    )


# Helvetica's advance widths, in thousandths of the font size, of the characters of the hints[...] lines' code and
# of the tokens
ADVANCE = {
    **dict.fromkeys("abdeghnopqu_0123456789", 556),
    **dict.fromkeys("cksvxyz", 500),
    **dict.fromkeys("ijl", 222),
    **dict.fromkeys("ft.[],I", 278),
    **dict.fromkeys("r()-", 333),
    **dict.fromkeys("Nw", 722),
    **dict.fromkeys("PS", 667),
    "G": 778,
    "T": 611,
    "m": 833,
}


def code_line(
    pieces, y, x, at=None, size=22, gap=5, anchor="middle", boxes=(34, 20, 24), snug=False
):
    """A line of code and boxes, left to right: a piece is a string or box()'s (s, color, fill, w). The line is
    centred at x, or its box `at` (an index into pieces) is; or it starts (anchor "start") or ends ("end") at x.
    `boxes` is box()'s h, size and dy, the code's baseline too. With `snug`, each string gets a fifth more room than
    Helvetica needs and keeps to the boxes beside it (a wider font's string then grows away from them): the first ends
    at the first box, the last starts at the last box, those between are centred. Returns its svg and its boxes'
    centres."""
    widths = [
        sum(ADVANCE[c] for c in p) * size / 1000 * (1.2 if snug else 1) if isinstance(p, str) else p[3]
        for p in pieces
    ]
    left = [sum(widths[:k]) + gap * k for k in range(len(pieces))]
    x0 = (
        x - (left[-1] + widths[-1]) / 2 if at is None else x - left[at] - widths[at] / 2
    )
    if anchor != "middle":
        x0 = x if anchor == "start" else x - left[-1] - widths[-1]
    h, box_size, dy = boxes
    o, centres = [], []
    for k, (p, a, w) in enumerate(zip(pieces, left, widths)):
        if isinstance(p, str) and snug and k < len(pieces) - 1:
            end = k + 1 < len(pieces) and k == 0
            o.append(text(x0 + a + (w if end else w / 2), y + dy, p, size, "end" if end else "middle"))
        elif isinstance(p, str):
            o.append(text(x0 + a, y + dy, p, size, "start"))
        else:
            centres.append(x0 + a + w / 2)
            o.append(box(centres[-1], p[0], p[1], p[2], w, y, h, box_size, dy))
    return "".join(o), centres


def equation_line() -> tuple[str, list[float]]:
    """hints[partition][bucket].append(checksum + next token): its svg and its boxes' centres. The entry's box is
    only outlined, and empty: equation() writes its words, each in its own colour."""
    return code_line(
        [
            "hints[",
            ("partition", PART, LPART, 140),
            "][",
            ("bucket", BUCK, LBUCK, 140),
            "].append(",
            ("", STORE, "none", ENTRY_W),
            ")",
        ],
        EQ_Y,
        EQ_BUCKET,
        at=3,
    )


def equation() -> str:
    """hints[partition][bucket].append(checksum + next token), and an arrow from each piece of the bar to its box."""
    line_svg, (px, bx, sx) = equation_line()
    o = [line_svg]
    y0, y1 = Y + H + 2, EQ_Y - 3
    o.append(arrow(curve(mid("part"), y0, px, y1), PART))
    # bucket 1's and bucket 2's, dashed, merging 2/5 of the way down into one arrow, the label on it: the bands
    # first, so that neither arm breaks the other
    ym = y0 + 0.4 * (y1 - y0)
    yc = (y0 + ym) / 2
    arms = [
        f"M{x:.1f},{y0} C{x:.1f},{yc:.1f} {bx:.1f},{yc:.1f} {bx:.1f},{ym:.1f}"
        for x in (mid("b1"), mid("b2"))
    ]
    o += [f"<path d='{d}' style='stroke: {SURFACE}' stroke-width='6' fill='none'/>" for d in arms]
    o += [
        f"<path d='{d}' stroke='{BUCK}' stroke-width='1.6' fill='none' stroke-dasharray='5 4'/>"
        for d in arms
    ]
    o.append(arrow(f"M{bx:.1f},{ym:.1f} V{y1}", BUCK, dash=True))
    o.append(on_line(bx, (ym + y1 - 8) / 2 + 2, ["use the", "emptiest bucket"], BUCK))
    o.append(arrow(f"M{sx:.1f},{Y + H + 19} V{y1}", STORE, 1.8))
    # the entry's words: checksum in ink, the next token orange as in the bar
    words = (
        f"<tspan fill='{INK}'>checksum</tspan><tspan fill='{MUTED}'> + </tspan>"
        f"<tspan fill='{NEXT}'>next token</tspan>"
    )
    o.append(
        f"<text x='{sx:.1f}' y='{EQ_Y + 17 + 16 * 0.36:.1f}' font-size='16' text-anchor='middle' "
        f"font-weight='bold' {FONT}>{words}</text>"
    )
    return "".join(o)


def slot_index_svg(sizes=False, phone=False) -> str:
    """The figure: the 10-gram and its next token, the bar they make, and where in hints its entry goes; with `sizes`,
    each piece's size in bits; with `phone`, laid out for a phone's width."""
    if phone:
        return (
            f"<svg viewBox='0 0 {PHONE_W} {P_EQ_Y + 2 * P_EQ_H + P_EQ_GAP + 16}' xmlns='http://www.w3.org/2000/svg' "
            f"role='img' aria-label='Where a 10-gram is stored in the SlotIndex'>{defs()}"
            f"{phone_tokens(p_mid('next'))}{phone_hash()}{phone_bar(sizes)}{phone_equation(sizes)}</svg>"
        )
    return (
        f"<svg viewBox='0 0 {W} {EQ_Y + 50}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='Where a 10-gram is stored in the SlotIndex'>{defs()}{tokens()}{bar(sizes)}{equation()}</svg>"
    )


def slot_index_figure() -> str:
    """The figure without sizes, and with them in its place while the mouse is over it (or after a tap): .svg-swap in
    report.CSS."""
    return (
        "<div class='svg-swap' onclick=\"this.classList.toggle('on')\">"
        f"<div class='base'>{slot_index_svg()}</div><div class='alt' aria-hidden='true'>{slot_index_svg(True)}</div>"
        "</div>"
    )


# simple_index_svg's layout: the centres of the 10 tokens, the next token and the hints[...] line, so that the
# drawing is centred; the top of the line's boxes
SIMPLE = {"tokens": 400, "noise": 770, "line": 520}
SIMPLE_Y = 130


def simple_index_svg(phone=False) -> str:
    """The 10-gram and its next token, and hints[hash].append(next_token): an arrow from the 10 tokens to hash and one
    from the next token to next_token; with `phone`, laid out for a phone's width."""
    if phone:
        return phone_simple_index_svg()
    o, y, h, gap = [], 20, 26, 5
    ws = [max(40, len(t) * 7.2 + 18) for t in CONTEXT]
    x = x0 = SIMPLE["tokens"] - (sum(ws) + gap * (len(ws) - 1)) / 2
    for token, w in zip(CONTEXT, ws):
        o.append(rect(x, y, w, h, "#f2f2f2", "#9a9a9a", 4))
        o.append(text(x + w / 2, y + h / 2 + 4.3, token, 12, "middle", INK))
        x += w + gap
    xe, nx, nw = x - gap, SIMPLE["noise"], 72
    o.append(arrow(f"M{xe + 8:.1f},{y + h / 2:.1f} H{nx - nw / 2 - 6:.1f}", GREY, 1.3))
    o.append(rect(nx - nw / 2, y, nw, h, LNEXT, NEXT, 4))
    o.append(text(nx, y + h / 2 + 4.3, NEXT_TOKEN, 12, "middle", NEXT, "bold"))
    o.append(brace(x0, xe, y + h + 7, GREY, h=4))  # all ten are hashed
    line_svg, (hb, nb) = code_line(
        [
            "hints[",
            ("hash", BUCK, LBUCK, 96),
            "].append(",
            ("next_token", NEXT, LNEXT, 160),
            ")",
        ],
        SIMPLE_Y,
        SIMPLE["line"],
    )
    o.append(line_svg)
    o.append(arrow(curve(SIMPLE["tokens"], y + h + 12, hb, SIMPLE_Y - 3), BUCK))
    o.append(arrow(curve(nx, y + h + 2, nb, SIMPLE_Y - 3), NEXT))
    return (
        f"<svg viewBox='0 0 {W} {SIMPLE_Y + 50}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='How the Python version stores a 10-gram'>{defs()}{''.join(o)}</svg>"
    )


# The phone layouts, drawn about 1:1 in a phone's CSS pixels. Only the context's last six tokens, in a row as wide as
# P_ROWS_W, from x = P_X (the ten would need two rows); the next token is beside them.
PHONE_W, P_X = 350, 10
P_ROWS = (CONTEXT[4:],)  # the rows' tokens
P_ROWS_W, P_TOK_Y, P_TOK_H, P_TOK_GAP, P_TOK_SIZE = 212, 10, 28, 6, 13
P_NEXT_W = 68  # the next token's box: "seconds" in Helvetica Bold 13.5 (55) and room for a wider font
P_BOTTOM = P_TOK_Y + len(P_ROWS) * (P_TOK_H + P_TOK_GAP) - P_TOK_GAP  # the rows' bottom
# the bar's pieces, each as wide as its label (in bold 13, the next token's on two lines) with room for a wider font,
# rather than as its bits; its top and height
P_SEG = {
    "part": (P_X, 80),
    "b1": (80, 150),
    "b2": (150, 220),
    "rest": (220, 274),
    "next": (280, 338),
}
P_Y, P_H = P_BOTTOM + 74, 52
# the hints[...] lines: the first's top, the boxes' height, the space between the lines; the code's size, and its
# boxes' words'
P_EQ_Y, P_EQ_H, P_EQ_GAP = P_Y + 154, 30, 12
P_CODE, P_WORDS = 16, 15
P_BOXES = (P_EQ_H, P_WORDS, P_EQ_H / 2 + 0.36 * P_WORDS)
# the boxes' widths: their words, in Helvetica Bold (partition 60, bucket 48, checksum + next token in 13 143), and
# room for a wider font
P_PART_W, P_BUCKET_W, P_ENTRY_W = 84, 68, 178


def phone_tokens(next_x) -> str:
    """tokens() for phones: P_ROWS, each row's tokens with the same padding, and the next token, centred at next_x,
    beside them; under the rows, the bracket of the hashed tokens."""
    o, size = [], P_TOK_SIZE
    for r, row in enumerate(P_ROWS):
        y = P_TOK_Y + r * (P_TOK_H + P_TOK_GAP)
        tw = [sum(ADVANCE[c] for c in t) * size / 1000 for t in row]
        pad = (P_ROWS_W - 5 * (len(tw) - 1) - sum(tw)) / len(tw)
        x = P_X
        for token, t in zip(row, tw):
            o.append(rect(x, y, t + pad, P_TOK_H, "#f2f2f2", "#9a9a9a", 4))
            o.append(text(x + (t + pad) / 2, y + P_TOK_H / 2 + size * 0.36, token, size, "middle", INK))
            x += t + pad + 5
    xe, ym = P_X + P_ROWS_W, (P_TOK_Y + P_BOTTOM) / 2
    o.append(arrow(f"M{xe + 8:.1f},{ym:.1f} H{next_x - P_NEXT_W / 2 - 6:.1f}", GREY, 1.3))
    o.append(rect(next_x - P_NEXT_W / 2, ym - P_TOK_H / 2, P_NEXT_W, P_TOK_H, LNEXT, NEXT, 4))
    o.append(text(next_x, ym + 13.5 * 0.36, NEXT_TOKEN, 13.5, "middle", NEXT, "bold"))
    o.append(brace(P_X, xe, P_BOTTOM + 7, GREY, h=4))  # all of them are hashed
    return "".join(o)


def p_mid(k):
    a, b = P_SEG[k]
    return (a + b) / 2


def phone_hash() -> str:
    """The tokens' bracket down to the hash's, n-gram hash on the arrow; the next token's arrow down to its piece."""
    hx, nx = P_X + P_ROWS_W / 2, p_mid("next")
    y0, y1 = P_BOTTOM + 13, P_Y - 19
    return (
        brace(P_SEG["part"][0], P_SEG["rest"][1], P_Y - 12, INK, up=True, h=5, tick=hx)
        + arrow(f"M{hx:.1f},{y0} V{y1}", GREY, 1.3)
        + on_line(hx, (y0 + y1) / 2, ["n-gram hash"], INK, 12.5)
        + arrow(f"M{nx:.1f},{(P_TOK_Y + P_BOTTOM + P_TOK_H) / 2 + 2:.1f} V{P_Y - 3}", GREY, 1.3)
    )


def phone_bar(sizes=False) -> str:
    """bar() for phones: the next token's label, and the partition's size, on two lines; each piece's lines centred
    in it."""
    o = []
    for k, stroke, fill, label in (
        ("part", PART, LPART, ["partition"]),
        ("b1", BUCK, LBUCK, ["bucket 1"]),
        ("b2", BUCK, LBUCK, ["bucket 2"]),
        ("rest", REST, LREST, []),
        ("next", NEXT, LNEXT, ["next", "token"]),
    ):
        a, b = P_SEG[k]
        o.append(rect(a, P_Y, b - a, P_H, fill, stroke, 0, "stroke-width='1.4'"))
        if k == "part":
            o.append(rect(a, P_Y, b - a, P_H, "url(#slot-hatch)"))
        bits = BITS[k].replace(", ", ",|").split("|") if sizes else []
        n = len(label) + len(bits)
        for j, s in enumerate(label + bits):
            y = P_Y + P_H / 2 + 4.5 + 14 * (j - (n - 1) / 2)
            if j < len(label):
                o.append(text(p_mid(k), y, s, 13, "middle", stroke, "bold"))
            else:
                color = MUTED if k == "rest" else stroke
                o.append(text(p_mid(k), y - 0.5, s, 11.5, "middle", color, extra="font-style='italic'"))
    sx = phone_equation_lines()[2]
    o.append(brace(P_SEG["b1"][0], P_SEG["next"][1], P_Y + P_H + 12, STORE, tick=sx))
    return "".join(o)


def phone_equation_lines() -> tuple[str, list[float], float]:
    """The hints[...] line on two lines, hints[partition][bucket] from the left and .append(checksum + next token) to
    the right under it: their svg, the centres of the partition's and bucket's boxes, and where the green arrow comes
    into the entry's box, its centre if that is clear of the first line, else right of the first line."""
    first, (px, bx) = code_line(
        ["hints[", ("partition", PART, LPART, P_PART_W), "][", ("bucket", BUCK, LBUCK, P_BUCKET_W), "]"],
        P_EQ_Y,
        P_X,
        size=P_CODE,
        anchor="start",
        boxes=P_BOXES,
        snug=True,
    )
    end = bx + P_BUCKET_W / 2 + 5 + 1.2 * ADVANCE["]"] * P_CODE / 1000  # the first line's right end
    second, (sx,) = phone_append_line()
    return first + second, [px, bx], max(sx, end + 24)


def phone_append_line():
    """.append(checksum + next token): the box empty (phone_equation writes its words), ending at the bar's end."""
    return code_line(
        [".append(", ("", STORE, "none", P_ENTRY_W), ")"],
        P_EQ_Y + P_EQ_H + P_EQ_GAP,
        P_SEG["next"][1],
        size=P_CODE,
        anchor="end",
        boxes=P_BOXES,
        snug=True,
    )


def phone_equation(sizes=False) -> str:
    """equation() for phones: the entry's words, in 13, in its box on the second line; with `sizes`, the entry's 64
    stored bits on the green arrow."""
    lines, (px, bx), sx = phone_equation_lines()
    o = [lines]
    y0, y1 = P_Y + P_H + 2, P_EQ_Y - 3
    o.append(arrow(curve(p_mid("part"), y0, px, y1), PART))
    ym = y0 + 0.45 * (y1 - y0)
    yc = (y0 + ym) / 2
    arms = [
        f"M{x:.1f},{y0} C{x:.1f},{yc:.1f} {bx:.1f},{yc:.1f} {bx:.1f},{ym:.1f}"
        for x in (p_mid("b1"), p_mid("b2"))
    ]
    o += [f"<path d='{d}' style='stroke: {SURFACE}' stroke-width='6' fill='none'/>" for d in arms]
    o += [
        f"<path d='{d}' stroke='{BUCK}' stroke-width='1.6' fill='none' stroke-dasharray='5 4'/>"
        for d in arms
    ]
    o.append(arrow(f"M{bx:.1f},{ym:.1f} V{y1}", BUCK, dash=True))
    o.append(on_line(bx, (ym + y1 - 8) / 2 + 2, ["use the", "emptiest bucket"], BUCK))
    y2 = P_EQ_Y + P_EQ_H + P_EQ_GAP  # the second line's top
    o.append(arrow(f"M{sx:.1f},{P_Y + P_H + 17} V{y2 - 3}", STORE, 1.8))
    if sizes:
        o.append(on_line(sx, P_Y + P_H + 40, ["64 stored bits"], STORE, 12.5))
    words = (
        f"<tspan fill='{INK}'>checksum</tspan><tspan fill='{MUTED}'> + </tspan>"
        f"<tspan fill='{NEXT}'>next token</tspan>"
    )
    cx = phone_append_line()[1][0]
    o.append(
        f"<text x='{cx:.1f}' y='{y2 + P_EQ_H / 2 + 13 * 0.36:.1f}' font-size='13' text-anchor='middle' "
        f"font-weight='bold' {FONT}>{words}</text>"
    )
    return "".join(o)


def phone_simple_index_svg() -> str:
    """simple_index_svg for phones: the last six tokens and the next token beside them, and the hints[...]
    line in smaller type."""
    nx, eq = 300, P_BOTTOM + 68
    line_svg, (hb, nb) = code_line(
        ["hints[", ("hash", BUCK, LBUCK, 54), "].append(", ("next_token", NEXT, LNEXT, 104), ")"],
        eq,
        (P_X + P_SEG["next"][1]) / 2,
        size=P_CODE,
        boxes=P_BOXES,
        snug=True,
    )
    o = [
        phone_tokens(nx),
        line_svg,
        arrow(curve(P_X + P_ROWS_W / 2, P_BOTTOM + 12, hb, eq - 3), BUCK),
        arrow(curve(nx, (P_TOK_Y + P_BOTTOM + P_TOK_H) / 2 + 2, nb, eq - 3), NEXT),
    ]
    return (
        f"<svg viewBox='0 0 {PHONE_W} {eq + 50}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='How the Python version stores a 10-gram'>{defs()}{''.join(o)}</svg>"
    )
