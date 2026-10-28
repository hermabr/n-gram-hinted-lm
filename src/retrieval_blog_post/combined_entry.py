"""A static figure of one entry of the SlotIndex that keeps the 6-, 10- and 20-gram in one table, drawn like the
10-gram-only figure: one bar of named cells.

The entry (exact_match/src/slot.rs): the 6-gram's 44-bit key, its first 14 bits the partition (not stored) and the
other 30 the check; a 12-bit fingerprint of the 10-gram; a 6-bit fingerprint of the 20-gram; the 16-bit next token.
A slot holds the last 64. slot.rs packs them check | 20-gram | 10-gram | next token; the drawing puts the n-grams in
order. The cells' widths are proportional to their bits, but for the 20-gram's, widened to fit its name; under the
bar, a bracket under the partition and one under the 64 stored bits.

On phones (`phone`), the cells are only as wide as their names (the 6-gram's the rest of the width), the next token's
on two lines, and the partition's name, under its narrow bracket, too."""

from retrieval_blog_post.slot_index import (
    BUCK,
    LBUCK,
    LNEXT,
    LPART,
    NEXT,
    PART,
    STORE,
    brace,
    line,
    rect,
    text,
)

W = 960
TEN, LTEN = "#0f7f86", "#c6e6e7"  # the 10-gram's fingerprint
TWENTY, LTWENTY = "#b0365e", "#f3d2dd"  # the 20-gram's fingerprint
PBITS, KBITS, F10, F20, TBITS = 14, 30, 12, 6, 16
assert KBITS + F10 + F20 + TBITS == 64
BIT, MINW = 11, 80  # px per bit; a cell's least width, for the 20-gram's name
GAP = 8  # between the cells
Y, H = 8, 52  # the bar's top and height


def cells():
    """The x ranges of the 6-gram, 10-gram, 20-gram and next token cells, centred."""
    ws = [max(b * BIT, MINW) for b in (PBITS + KBITS, F10, F20, TBITS)]
    x = (W - sum(ws) - GAP * (len(ws) - 1)) / 2
    out = []
    for w in ws:
        out.append((x, x + w))
        x += w + GAP
    return out


def label(x, name, sub, color, size=14):
    """A cell's name, bold, and its bit count under it, centred at x."""
    return text(x, Y + H / 2 - 1, name, size, "middle", color, "bold") + text(
        x, Y + H / 2 + 15, sub, 11.5, "middle", color
    )


def phone_label(x, names, sub, color):
    """label() for phones: the name on one or two lines, and the lines centred in the cell."""
    lines = len(names) + 1
    y = Y + P_H / 2 + 4.5 - 7 * (lines - 1)  # the first line's baseline
    return "".join(
        text(x, y + 14 * k, s, 13, "middle", color, "bold") for k, s in enumerate(names)
    ) + text(x, y + 14 * len(names) - 0.5, sub, 11.5, "middle", color)


def bracket(a, b, s, color):
    """A bracket under [a, b], s under it."""
    yb = Y + H + 12
    return brace(a, b, yb, color, h=5) + text(
        (a + b) / 2, yb + 22, s, 13, "middle", color, "bold"
    )


# The phone layout: the cells' widths (the 6-gram, 10-gram, 20-gram and next token), from x = P_X; their height
P_X, P_WIDTHS, P_H = 10, (114, 68, 68, 62), 56


def phone_cells():
    """cells() for phones."""
    out, x = [], P_X
    for w in P_WIDTHS:
        out.append((x, x + w))
        x += w + GAP - 2
    return out


def combined_entry_svg(phone=False) -> str:
    """The bar: the 6-gram, its partition hatched as in the 10-gram-only figure, then the 10-gram, the 20-gram and the
    next token; under it, the partition and the 64 stored bits, bracketed; with `phone`, laid out for a phone's
    width."""
    if phone:
        return phone_combined_entry_svg()
    (g0, g1), (t0, t1), (w0, w1), (n0, n1) = cells()
    p1 = g0 + PBITS * BIT  # the partition's end
    hatch = (
        "<defs><pattern id='combined-hatch' width='7' height='7' patternUnits='userSpaceOnUse' "
        f"patternTransform='rotate(45)'><line x1='0' y1='0' x2='0' y2='7' stroke='{PART}' "
        "stroke-width='1.2' opacity='0.35'/></pattern></defs>"
    )
    o = [
        hatch,
        rect(g0, Y, g1 - g0, H, LBUCK),
        rect(g0, Y, p1 - g0, H, LPART),
        rect(g0, Y, p1 - g0, H, "url(#combined-hatch)"),
        line(p1, Y, p1, Y + H, BUCK, 1.2, "stroke-dasharray='4 3'"),
        rect(g0, Y, g1 - g0, H, "none", BUCK, 0, "stroke-width='1.4'"),
        label((g0 + g1) / 2, "6-gram", "(44 bits)", BUCK),
    ]
    for (a, b), name, sub, color, fill, size in (
        ((t0, t1), "10-gram", "(12 bits)", TEN, LTEN, 14),
        ((w0, w1), "20-gram", "(6 bits)", TWENTY, LTWENTY, 13.5),
        ((n0, n1), "next token", "(16 bits)", NEXT, LNEXT, 14),
    ):
        o.append(rect(a, Y, b - a, H, fill, color, 0, "stroke-width='1.4'"))
        o.append(label((a + b) / 2, name, sub, color, size))
    o.append(bracket(g0, p1 - 3, "partition (14 bits)", PART))
    o.append(bracket(p1 + 3, n1, "64 stored bits", STORE))
    return (
        f"<svg viewBox='0 0 {W} {Y + H + 44}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='One entry holding the 6-, 10- and 20-gram'>{''.join(o)}</svg>"
    )


def phone_combined_entry_svg() -> str:
    """combined_entry_svg for phones: the partition's bracket's label on two lines, starting where the bar does."""
    (g0, g1), (t0, t1), (w0, w1), (n0, n1) = phone_cells()
    p1 = g0 + (g1 - g0) * PBITS / (
        PBITS + KBITS
    )  # the partition's end, its share of the 6-gram's bits
    hatch = (
        "<defs><pattern id='combined-hatch' width='7' height='7' patternUnits='userSpaceOnUse' "
        f"patternTransform='rotate(45)'><line x1='0' y1='0' x2='0' y2='7' stroke='{PART}' "
        "stroke-width='1.2' opacity='0.35'/></pattern></defs>"
    )
    o = [
        hatch,
        rect(g0, Y, g1 - g0, P_H, LBUCK),
        rect(g0, Y, p1 - g0, P_H, LPART),
        rect(g0, Y, p1 - g0, P_H, "url(#combined-hatch)"),
        line(p1, Y, p1, Y + P_H, BUCK, 1.2, "stroke-dasharray='4 3'"),
        rect(g0, Y, g1 - g0, P_H, "none", BUCK, 0, "stroke-width='1.4'"),
        phone_label((p1 + g1) / 2, ["6-gram"], "(44 bits)", BUCK),
    ]
    for (a, b), names, sub, color, fill in (
        ((t0, t1), ["10-gram"], "(12 bits)", TEN, LTEN),
        ((w0, w1), ["20-gram"], "(6 bits)", TWENTY, LTWENTY),
        ((n0, n1), ["next", "token"], "(16 bits)", NEXT, LNEXT),
    ):
        o.append(rect(a, Y, b - a, P_H, fill, color, 0, "stroke-width='1.4'"))
        o.append(phone_label((a + b) / 2, names, sub, color))
    yb = Y + P_H + 12
    o.append(brace(g0, p1 - 3, yb, PART, h=5))
    o.append(text(g0, yb + 22, "partition", 13, "start", PART, "bold"))
    o.append(text(g0, yb + 38, "(14 bits)", 13, "start", PART, "bold"))
    o.append(brace(p1 + 3, n1, yb, STORE, h=5))
    o.append(
        text((p1 + 3 + n1) / 2, yb + 22, "64 stored bits", 13, "middle", STORE, "bold")
    )
    return (
        f"<svg viewBox='0 0 {P_X + sum(P_WIDTHS) + 3 * (GAP - 2) + 10} {yb + 50}' xmlns='http://www.w3.org/2000/svg' "
        f"role='img' aria-label='One entry holding the 6-, 10- and 20-gram'>{''.join(o)}</svg>"
    )
