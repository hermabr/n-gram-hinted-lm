"""A static figure of where the model takes in its hints: the stack (embedding, blocks 0-6, blocks 7-10, output head)
with the hint added at three sites, each scaled by its alpha, and the hint itself, sigma[l, c, p] * h_t + beta[l, c, p],
with its parts named.

The math is set by MathJax in New Computer Modern (hints_math.mjs, run once) and drawn from its paths, so it looks the
same in every browser and the leader lines start from where MathJax put each symbol. Only the first l, c, p (sigma's)
are named; the leaders are elbows: a stem out of each symbol, the outer ones bending into a short run to the label.

On phones (`phone`), the formula goes under the stack, the hint's line on the stack's left coming down to it, and all
its labels are under it, on a staircase of elbows, each running right to its label."""

import json
from pathlib import Path

from retrieval_blog_post.slot_index import SURFACE, line, rect, text

INK, MUTED = "#1d1d1f", "#6b6b70"
HINT, HINT_DARK = "#e8772e", "#b4541a"  # the hint's lines, its formula
BLOCK = ("#dcebf7", "#5b8dc9")  # fill, stroke
EMBED = ("#eee6f7", "#8f6fc0")

W, H = 960, 550
CX, R, BUS = (
    170,
    13,
    414,
)  # the stack's centre, the plus signs' radius, x of the hint's vertical line
SITES = (("in", 462), ("mid", 327), ("out", 195))  # where the hint is added

FX, FY, FS = 516, 357, 34  # the formula's left edge, baseline, size
ALPHA_FS = 23  # the alphas' size
# hints_math.mjs's output: per piece of math, its paths (y up, 1000 units to the em), width, height, depth and the x
# centres of its named symbols
MATH = json.loads(Path(__file__).with_name("hints_math.json").read_text())
TOP, BOTTOM = 327, 372  # where the leaders above and below the formula start
LS = 16.5  # the labels' size


def arrow_defs() -> str:
    return (
        "<defs>"
        + "".join(
            f"<marker id='hints-arrow-{c[1:]}' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='{w}' "
            f"markerHeight='{w}' markerUnits='userSpaceOnUse' orient='auto'><path d='M0,1 L10,5 L0,9 z' "
            f"fill='{c}'/></marker>"
            for c, w in ((MUTED, 10), (HINT, 11))
        )
        + "</defs>"
    )


def arrow(x0, y0, x1, y1, color=MUTED, width=2, head=True):
    return line(
        x0,
        y0,
        x1,
        y1,
        color,
        width,
        f"marker-end='url(#hints-arrow-{color[1:]})'" if head else "",
    )


def box(x0, y0, x1, y1, style, label, sub=None, size=16.5, rx=11, cx=CX):
    o = [rect(x0, y0, x1 - x0, y1 - y0, style[0], style[1], rx, "stroke-width='1.8'")]
    cy = (y0 + y1) / 2
    if sub:
        o.append(text(cx, cy - 4, label, size, "middle", INK))
        o.append(text(cx, cy + 21, sub, 13.8, "middle", MUTED))
    else:
        o.append(text(cx, cy + size * 0.35, label, size, "middle", INK))
    return "".join(o)


def plus(y, cx=CX):
    a = R * 0.6
    return (
        f"<circle cx='{cx}' cy='{y}' r='{R}' style='fill: {SURFACE}' stroke='{HINT}' stroke-width='2.4'/>"
        + line(cx - a, y, cx + a, y, HINT, 2.4)
        + line(cx, y - a, cx, y + a, HINT, 2.4)
    )


def stack() -> str:
    """The left side: embedding, blocks 0-6, blocks 7-10 and head, the hint added before each of the last three, and
    the hint's vertical line feeding the three sites."""
    o = [
        box(54, 492, 282, 537, EMBED, "Embedding"),
        arrow(CX, 492, CX, 462 + R, head=False),
        arrow(CX, 462 - R, CX, 426),
        box(30, 354, 306, 426, BLOCK, "Transformer blocks", "layers 0–6", rx=12),
        arrow(CX, 354, CX, 327 + R, head=False),
        arrow(CX, 327 - R, CX, 294),
        box(30, 222, 306, 294, BLOCK, "Transformer blocks", "layers 7–10", rx=12),
        arrow(CX, 222, CX, 195 + R, head=False),
        arrow(CX, 195 - R, CX, 165),
        box(54, 120, 282, 165, EMBED, "Output head"),
        arrow(CX, 120, CX, 96),
        text(CX, 86, "logits", 15.4, "middle", MUTED),
        line(BUS, 462, BUS, 195, HINT, 2.9),
    ]
    for name, y in SITES:
        o.append(plus(y))
        o.append(arrow(BUS, y, CX + R + 1, y, HINT, 2.6))
        a = MATH[f"alpha_{name}"]
        o.append(math(a, 310 - a["width"] * ALPHA_FS / 2000, y - 11, ALPHA_FS))
    return "".join(o)


def phone_stack() -> str:
    """stack() for phones: the boxes narrower and lower, and the hint's line on their left (the alphas between it and
    the blocks), going on down past the embedding to the formula under them."""
    c, (_, y_in), (_, y_mid), (_, y_out) = PCX, *PSITES
    o = [
        box(c - 96, 443, c + 96, 485, EMBED, "Embedding", cx=c),
        arrow(c, 443, c, y_in + R, head=False),
        arrow(c, y_in - R, c, 389),
        box(
            c - 112,
            325,
            c + 112,
            389,
            BLOCK,
            "Transformer blocks",
            "layers 0–6",
            rx=12,
            cx=c,
        ),
        arrow(c, 325, c, y_mid + R, head=False),
        arrow(c, y_mid - R, c, 271),
        box(
            c - 112,
            207,
            c + 112,
            271,
            BLOCK,
            "Transformer blocks",
            "layers 7–10",
            rx=12,
            cx=c,
        ),
        arrow(c, 207, c, y_out + R, head=False),
        arrow(c, y_out - R, c, 150),
        box(c - 96, 108, c + 96, 150, EMBED, "Output head", cx=c),
        arrow(c, 108, c, 88),
        text(c, 78, "logits", 15.4, "middle", MUTED),
        line(PBUS, PFY - 9, PBUS, y_out, HINT, 2.9),
    ]
    for name, y in PSITES:
        o.append(plus(y, c))
        o.append(arrow(PBUS, y, c - R - 1, y, HINT, 2.6))
        a = MATH[f"alpha_{name}"]
        o.append(
            math(
                a,
                (PBUS + c - 112) / 2 + 5 - a["width"] * ALPHA_FS / 2000,
                y - 11,
                ALPHA_FS,
            )
        )
    return "".join(o)


def math(m: dict, x: float, y: float, size: float) -> str:
    """A piece of hints_math.json's math, its left edge at x, its baseline at y."""
    k = size / 1000
    return f"<g transform='translate({x:.1f} {y:.1f}) scale({k:g} {-k:g})' fill='{HINT_DARK}'>{m['svg']}</g>"


def formula(fx=FX, fy=FY, fs=FS) -> tuple[str, dict[str, float]]:
    """The formula, its left edge at fx, its baseline at fy, and the x centre of each named symbol."""
    f = MATH["formula"]
    return math(f, fx, fy, fs), {
        name: fx + x * fs / 1000 for name, x in f["marks"].items()
    }


def label(x, y, s, anchor="middle", size=LS):
    return text(x, y, s, size, anchor, INK)


def leaders(xs: dict[str, float]) -> str:
    """Above: trust amount and hint embedding straight up, trust embedding bending right. Below: count straight down,
    n-gram length bending left and purity right."""
    up, down, run = 267, 447, 21  # the elbows' heights, their runs' length
    b = MUTED
    return "".join(
        [
            line(xs["sigma"], TOP, xs["sigma"], up - 21, b, 1.3),
            label(xs["sigma"], up - 28, "trust amount"),
            line(xs["h"], TOP, xs["h"], up - 21, b, 1.3),
            label(xs["h"], up - 28, "hint embedding"),
            f"<path d='M{xs['beta']:.1f},{TOP} V{up} h{run}' stroke='{b}' stroke-width='1.3' fill='none'/>",
            label(xs["beta"] + run + 6, up + LS * 0.35, "trust embedding", "start"),
            f"<path d='M{xs['l']:.1f},{BOTTOM} V{down} h{-run}' stroke='{b}' stroke-width='1.3' fill='none'/>",
            label(xs["l"] - run - 6, down - 4, "n-gram length", "end"),
            label(xs["l"] - run - 6, down + 17, "{6, 10, 20}", "end"),
            line(xs["c"], BOTTOM, xs["c"], down + 21, b, 1.3),
            label(xs["c"], down + 21 + LS + 2, "count"),
            f"<path d='M{xs['p']:.1f},{BOTTOM} V{down} h{run}' stroke='{b}' stroke-width='1.3' fill='none'/>",
            label(xs["p"] + run + 6, down + LS * 0.35, "purity", "start"),
        ]
    )


# The phone layout: the stack at x = PCX, the hint added at PSITES, the hint's line at x = PBUS, and the formula
# under the stack (its left edge, baseline and size), its labels at size PLS, PSTEP apart
PW, PCX, PBUS = 360, 200, 30
PSITES = (("in", 416), ("mid", 298), ("out", 180))
PFX, PFY, PFS = 46, 548, 25
PLS, PSTEP = 15, 23


def phone_leaders(xs: dict[str, float]) -> str:
    """leaders() for phones: every label under the formula, on a staircase of elbows, each stem down to its step and
    a short run right to its label. The rightmost symbol's step is the top one, so that no run crosses a stem: trust
    embedding, hint embedding, purity, count, n-gram length, trust amount."""
    bottom, run = (
        PFY + 14,
        12,
    )  # where the stems start, under the formula; the runs' length
    o = []
    for step, (name, s) in enumerate(
        (
            ("beta", "trust embedding"),
            ("h", "hint embedding"),
            ("p", "purity"),
            ("c", "count"),
            ("l", "n-gram length {6, 10, 20}"),
            ("sigma", "trust amount"),
        )
    ):
        y = bottom + 12 + PSTEP * step
        o.append(
            f"<path d='M{xs[name]:.1f},{bottom} V{y} h{run}' stroke='{MUTED}' stroke-width='1.3' fill='none'/>"
        )
        o.append(label(xs[name] + run + 6, y + PLS * 0.35, s, "start", PLS))
    return "".join(o)


def hints_svg(phone=False) -> str:
    """The figure; with `phone`, laid out for a phone's width."""
    if phone:
        f, xs = formula(PFX, PFY, PFS)
        feed = line(PBUS - 1.45, PFY - 9, PFX - 8, PFY - 9, HINT, 2.9)
        w, h, parts = (
            PW,
            PFY + 26 + 6 * PSTEP,
            f"{phone_stack()}{feed}{f}{phone_leaders(xs)}",
        )
    else:
        f, xs = formula()
        feed = line(BUS, FY - 10, FX - 9, FY - 10, HINT, 2.9)
        w, h, parts = W, H, f"{stack()}{feed}{f}{leaders(xs)}"
    return (
        f"<svg viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='Where the model adds its hints, and what a hint is made of'>"
        f"{arrow_defs()}{parts}</svg>"
    )
