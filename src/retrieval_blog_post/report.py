"""Report pages with interactive Vega-Lite charts (adapted from the author's temporal-lm project): a chart is a gzipped
Vega-Lite spec next to its page, loaded and rendered by /assets/report-charts.js (with /assets/vega-runtime.js), as
served by ./serve_html from the repo root."""

import gzip
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

import altair as alt

from retrieval_blog_post.plum import recolor

PURPLE, BLUE, GREEN, ORANGE = "#8b67c7", "#3987e5", "#1a9a1a", "#c98500"

TICK_LABEL_EXPRESSION = (
    "abs(datum.value) >= 1000 ? format(datum.value, '.3~s') : "
    "abs(datum.value) >= 10 ? format(datum.value, '.0f') : format(datum.value, '.3~f')"
)


LINE_WIDTH = 2.25  # px, of a line chart's lines and its legend's symbols


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    points: tuple[tuple[float, float], ...]
    color: str
    dashed: bool = False
    dotted: bool = False  # in place of dashed, e.g. a reference level
    dots: bool = True  # a point at each x; none for a level drawn across the chart


@dataclass(frozen=True, slots=True)
class Chart:
    x_label: str
    y_label: str
    series: tuple[Series, ...]
    log_x: bool = False
    x_ticks: tuple[float, ...] | None = None
    x_labels: dict[float, str] | None = (
        None  # tick and tooltip label per x, in place of the number
    )
    # the x ticks labelled, if not all: the others keep their tick and grid line, unlabelled
    x_tick_labels: tuple[float, ...] | None = None
    y_domain: tuple[float, float] | None = None
    x_prefix: str = ""  # of the x value in the tooltip, e.g. "n = "
    tooltip_decimals: int = 4
    value_suffix: str = ""
    # of the axis and legend text, in place of report-charts.js's 11 and 12
    font_size: float | None = None
    legend_columns: int | None = None  # in place of report-charts.js's 4
    # in place of report-charts.js's top, e.g. "bottom-left", inside the plot, where a falling line leaves room
    legend_orient: str | None = None
    # px between each axis's labels and its title, in place of report-charts.js's 14
    title_padding: float | None = None
    # px (at least) of the x axis beyond the data at each end, in place of the axis rounded out (to powers of 2, on a
    # log one)
    x_padding: float | None = None
    # a series on an axis of its own on the right, titled right_label, on the same scale as the left one's; each axis's
    # title is in its line's color (the left one's if the chart has one series)
    right: Series | None = None
    right_label: str = ""


def line_chart(chart: Chart, height: int = 278) -> alt.TopLevelMixin:
    """An interactive multi-series line chart: hovering shows every series' value at the nearest x."""
    right = chart.right
    every = chart.series + ((right,) if right else ())
    values = [
        {
            "x": x,
            "y": y,
            "series": series.label,
        }
        for series in every
        for x, y in series.points
    ]
    labels = [series.label for series in every]
    colors = [series.color for series in every]
    nearest = alt.selection_point(
        name="nearest_x", fields=["x"], nearest=True, on="pointermove", empty=False
    )
    y_scale = (
        alt.Scale(zero=False, domain=list(chart.y_domain))
        if chart.y_domain
        else alt.Scale(zero=False)
    )
    fonts = (
        {"labelFontSize": chart.font_size, "titleFontSize": chart.font_size}
        if chart.font_size
        else {}
    )
    if chart.title_padding is not None:
        fonts["titlePadding"] = chart.title_padding
    # labelLimit 0: the longer labels would be cut short at the default 160 px
    legend_fonts = (
        {"labelFontSize": chart.font_size, "labelLimit": 0} if chart.font_size else {}
    )
    if chart.legend_columns:
        legend_fonts["columns"] = chart.legend_columns
    if chart.legend_orient:
        legend_fonts["orient"] = chart.legend_orient
    # none for a lone line, as in the post: the y axis says what it is; nor with a right axis, whose titles are in
    # their lines' colors (a legend of two long labels would run past a phone's card)
    legend = (
        # its symbols as wide as the lines, so that a dotted line's dots and a dashed one's dashes are the lines'
        alt.Legend(symbolType="stroke", symbolStrokeWidth=LINE_WIDTH, **legend_fonts)
        if len(chart.series) > 1 and not right
        else None
    )
    x_label_expression = (
        _label_expression(chart.x_labels) if chart.x_labels else TICK_LABEL_EXPRESSION
    )
    if chart.x_tick_labels is not None:
        labelled = json.dumps(list(chart.x_tick_labels))
        x_label_expression = (
            f"indexof({labelled}, datum.value) >= 0 ? ({x_label_expression}) : ''"
        )
    # x_padding, and under 6 px more: the data's width then ends with a whole dash ([7, 5]) and dot ([1, 5]), so that
    # dashed and dotted levels over all of it end flush
    tight = (
        {
            "nice": False,
            "padding": alt.ExprRef(
                f"(width - 7 - 12 * floor((width - {2 * chart.x_padding} - 7) / 12)) / 2"
            ),
        }
        if chart.x_padding is not None
        else {}
    )
    x = alt.X(
        "x:Q",
        title=chart.x_label,
        scale=alt.Scale(type="log", base=2, zero=False, **tight)
        if chart.log_x
        else alt.Scale(zero=False, **tight),
        axis=alt.Axis(
            values=list(chart.x_ticks or ()) or alt.Undefined,
            labelExpr=x_label_expression,
            **fonts,
        ),
    )
    title_color = (
        {"titleColor": chart.series[0].color}
        if right and len(chart.series) == 1
        else {}
    )
    y = alt.Y(
        "y:Q",
        title=chart.y_label,
        scale=y_scale,
        axis=alt.Axis(labelExpr=TICK_LABEL_EXPRESSION, **fonts, **title_color),
    )
    color = alt.Color(
        "series:N",
        sort=labels,
        scale=alt.Scale(domain=labels, range=colors),
        legend=legend,
    )
    # by series, with the color's legend, so that the legend draws a dashed series' symbol dashed too
    dash = alt.StrokeDash(
        "series:N",
        sort=labels,
        scale=alt.Scale(
            domain=labels,
            range=[
                [1, 5] if series.dotted else [7, 5] if series.dashed else [1, 0]
                for series in every
            ],
        ),
        legend=legend,
    )
    # Any: ty cannot follow the chart methods Altair generates (mark_*, encode)
    base: Any = alt.Chart(alt.Data(values=values)).encode(x=x)
    stroke = {"strokeWidth": LINE_WIDTH, "strokeCap": "round", "strokeJoin": "round"}
    line = base.mark_line(**stroke).encode(
        y=y, color=color, detail="series:N", strokeDash=dash
    )
    lines = [line]
    if right:
        # the chart's series on the left axis, `right` on the right one: the two axes of one scale
        mine = alt.FieldOneOfPredicate(
            field="series", oneOf=[s.label for s in chart.series]
        )
        right_y = alt.Y(
            "y:Q",
            title=chart.right_label,
            scale=y_scale,
            axis=alt.Axis(
                labelExpr=TICK_LABEL_EXPRESSION,
                orient="right",
                grid=False,
                titleColor=right.color,
                **fonts,
            ),
        )
        lines = [
            line.transform_filter(mine),
            base.mark_line(**stroke)
            .transform_filter(
                alt.FieldEqualPredicate(field="series", equal=right.label)
            )
            .encode(y=right_y, color=color, detail="series:N", strokeDash=dash),
        ]
    points = base.mark_point(filled=True, size=30, opacity=1).encode(
        # with independent axes, a third: none (with shared ones, none would win the merge and hide the line's)
        y=alt.Y("y:Q", scale=y_scale, axis=None) if right else y,
        color=color,
        detail="series:N",
    )
    if not all(series.dots for series in every):
        points = points.transform_filter(
            alt.FieldOneOfPredicate(
                field="series", oneOf=[s.label for s in every if s.dots]
            )
        )
    rule = (
        base.transform_pivot("series", value="y", groupby=["x"])
        .mark_rule(color="#75666e", strokeDash=[2, 3])
        .encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("x:Q", title="__x")]
            + [
                alt.Tooltip(field=_field(label), type="quantitative", title=label)
                for label in labels
            ],
        )
        # The hover selection lives on the rule layer: Vega-Lite refuses `nearest` on line marks, and its voronoi hit
        # area copies this layer's tooltip, so the tooltip fires anywhere over the plot.
        .add_params(nearest)
    )
    metadata = {
        "kind": "line",
        "series": labels,
        "colors": dict(zip(labels, colors, strict=True)),
        "xField": "x",
        "xPrefix": chart.x_prefix,
        **({"xLabels": chart.x_labels} if chart.x_labels else {}),
        "decimals": chart.tooltip_decimals,
        "valueSuffix": chart.value_suffix,
        "seriesField": "series",
    }
    layered = alt.layer(*lines, points, rule).properties(
        width="container", height=height, usermeta={"report": metadata}
    )
    return layered.resolve_axis(y="independent") if right else layered


@dataclass(frozen=True, slots=True)
class Bar:
    label: str
    value: float
    text: str  # beside the bar
    tip: tuple[str, ...]  # the hover card: a title, then paragraphs
    color: str | None = None  # in place of the chart's


@dataclass(frozen=True, slots=True)
class BarChart:
    x_label: str  # of the value axis, as are x_domain, x_ticks and the rule's position (up, for vertical bars)
    bars: tuple[Bar, ...]
    color: str
    x_domain: tuple[float, float]
    x_ticks: tuple[float, ...]
    # a dashed reference line at x, with its label
    rule: tuple[float, str] | None = None
    # log scales labelled in tokens (10M, 1B) or durations (10 s, 1 min, 1 h), or a linear one, plain or in tokens
    axis: Literal["tokens", "seconds", "linear", "linear tokens"] = "tokens"
    # bars standing left to right, their labels below them (a newline in one breaks its line), the text above them
    vertical: bool = False
    # of the labels, the axis text and the text by the bars, in place of report-charts.js's 11 and bar_chart's 12
    font_size: float | None = None


def bar_chart(chart: BarChart) -> alt.TopLevelMixin:
    """Bars, top to bottom (or left to right, if vertical) in the given order, on a log scale of token counts or
    durations or a linear one (of token counts or not); hovering a bar shows its card (report-charts.js's bar tooltip)."""
    values = [
        {
            "label": b.label,
            "value": b.value,
            "text": b.text,
            "color": b.color or chart.color,
        }
        | {f"tip{i}": line for i, line in enumerate(b.tip)}
        for b in chart.bars
    ]
    lines = max(len(b.tip) for b in chart.bars)
    size = chart.font_size or 12
    fonts = (
        {"labelFontSize": chart.font_size, "titleFontSize": chart.font_size}
        if chart.font_size
        else {}
    )
    Across, Up = (alt.Y, alt.X) if not chart.vertical else (alt.X, alt.Y)
    category = Across(
        "label:N",
        sort=None,
        title=None,
        axis=alt.Axis(
            labelLimit=260,
            labelFontSize=size,
            labelColor="#4f4249",  # --ink-2
            ticks=False,
            domain=False,
            **(
                {
                    "labelAngle": 0,
                    "labelPadding": 8,
                    "labelExpr": "split(datum.label, '\\n')",
                }
                if chart.vertical
                else {}
            ),
        ),
    )
    value = Up(
        "value:Q",
        title=chart.x_label,
        scale=alt.Scale(
            type="linear" if chart.axis.startswith("linear") else "log",
            domain=list(chart.x_domain),
        ),
        axis=alt.Axis(
            values=list(chart.x_ticks),
            labelExpr={
                "tokens": TOKEN_LABEL_EXPRESSION,
                "seconds": DURATION_LABEL_EXPRESSION,
                "linear": TICK_LABEL_EXPRESSION,
                "linear tokens": TOKEN_LABEL_EXPRESSION,
            }[chart.axis],
            **fonts,
        ),
    )
    start = alt.datum(chart.x_domain[0])
    # Any: ty cannot follow the chart methods Altair generates (mark_*, encode)
    base: Any = alt.Chart(alt.Data(values=values))
    tooltip = [alt.Tooltip(f"tip{i}:N") for i in range(lines)]
    color = alt.Color("color:N", scale=None)  # each bar's own
    # A log scale has no zero, so the bars start at the left (or bottom) end of the axis (zero, on a linear one).
    if chart.vertical:
        bars = base.mark_bar(width={"band": 0.62}).encode(
            x=category, y=value, y2=start, color=color, tooltip=tooltip
        )
        text = base.mark_text(
            align="center", baseline="bottom", dy=-5, color="#2b2025", fontSize=size
        ).encode(x=category, y=value, text="text:N", tooltip=tooltip)
    else:
        bars = base.mark_bar(height={"band": 0.62}).encode(
            y=category, x=value, x2=start, color=color, tooltip=tooltip
        )
        text = base.mark_text(
            align="left", dx=6, color="#2b2025", fontSize=size
        ).encode(y=category, x=value, text="text:N", tooltip=tooltip)
    layers = [bars, text]
    if chart.rule is not None:
        at, label = chart.rule
        ref: Any = alt.Chart(alt.Data(values=[{"value": at, "label": label}]))
        rule: Any = ref.mark_rule(color="#e86767", strokeDash=[4, 3])
        # across the top, or along the right end of the line when it runs across
        at_end = (
            {"align": "right", "y": value, "x": alt.value("width")}
            if chart.vertical
            else {"align": "center", "x": value, "y": alt.value(0)}
        )
        layers.append(rule.encode(**{"y" if chart.vertical else "x": value}))
        layers.append(
            ref.mark_text(
                color="#e86767",
                fontSize=11,
                align=at_end.pop("align"),
                baseline="bottom",
                dy=-4,
            ).encode(text="label:N", **at_end)
        )
    return alt.layer(*layers).properties(
        width="container",
        height=300 if chart.vertical else 36 * len(chart.bars),
        usermeta={"report": {"kind": "bar-card"}},
    )


# 1e7 -> "10M", 1e9 -> "1B"
TOKEN_LABEL_EXPRESSION = (
    "datum.value >= 1e9 ? format(datum.value / 1e9, '~g') + 'B' : "
    "datum.value >= 1e6 ? format(datum.value / 1e6, '~g') + 'M' : format(datum.value, '~s')"
)
# 10 -> "10 s", 600 -> "10 min", 3600 -> "1 h"
DURATION_LABEL_EXPRESSION = (
    "datum.value >= 3600 ? format(datum.value / 3600, '~g') + ' h' : "
    "datum.value >= 60 ? format(datum.value / 60, '~g') + ' min' : format(datum.value, '~g') + ' s'"
)


def _label_expression(labels: dict[float, str]) -> str:
    """A Vega label expression giving each x of `labels` its label, other values the default."""
    cases = "".join(
        f"datum.value === {x!r} ? {json.dumps(label)} : " for x, label in labels.items()
    )
    return f"{cases}({TICK_LABEL_EXPRESSION})"


def _field(name: str) -> str:
    """`name` as a Vega field: a backslash before each character its access paths treat specially."""
    return re.sub(r"([\\.\[\]'\"])", r"\\\1", name)


def write_chart(path: Path, chart: alt.TopLevelMixin) -> Path:
    """A compact, deterministically gzipped Vega-Lite spec (the name must end in .vl.json.gz)."""
    if not path.name.endswith(".vl.json.gz"):
        raise ValueError("Vega-Lite chart paths must end in .vl.json.gz")
    spec = chart.to_dict()
    spec.pop("config", None)  # report-charts.js supplies the theme
    payload = recolor(
        json.dumps(spec, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    ).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    return path


def figure_card(
    title: str, chart: Path | dict[str, Path], caption: str = "", label: str = ""
) -> str:
    """A card with the chart at `chart` (a file next to the page), or with the first chart of a {label: file} dict
    and buttons that switch between them (report-charts.js); without a title if `title` is empty, as the post's
    figures are, `label` naming the chart in its place."""
    views = chart if isinstance(chart, dict) else {"": chart}
    name = escape(next(iter(views.values())).name, quote=True)
    label = title or label
    embed = (
        f'<div class="altair-chart" data-spec="{name}" role="img" aria-label="{escape(label, quote=True)}">'
        f'<noscript><a href="{name}">Open the {escape(label)} specification</a></noscript></div>'
    )
    toggle = (
        '<div class="view-toggle" role="group">'
        + "".join(
            f'<button type="button" data-spec="{escape(path.name, quote=True)}" aria-pressed="{str(i == 0).lower()}">'
            f"{escape(label)}</button>"
            for i, (label, path) in enumerate(views.items())
        )
        + "</div>"
        if len(views) > 1
        else ""
    )
    caption_html = (
        f'<figcaption class="chart-caption">{caption}</figcaption>' if caption else ""
    )
    head = (
        f'<div class="chart-head"><p class="chart-title">{escape(title)}</p>{toggle}</div>'
        if title or toggle
        else ""
    )
    return f'<figure class="card chart-card">{head}{embed}{caption_html}</figure>'


def charts_card(title: str, charts: dict[str, Path], caption: str = "") -> str:
    """A card with the charts of a {subtitle: file} dict side by side, each under its subtitle (one above the other
    on narrow screens)."""
    cells = "".join(
        f'<div><p class="chart-subtitle">{escape(subtitle)}</p>'
        f'<div class="altair-chart" data-spec="{escape(path.name, quote=True)}" role="img" '
        f'aria-label="{escape(subtitle, quote=True)}"><noscript><a href="{escape(path.name, quote=True)}">Open the '
        f"{escape(subtitle)} specification</a></noscript></div></div>"
        for subtitle, path in charts.items()
    )
    caption_html = (
        f'<figcaption class="chart-caption">{caption}</figcaption>' if caption else ""
    )
    return (
        f'<figure class="card chart-card"><div class="chart-head"><p class="chart-title">{escape(title)}</p></div>'
        f'<div class="chart-grid">{cells}</div>{caption_html}</figure>'
    )


def svg_card(title: str, svg: str, caption: str = "") -> str:
    """A card like figure_card's, with a static drawing (inline svg) in place of a chart; without a title if `title`
    is empty (for a drawing with its own)."""
    caption_html = (
        f'<figcaption class="chart-caption">{caption}</figcaption>' if caption else ""
    )
    head = (
        f'<div class="chart-head"><p class="chart-title">{escape(title)}</p></div>'
        if title
        else ""
    )
    return (
        f'<figure class="card chart-card">{head}'
        f'<div class="svg-figure">{svg}</div>{caption_html}</figure>'
    )


def section(heading: str, description: str, body: str) -> str:
    """A titled section; `description` and `body` may contain HTML."""
    section_id = escape(heading.lower().replace(" ", "-"), quote=True)
    return (
        f'<section class="block" id="{section_id}">'
        f'<div class="section-head"><h2>{escape(heading)}</h2><p>{description}</p></div>{body}</section>'
    )


def post_section(
    kicker: str,
    heading: str,
    paragraphs: list[str],
    figure: str,
    notes: dict[int, str] | None = None,
) -> str:
    """A section of the blog post as the post shows it: its heading, its paragraphs (HTML; a footnote's marker is
    footnote(n)), its figure, and its footnotes' text under them, numbered as in the post. `kicker`, above the heading,
    says which version of the figure it is."""
    note_html = "".join(
        f'<p class="footnote" id="fn-{kicker_id(kicker)}-{n}"><sup>{n}</sup> {text}</p>'
        for n, text in (notes or {}).items()
    )
    return (
        f'<section class="block post-section" id="{kicker_id(kicker)}">'
        f'<p class="kicker">{escape(kicker)}</p><h2>{escape(heading)}</h2>'
        + "".join(f"<p>{p}</p>" for p in paragraphs)
        + f"{figure}{note_html}</section>"
    )


def footnote(n: int) -> str:
    """A footnote's marker, as the post's."""
    return f'<sup class="footnote-ref">{n}</sup>'


def kicker_id(kicker: str) -> str:
    return escape(re.sub(r"[^a-z0-9]+", "-", kicker.lower()).strip("-"), quote=True)


def page(
    *,
    title: str,
    eyebrow: str,
    heading: str,
    lede: str,
    body: str,
    path: str,
    artifact: str,
) -> str:
    """A full page, at `path` from the repo root; `lede` and `body` may contain HTML. Relative links resolve against
    the page's own directory, so the page also works through a symlink (./index.html). Its colors are the blog
    post's (plum.py)."""
    base = f"/{path.rpartition('/')[0]}/"
    return recolor(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="{escape(base, quote=True)}">
  <title>{escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
<main>
  <header class="page">
    <div class="eyebrow">{escape(eyebrow)}</div>
    <h1>{escape(heading)}</h1>
    <p class="lede">{lede}</p>
  </header>

  {body}

  <footer><p class="source" data-path="{escape(path, quote=True)}"></p><p>Artifact: <code>{escape(artifact)}</code></p></footer>
</main>
<script defer src="/assets/vega-runtime.js"></script>
<script defer src="/assets/report-charts.js"></script>
<script>{SOURCE_SCRIPT}</script>
</body>
</html>
""")


# Fills the footer's source line from ./serve_html's /__git__: the page's GitHub link and a button copying its branch, and
# whether the page's directory is in the latest commit as served (else GitHub shows another version, or nothing), and
# if not, a button reloading the page, to check again after committing it.
# Over plain http to another host the page is not a secure context and has no clipboard API, so copying falls back
# to a selection.
SOURCE_SCRIPT = """
(async () => {
  const line = document.querySelector("p.source[data-path]");
  const path = line.dataset.path;
  const dir = path.slice(0, path.lastIndexOf("/") + 1);
  let git;
  try {
    git = await (await fetch(`/__git__?dir=${encodeURIComponent(dir)}`, {cache: "no-store"})).json();
  } catch {
    return;
  }
  if (!git.repository) return;
  const url = `${git.repository}/blob/${git.branch}/${path}`;
  const commit = Object.assign(document.createElement("code"), {textContent: git.commit.slice(0, 10)});
  const warning = git.committed ? "" : "This page is not in the latest commit as served here, so on GitHub it is an older version or missing.";
  const link = Object.assign(document.createElement("a"), {href: url, textContent: "This page on GitHub"});
  const button = Object.assign(document.createElement("button"), {type: "button", className: "copy", textContent: "Copy branch"});
  const copied = Object.assign(document.createElement("span"), {className: "copy-status"});
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(git.branch);
    } catch {
      const area = Object.assign(document.createElement("textarea"), {value: git.branch});
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    copied.textContent = warning ? `Copied. ⚠ ${warning}` : "Copied.";
    copied.classList.toggle("warn", Boolean(warning));
  });
  const refresh = Object.assign(document.createElement("button"), {type: "button", className: "copy", textContent: "Refresh"});
  refresh.addEventListener("click", () => location.reload());
  const state = document.createElement("span");
  if (warning) {
    state.className = "warn";
    state.append(`⚠ ${warning} Latest commit: `, commit, ".");
  } else {
    state.append("✓ In the latest commit, ", commit, ".");
  }
  line.replaceChildren(link, ` (${git.branch})`, button, ...(warning ? [refresh] : []), copied, document.createElement("br"), state);
})();
"""

# The blog post's page (brunborg-com's app.css and its post route): a narrow column of serif text on dark plum, its h2s
# under a pink bar, and the figures on light cards that break out of the column, wider, in Helvetica. Iowan Old Style
# and Helvetica Neue are the post's fonts (bundled there; here, the reader's own, as on a Mac).
CSS = """
:root {
  color-scheme: dark;
  --page: #121014; --text: #e4dde1; --heading: #fff7fb; --rule: #2d2630; --dim: #9b8d95; --pink: #f7329a; --link: #fda5d5;
  /* inside the figure cards */
  --surface: #fffbfd; --ink: #2b2025; --ink-2: #4f4249; --muted: #75666e; --line: #ead9e1; --baseline: #d6c1cb;
  --chip-wash: rgba(43, 32, 37, 0.06);
  --serif: 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, ui-serif, serif;
  --sans: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  --mono: 'Berkeley Mono Variable', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text); font: 400 1.15rem/1.75 var(--serif); }
main { max-width: 42rem; margin: 0 auto; padding: 48px 24px 112px; }
.eyebrow, .kicker { font: 400 .75rem/1.5 var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--pink); }
header.page { padding-bottom: 32px; border-bottom: 1px solid var(--rule); margin-bottom: 48px; }
h1 { margin: 16px 0 0; font-size: clamp(2.4rem, 7vw, 3.75rem); font-weight: 400; line-height: 1.05; letter-spacing: -.02em; color: var(--heading); }
h2 { margin: 3.5rem 0 1rem; font-size: 1.6rem; font-weight: 400; line-height: 1.25; color: var(--heading); }
h2::before { content: ""; display: block; width: 40px; height: 3px; margin-bottom: 1rem; background: var(--pink); }
.kicker + h2 { margin-top: .5rem; }
p { margin: 1.25rem 0; }
.lede { margin-bottom: 0; }
a { color: var(--link); text-decoration-color: rgba(253, 165, 213, .4); text-underline-offset: 4px; }
section.block { margin: 0; }
.section-head p { margin-bottom: 0; }
.post-section { margin-top: 5rem; }
sup { font-size: .65em; line-height: 0; vertical-align: .4em; }
.footnote-ref { color: var(--link); }
.footnote { font-size: .85rem; line-height: 1.5; color: var(--dim); }
code { padding: .1em .35em; border-radius: 4px; background: rgba(255, 255, 255, .08); font: .85em var(--mono); overflow-wrap: anywhere; }
/* the figures: as wide as the page allows, up to 64rem, on light cards in Helvetica */
.card { position: relative; left: 50%; width: min(64rem, calc(100vw - 2rem)); transform: translateX(-50%); background: var(--surface); color: var(--ink); border-radius: 6px; font: 400 15px/1.5 var(--sans); }
.chart-card { margin: 3rem 0; padding: 40px; }
.chart-card + .chart-card { margin-top: -1.5rem; }
.card code { border: 1px solid var(--line); background: var(--surface); color: var(--ink); font: 12px/1.45 var(--sans); }
.chart-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 0 0 6px; }
.chart-title { font-size: 15px; font-weight: 700; color: var(--ink-2); margin: 0; }
.view-toggle { display: inline-flex; flex: none; border: 1px solid var(--baseline); border-radius: 6px; overflow: hidden; }
.view-toggle button { padding: 2px 10px; border: 0; background: transparent; color: var(--muted); font: 13px/1.5 var(--sans); cursor: pointer; }
.view-toggle button + button { border-left: 1px solid var(--baseline); }
.view-toggle button:hover { color: var(--ink); }
.view-toggle button[aria-pressed="true"] { background: var(--chip-wash); color: var(--ink); font-weight: 700; }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 4px 28px; }
.chart-subtitle { font-size: 13px; color: var(--muted); margin: 8px 0 0; }
.chart-caption { color: var(--muted); font-size: 13px; font-style: italic; line-height: 1.45; margin: 12px 0 0; }
.chart-caption a { color: #c2185b; }
.altair-chart { width: 100%; min-height: 240px; }
.svg-figure svg { display: block; width: 100%; height: auto; margin: 8px 0 4px; }
/* a drawing and, while the mouse is over it (or after a tap), another in its place: only the second fades, over the
   card's background, so that where the two are the same nothing changes */
.svg-swap { position: relative; margin: 8px 0 4px; }
.svg-swap svg { margin: 0; }
.svg-swap .alt { position: absolute; inset: 0; opacity: 0; transition: opacity .15s; background: var(--surface); }
.svg-swap:hover .alt, .svg-swap.on .alt { opacity: 1; }
.altair-chart.is-loading { background: linear-gradient(90deg, transparent, var(--chip-wash), transparent); background-size: 200% 100%; animation: chart-loading 1.2s linear infinite; }
.altair-chart .vega-embed, .altair-chart .vega-embed > div { width: 100%; }
.altair-chart .vega-actions { display: none; }
.altair-chart .chart-error { margin: 12px; color: #b3261e; font: 12px/1.45 var(--sans); white-space: pre-wrap; }
@keyframes chart-loading { to { background-position: -200% 0; } }
button.copy { margin: 0 8px 0 4px; padding: 2px 9px; border: 1px solid var(--rule); border-radius: 5px; background: transparent; color: var(--text); font: inherit; cursor: pointer; }
button.copy:hover { border-color: var(--pink); color: var(--heading); }
.warn { color: #f0b35a; }
footer { margin-top: 5rem; border-top: 1px solid var(--rule); padding-top: 22px; color: var(--dim); font: 13px/1.6 var(--sans); }
footer code { color: var(--text); font-size: 12px; }
@media (max-width: 767px) {
  .chart-card { padding: 12px; }
}
"""
