"""The blog post's color scheme ("Plum" on #fffbfd, picked from drafts on the site), for the post's figures (blog.py)
and the overview page, which is drawn in it to look as the post does.

The figures are drawn in overview.py's colors; recolor makes them the post's. Grays become plum grays; the hints (next
token, the hint arrows, the hint-corpus line, the record's drop) are pink, the colors beside them pastels darkened to
read on the light background, and each light fill a pale tint of its color."""

import re

# each color the figures are drawn in, overview.py's, to the post's
PLUM = {
    # background, cells and text
    "#f2f2f2": "#f3ecef",
    "#ececec": "#f3ecef",
    "#1c1c1c": "#2b2025",
    "#1d1d1f": "#2b2025",
    "#6b6b70": "#75666e",
    "#666": "#75666e",
    "#8a8a8a": "#9a8f95",
    "#9a9a9a": "#9a8f95",
    # blue, the hints' orange (now pink), purple, green, teal, and rose (the 20-gram, now amber); then their fills
    "#1f5fa8": "#517ec7",
    "#5b8dc9": "#517ec7",
    "#c86b12": "#f7329a",
    "#b4541a": "#f7329a",
    "#e8772e": "#f7329a",
    "#7a4fa0": "#906cbe",
    "#8f6fc0": "#906cbe",
    "#2a8a4a": "#2b8d5a",
    "#73a656": "#2b8d5a",
    "#0f7f86": "#008c90",
    "#b0365e": "#9b7520",
    "#c9dbf0": "#dfe8ff",
    "#dcebf7": "#dfe8ff",
    "#f6dcc2": "#ffdfee",
    "#e6dbf0": "#f0e3fe",
    "#eee6f7": "#f0e3fe",
    "#e6f2df": "#d2efdc",
    "#c6e6e7": "#c8f0f1",
    "#f3d2dd": "#f5e6ce",
    # the line charts: top-1, top-2 and the hint-corpus line, top-4, the no-hint-corpus line
    "#8b67c7": "#906cbe",
    "#3987e5": "#f7329a",
    "#1a9a1a": "#2b8d5a",
    "#c98500": "#887781",
    # the build-speedup bars, top to bottom
    "#1b9aa3": "#008c90",
    "#3a7be6": "#517ec7",
    "#6a75da": "#7775c4",
    "#9a70ce": "#906cbe",
    "#cb6ac2": "#b45bac",
    "#fb64b6": "#f7329a",
}
# the records animation's (its palette, C in improvement-animation.js)
PLUM_ANIMATION = {
    "bg": "#fffbfd",
    "ink": "#2b2025",
    "muted": "#75666e",
    "grid": "#f0e5eb",
    "blue": "#3a7be6",  # overview.py's blue, as drawn
    "pink": "#f7329a",
}


def recolor(text: str) -> str:
    """text with each of PLUM's colors (#rrggbb or #rgb, in any case) replaced by the post's. No color is both one of
    PLUM's and one it becomes, so recoloring twice changes nothing."""
    return re.sub(
        r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b",
        lambda m: PLUM.get(m[0].lower(), m[0]),
        text,
    )
