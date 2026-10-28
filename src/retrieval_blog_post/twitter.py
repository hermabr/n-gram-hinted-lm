"""The tweet thread as it would look on X: each tweet's text as drafted, with the overview page's figures as drawn now,
in a mock of X (dark, as on a 1440 px wide screen; narrower, it folds as X's). As X shows a thread, the home
timeline has only its first post, second in the feed, among anonymized placeholder posts (x_feed.py); clicking it
opens the post's page (#post): the first post large, then the rest of the thread under it.

    uv run src/retrieval_blog_post/twitter.py              # draws it from the overview's cached statistics, links
                                                           # ./twitter.html
    ./twitter                                              # from the repo root; open the printed address
    uv run src/retrieval_blog_post/twitter.py --publish    # draws it, and copies it to brunborg.com's SITE_PATH

The page is the Twitter spec's result in final-results/. Run overview.py first: the hint-corpus sweep must be computed.
Running this file always draws it anew (its identity does not change with the code). Beside the page, site/ holds it as
brunborg.com serves it, at SITE_PATH: its assets under SITE_PATH (publish copies them there), its chart's spec not
gzipped (a static host would not say it is), and hidden from search engines (it shows other people's posts).

The figures are drawn in overview.py (tweet_hint_chart, tweet_build_svg; the overview page shows them too). The mock
copies X's sizes, colors and icons (x_icons.py) from a saved copy of its home page. A figure is shown as its
image would be: drawn SHEET px wide on the post's card color and scaled down to the tweet's media width, so its text
shrinks as an uploaded screenshot's would; the build-speed bars, with larger labels and less padding. The hint-corpus
chart is drawn narrower instead (CHART_SHEET px wide), still wider than tall, and the records animation in its tweet
layout (the blog post's phone one, shorter): their text stays readable at a tweet's width. The animation plays as a
video, looping while on screen (a click on the post's page pauses it). A tweet longer than X's 280 characters is cut there with Show more, as X shows a Premium long post."""

import argparse
import gzip
import re
import shutil
from html import escape
from pathlib import Path

import furu

from retrieval_blog_post.config import SITE, final_results_root
from retrieval_blog_post.how_it_works import how_it_works_svg
from retrieval_blog_post.overview import (
    BLOG_CARD,
    BLOG_TITLE,
    CHART_PAD,
    CHART_SHEET,
    animation_json,
    records,
    tweet_build_svg,
    tweet_hint_chart,
)
from retrieval_blog_post.plum import recolor
from retrieval_blog_post.val_cache import ValCacheSweep
from retrieval_blog_post.x_feed import ABOVE, BELOW, Author, Post
from retrieval_blog_post.x_icons import ICONS

BLOG_URL = "https://brunborg.com/n-gram-hinted-lm"  # its card on X: overview.py's BLOG_CARD and BLOG_TITLE
# the page's path on brunborg.com (SITE, a folder of its static files)
SITE_PATH = "/n-gram-hinted-lm-twitter/"
ASSETS = Path(__file__).resolve().parents[2] / "assets"
# what the page loads from assets/, published beside it
SITE_ASSETS = (
    "vega-runtime.js",
    "report-charts.js",
    "improvement-animation.js",
    "twitter",
)
PR_URL = "https://github.com/KellerJordan/modded-nanogpt/pull/367"
LIMIT = 280  # X's characters per post, without Premium
SHEET = 1000  # px: a figure's width as uploaded, the overview's figure cards'
URL = re.compile(r"https?://\S+")


ME = Author("Herman Brunborg", "hermanbrunborg", "/assets/twitter/hermanbrunborg.jpg")
# The left column's links: icon, label (Home is the page shown).
NAV = (
    ("home", "Home"),
    ("explore", "Explore"),
    ("notifications", "Notifications"),
    ("follow", "Follow"),
    ("chat", "Chat"),
    ("calls", "Calls"),
    ("grok", "Grok"),
    ("money", "Money"),
    ("history", "History"),
    ("creator_studio", "Creator Studio"),
    ("premium", "Premium"),
    ("profile", "Profile"),
    ("more", "More"),
)
COMPOSE_TOOLS = (
    "media",
    "gif",
    "grok_image",
    "poll",
    "emoji",
    "schedule",
    "location",
    "flag",
)
FOOTER = (
    "Terms",
    "Privacy",
    "Cookies",
    "Accessibility",
    "Ads Info",
    "More ···",
    "© 2026 X Corp.",
)


class Twitter(furu.Spec[Path]):
    """The thread: its twitter.html, with its chart beside it."""

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=final_results_root())

    @furu.dependency
    def val_cache_sweep(self) -> ValCacheSweep:
        """The #367 record validated with hint corpora of several sizes (the overview's)."""
        return ValCacheSweep()

    def create(self) -> Path:
        data = self.directory.data
        # Each tweet: its text, and its media. The draft's headings: clickbait, the main idea, scaling hint corpus has
        # strong returns, wicked fast hint building, look at my blog/PR.
        tweets = [
            (
                (
                    "I improved NanoGPT Speedrun (GPT-2) from 39.9s → 6.7s (-83.2%), more than every record since 2025 "
                    "combined (-81.4%)!\n\n"
                    "The web is full of rare, repeated content that the LM can’t memorize in time. I solve this with "
                    "retrieval-augmented n-gram hints🧵"
                ),
                video(),
            ),
            (
                (
                    "I build a hint corpus of the tokens that follow each n-gram in the training data.\n\n"
                    "The model receives these tokens as hints and learns how to use them."
                ),
                photo(recolor(how_it_works_svg(stacked=True))),
            ),
            (
                (
                    "Scaling the hint corpus has accelerating returns. On a fixed model checkpoint, each doubling cuts the "
                    "loss more than the last.\n\n"
                    "I build the hint corpus in the timed run, so a faster builder means a larger corpus."
                ),
                photo(
                    tweet_hint_chart(
                        data / "val-cache-only.vl.json.gz",
                        self.val_cache_sweep.create(),
                    ),
                    CHART_SHEET,
                    CHART_PAD,
                ),
            ),
            (
                (
                    "The CPUs used to be idle, so I build the corpus on them, in parallel with model training on the "
                    "GPUs.\n\n"
                    "My builder is branch-free SIMD code that fits in L2 cache and uses a custom data structure. It is "
                    "500x faster than Python and 11x faster than the best auto-researched Rust code."
                ),
                photo(recolor(tweet_build_svg()), pad=12),
            ),
            # the blog post last: X draws the card of a post's last link
            (
                f"Record PR: {PR_URL}\n\nBlog post: {BLOG_URL}",
                link_card(BLOG_CARD, BLOG_TITLE, "brunborg.com"),
            ),
        ]
        (first, first_media), rest = tweets[0], tweets[1:]
        timeline = (
            feed_post(ABOVE)
            + tweet(ME, "1m", first, first_media, opens=True)
            + "".join(feed_post(post) for post in BELOW)
        )
        post = focal(ME, first, first_media) + "".join(
            tweet(ME, "1m", text, media, up=i > 0, down=i < len(rest) - 1)
            for i, (text, media) in enumerate(rest)
        )
        path = data / "index.html"
        page = x_page(
            timeline, post, str(path.relative_to(final_results_root().parent))
        )
        path.write_text(page, encoding="utf-8")
        site = data / "site"
        site.mkdir()
        (site / "index.html").write_text(
            for_site(page, f"/{path.parent.relative_to(final_results_root().parent)}/"),
            encoding="utf-8",
        )
        for spec in data.glob("*.vl.json.gz"):
            (site / spec.name.removesuffix(".gz")).write_bytes(
                gzip.decompress(spec.read_bytes())
            )
        return path


def for_site(page: str, base: str) -> str:
    """The page (with its <base> at `base`) as brunborg.com serves it at SITE_PATH: its base there, its assets under it
    (every one is referenced from /assets/), its charts' specs not gzipped, and a robots tag keeping it out of search
    engines."""
    out = page.replace(
        f'<base href="{base}">',
        f'<base href="{SITE_PATH}">\n  <meta name="robots" content="noindex">',
    )
    out = out.replace('"/assets/', f'"{SITE_PATH}assets/').replace(
        '.vl.json.gz"', '.vl.json"'
    )
    assert out.count(SITE_PATH) > page.count("/assets/"), (
        "the page's base or assets were not where expected"
    )
    return out


def publish(page: Path, site: Path = SITE) -> Path:
    """Replace site's static/SITE_PATH with the drawn page (Twitter's result, `page`): its site/ copy, and the assets it
    loads."""
    target = site / "static" / SITE_PATH.strip("/")
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(page.parent / "site", target)
    for name in SITE_ASSETS:
        source, dest = ASSETS / name, target / "assets" / name
        if source.is_dir():
            shutil.copytree(source, dest)
        else:
            dest.parent.mkdir(exist_ok=True)
            shutil.copy(source, dest)
    return target


def icon(name: str, cls: str = "") -> str:
    view_box, paths = ICONS[name]
    return f'<svg viewBox="{view_box}" aria-hidden="true" class="x-icon {cls}">{paths}</svg>'


def weight(c: str) -> int:
    """A character's weight in X's count: 1 for Latin and most punctuation, 0 inside an emoji sequence, else 2."""
    o = ord(c)
    if o in (0x200D, 0xFE0F):
        return 0
    return (
        1
        if o <= 4351 or 8192 <= o <= 8205 or 8208 <= o <= 8223 or 8242 <= o <= 8247
        else 2
    )


def length(text: str) -> int:
    """`text`'s length as X counts it: weighted characters, each link 23."""
    return sum(map(weight, URL.sub("x" * 23, text)))


def linked(text: str) -> str:
    """`text` escaped, its links as X shows them: without the scheme, cut after 24 characters."""
    out, at = [], 0
    for m in URL.finditer(text):
        shown = re.sub(r"^https?://(www\.)?", "", m[0])
        shown = shown if len(shown) <= 25 else shown[:24] + "…"
        out += [
            escape(text[at : m.start()]),
            f'<a href="{escape(m[0], quote=True)}">{escape(shown)}</a>',
        ]
        at = m.end()
    return "".join(out) + escape(text[at:])


def tweet_text(text: str) -> str:
    """The tweet's text; past LIMIT, cut at the last space before it, with Show more."""
    if length(text) <= LIMIT:
        return f'<div class="x-text">{linked(text)}</div>'
    total, cut = 0, 0
    for i, c in enumerate(text):
        total += weight(c)
        if total > LIMIT:
            break
        if c.isspace():
            cut = i
    return (
        f'<div class="x-text" title="{length(text)} characters, over X\'s {LIMIT}">'
        f'<span class="x-short">{linked(text[:cut].rstrip())}…</span><span class="x-full">{linked(text)}</span>'
        '<button type="button" class="x-show-more">Show more</button></div>'
    )


def short_count(n: int) -> str:
    """A count as X shows it: up to 999 as is, then cut (not rounded) to 2.5K, 35K, 1.2M."""
    for scale, unit in ((1e6, "M"), (1e3, "K")):
        if n >= scale:
            value = n / scale
            shown = f"{int(value * 10) / 10:g}" if value < 10 else f"{int(value)}"
            return shown + unit
    return str(n)


def avatar(author: Author) -> str:
    """The author's picture, or a blank circle if they have none."""
    if author.avatar is None:
        return '<span class="x-avatar x-blank"></span>'
    return f'<img class="x-avatar" src="{author.avatar}" alt="{escape(author.name)}">'


def who(author: Author) -> str:
    """The author's name, with X's verified badge if they have it."""
    badge = icon("verified", "x-verified") if author.verified else ""
    return f'<span class="x-name">{escape(author.name)}{badge}</span>'


def tweet(
    author: Author,
    ago: str,
    text: str,
    media: str,
    *,
    counts: tuple[int, int, int, int] | None = None,
    opens: bool = False,
    up: bool = False,
    down: bool = False,
) -> str:
    """A post as the timeline shows it, with its counts if given; one that `opens` opens the post's page on a click,
    and `up` and `down` join it to the posts above and below by the line under the avatars, as in a thread."""
    classes = " ".join(
        ["x-tweet", *(["x-up"] if up else []), *(["x-down"] if down else [])]
    )
    actions = "".join(
        f'<span class="x-action">{icon(name)}<span>{short_count(n) if counts else ""}</span></span>'
        for name, n in zip(("reply", "repost", "like", "views"), counts or (0, 0, 0, 0))
    )
    return (
        f'<article class="{classes}"{" data-opens" if opens else ""}>{avatar(author)}'
        f'<div class="x-body"><div class="x-head">{who(author)}'
        f'<span class="x-muted x-handle">@{author.handle}</span>'
        f'<span class="x-muted">·</span><span class="x-muted">{ago}</span>'
        f'<span class="x-head-icons">{icon("grok")}{icon("caret")}</span></div>'
        f"{tweet_text(text)}{media}"
        f'<div class="x-actions">{actions}<span class="x-action-end">{icon("bookmark")}{icon("share")}</span>'
        "</div></div></article>"
    )


def focal(author: Author, text: str, media: str) -> str:
    """The post its page is of, as X shows it there: larger text, whole however long, the time it was posted (filled in
    by SCRIPT, as now) and its views, the actions between rules, and the reply box under it."""
    actions = "".join(
        icon(name) for name in ("reply", "repost", "like", "bookmark", "share")
    )
    return (
        '<article class="x-focal"><div class="x-focal-head">'
        f'<img class="x-avatar" src="{author.avatar}" alt="{escape(author.name)}">'
        f'<div>{who(author)}<div class="x-muted">@{author.handle}</div></div>'
        f'<span class="x-head-icons">{icon("grok")}{icon("caret")}</span></div>'
        f'<div class="x-text">{linked(text)}</div>{media}'
        '<div class="x-focal-time"><span class="x-posted"></span> · <b>1</b> View</div>'
        f'<div class="x-focal-actions">{actions}</div></article>'
        f'<div class="x-reply-box"><img class="x-avatar" src="{author.avatar}" alt="">'
        '<span class="x-placeholder">Post your reply</span><span class="x-compose-post">Reply</span></div>'
    )


def feed_post(post: Post) -> str:
    """One of the timeline's other posts."""
    return tweet(post.author, post.ago, post.text, "", counts=post.counts)


def link_card(image: str, title: str, site: str) -> str:
    """A link's large card: its image, with its title over the image's corner, and its site under it."""
    return (
        f'<div class="x-media x-link-card"><img src="{image}" alt=""><span>{escape(title)}</span></div>'
        f'<div class="x-card-site">From {escape(site)}</div>'
    )


def photo(figure: str, width: int = SHEET, pad: int = 28) -> str:
    """An image: `figure` on a `width` px wide card padded `pad` px, scaled to the media's width (by SCRIPT)."""
    return (
        f'<div class="x-media x-photo"><div class="x-sheet" style="width: {width}px; padding: {pad}px">'
        f"{figure}</div></div>"
    )


def video() -> str:
    """The records animation in its tweet layout (its phone one, shorter), played by improvement-animation.js, its
    controls hidden."""
    payload = animation_json(records())
    return (
        '<div class="x-media x-video"><figure class="animation-card" data-layout="tweet"><canvas role="img" '
        'aria-label="The world record over time, animated"></canvas>'
        f'<script type="application/json">{payload}</script></figure></div>'
    )


def x_page(timeline: str, post: str, path: str) -> str:
    """X's home timeline with the posts of `timeline`, and a post's page (`post`, shown at #post), the page at `path`
    from the repo root."""
    base = f"/{path.rpartition('/')[0]}/"
    nav = "".join(
        f'<a class="x-nav-item{" x-on" if i == 0 else ""}"{" data-home" if i == 0 else ""}>{icon(name)}'
        f"<span>{label}</span></a>"
        for i, (name, label) in enumerate(NAV)
    )
    tools = "".join(icon(name) for name in COMPOSE_TOOLS)
    footer = "".join(f"<span>{escape(item)}</span>" for item in FOOTER)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="{escape(base, quote=True)}">
  <title>Home / X</title>
  <style>{CSS}</style>
</head>
<body>
<div class="x-app">
  <header class="x-nav"><div class="x-nav-inner">
    <div>
      <a class="x-logo" data-home>{icon("logo")}</a>
      <nav>{nav}</nav>
      <a class="x-post-button">Post</a>
    </div>
    <div class="x-account"><img class="x-avatar" src="{ME.avatar}" alt="">
      <div><div class="x-name">{escape(ME.name)}</div><div class="x-muted">@{ME.handle}</div></div>{icon("caret")}</div>
  </div></header>
  <main class="x-main">
    <div class="x-home">
      <div class="x-tabs"><a class="x-on"><span>For you</span></a><a><span>Following</span></a></div>
      <div class="x-compose"><img class="x-avatar" src="{ME.avatar}" alt="">
        <div class="x-compose-body"><div class="x-placeholder">What’s happening?</div>
          <div class="x-toolbar"><div>{tools}</div><span class="x-compose-post">Post</span></div></div></div>
      {timeline}
    </div>
    <div class="x-post">
      <div class="x-post-header"><a class="x-back" href="#" aria-label="Back">{icon("back")}</a><h2>Post</h2></div>
      {post}
    </div>
  </main>
  <aside class="x-side">
    <div class="x-search">{icon("search")}<span>Search</span></div>
    <section class="x-card"><h2>Subscribe to Premium</h2>
      <p>Get rid of ads, see your analytics, boost your replies and unlock 20+ features.</p>
      <span class="x-subscribe">Subscribe</span></section>
    <nav class="x-footer">{footer}</nav>
  </aside>
</div>
<script defer src="/assets/vega-runtime.js"></script>
<script defer src="/assets/report-charts.js"></script>
<script defer src="/assets/improvement-animation.js"></script>
<script>{SCRIPT}</script>
</body>
</html>
"""


# Scales each figure's sheet to its media's width (and the media to the sheet's height, which changes as a chart
# loads); shows the post's page at #post, the timeline else, back where it was scrolled to; opens a long post on Show
# more; opens the post's page on a click on the thread's first post, and pauses or plays a video there on a click; and
# goes back to the timeline from the post's page on its back arrow, Home or the X logo (on the timeline, Home and the
# logo scroll it to the top, as X's do).
SCRIPT = """
let homeScroll = 0;
const route = () => {
  const post = location.hash === "#post";
  document.body.classList.toggle("x-on-post", post);
  scrollTo(0, post ? 0 : homeScroll);
};
addEventListener("hashchange", route);
addEventListener("popstate", route);
route();
const posted = new Date();
document.querySelector(".x-posted").textContent =
  `${posted.toLocaleTimeString("en-US", {hour: "numeric", minute: "2-digit"})} · ${posted.toLocaleDateString("en-US", {month: "short", day: "numeric", year: "numeric"})}`;
const fit = () => document.querySelectorAll(".x-photo").forEach((media) => {
  const sheet = media.firstElementChild, scale = media.clientWidth / sheet.offsetWidth;
  sheet.style.transform = `scale(${scale})`;
  media.style.height = `${sheet.offsetHeight * scale}px`;
});
const resized = new ResizeObserver(fit);
document.querySelectorAll(".x-photo, .x-sheet").forEach((element) => resized.observe(element));
document.addEventListener("click", (event) => {
  const more = event.target.closest(".x-show-more");
  if (more) return more.parentElement.classList.add("x-open");
  if (event.target.closest(".x-back, [data-home]")) {
    event.preventDefault();
    if (location.hash !== "#post") return scrollTo({top: 0, behavior: "smooth"});
    return history.state?.fromHome ? history.back() : (location.hash = "");
  }
  const opens = event.target.closest("[data-opens]");
  if (opens && !event.target.closest("a")) {
    homeScroll = scrollY;
    history.replaceState({}, "");
    history.pushState({fromHome: true}, "", "#post");
    return route();
  }
  const video = event.target.closest(".x-video");
  if (video) video.querySelector('[data-a="play"]')?.click();
});
"""

# X's dark theme, as its web app draws it at 1440 px: a 275 px column of links, the 600 px timeline and a 350 px
# sidebar; below 1280 px the links lose their labels, below 1005 px the sidebar goes, below 700 px the links do.
CSS = """
* { box-sizing: border-box; }
html { font-size: 15px; color-scheme: dark; }
body { margin: 0; background: #000; color: #e7e9ea; font: 400 15px/20px TwitterChirp, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
a { color: inherit; text-decoration: none; }
.x-icon { width: 1.25em; height: 1.25em; fill: currentColor; flex-shrink: 0; }
.x-muted { color: #71767b; }
.x-name { display: inline-flex; align-items: center; font-weight: 700; }
.x-verified { width: 18.75px; height: 18.75px; margin-left: 2px; fill: #1d9bf0; }
.x-post, .x-on-post .x-home { display: none; }
.x-on-post .x-post { display: block; }
.x-avatar { width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0; object-fit: cover; }
.x-blank { display: block; background: #333639; }
.x-app { display: flex; justify-content: center; }

.x-nav { width: 275px; flex-shrink: 0; }
.x-nav-inner { position: sticky; top: 0; height: 100vh; padding: 0 8px; display: flex; flex-direction: column; justify-content: space-between; }
.x-logo { display: inline-flex; padding: 12px; margin-top: 2px; border-radius: 9999px; }
.x-logo .x-icon { width: 30px; height: 30px; }
.x-nav nav { display: flex; flex-direction: column; align-items: flex-start; }
.x-nav-item { display: inline-flex; align-items: center; gap: 20px; padding: 12px; margin: 4px 0; border-radius: 9999px; font-size: 20px; line-height: 24px; cursor: pointer; }
.x-nav-item:hover, .x-logo:hover, .x-account:hover { background: rgba(231, 233, 234, .1); }
.x-nav-item .x-icon { width: 26.25px; height: 26.25px; }
.x-nav-item.x-on { font-weight: 700; }
.x-post-button { display: flex; align-items: center; justify-content: center; width: 90%; height: 52px; margin: 16px 0; border-radius: 9999px; background: #eff3f4; color: #0f1419; font-size: 17px; font-weight: 700; cursor: pointer; }
.x-account { display: flex; align-items: center; gap: 12px; padding: 12px; margin: 12px 0; border-radius: 9999px; cursor: pointer; }
.x-account > div { flex: 1; min-width: 0; line-height: 20px; }
.x-account > .x-icon { width: 18.75px; height: 18.75px; }

.x-main { width: 600px; max-width: 100%; min-height: 100vh; border-left: 1px solid #2f3336; border-right: 1px solid #2f3336; }
.x-tabs { position: sticky; top: 0; z-index: 3; display: flex; height: 53px; border-bottom: 1px solid #2f3336; background: rgba(0, 0, 0, .65); backdrop-filter: blur(12px); }
.x-tabs a { flex: 1; display: flex; align-items: center; justify-content: center; color: #71767b; font-weight: 500; cursor: pointer; }
.x-tabs a:hover { background: rgba(231, 233, 234, .1); }
.x-tabs a span { position: relative; height: 100%; display: flex; align-items: center; }
.x-tabs a.x-on { color: #e7e9ea; font-weight: 700; }
.x-tabs a.x-on span::after { content: ""; position: absolute; left: -2px; right: -2px; bottom: 0; height: 4px; border-radius: 9999px; background: #1d9bf0; }
.x-compose { display: flex; gap: 8px; padding: 16px 16px 8px; border-bottom: 1px solid #2f3336; }
.x-compose-body { flex: 1; min-width: 0; }
.x-placeholder { padding: 8px 2px 20px; color: #71767b; font-size: 20px; line-height: 24px; }
.x-toolbar { display: flex; align-items: center; justify-content: space-between; }
.x-toolbar > div { display: flex; gap: 16px; min-width: 0; padding-left: 2px; overflow: hidden; }
.x-toolbar .x-icon { width: 20px; height: 20px; }
.x-compose-post { flex-shrink: 0; margin-left: 12px; display: flex; align-items: center; height: 36px; padding: 0 16px; border-radius: 9999px; background: #eff3f4; color: #0f1419; font-weight: 700; opacity: .5; }

.x-tweet { position: relative; display: flex; gap: 8px; padding: 12px 16px; border-bottom: 1px solid #2f3336; }
.x-tweet[data-opens] { cursor: pointer; }
.x-tweet[data-opens]:hover { background: rgba(255, 255, 255, .03); }
.x-tweet.x-down { border-bottom: 0; padding-bottom: 4px; }
.x-tweet.x-up::before, .x-tweet.x-down::after { content: ""; position: absolute; left: 35px; width: 2px; background: #333639; }
.x-tweet.x-up::before { top: 0; height: 8px; }
.x-tweet.x-down::after { top: 56px; bottom: 0; }
.x-body { flex: 1; min-width: 0; }
.x-head { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
.x-handle { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.x-head-icons { margin-left: auto; display: flex; gap: 12px; color: #71767b; }
.x-head-icons .x-icon { width: 18.75px; height: 18.75px; }
.x-text { white-space: pre-wrap; overflow-wrap: break-word; }
.x-text a, .x-show-more { color: #1d9bf0; }
.x-text a:hover, .x-show-more:hover { text-decoration: underline; }
.x-full, .x-open .x-short, .x-open .x-show-more { display: none; }
.x-open .x-full { display: inline; }
.x-show-more { display: block; padding: 0; border: 0; background: none; font: inherit; cursor: pointer; }
.x-media { position: relative; margin-top: 12px; border: 1px solid #2f3336; border-radius: 16px; overflow: hidden; background: #fffbfd; }
.x-sheet { transform-origin: 0 0; color: #2b2025; font: 400 15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; }
.x-sheet svg { display: block; width: 100%; height: auto; }
.x-post .x-video { cursor: pointer; }
.x-link-card img { display: block; width: 100%; aspect-ratio: 1.91; object-fit: cover; }
.x-link-card span { position: absolute; left: 12px; bottom: 12px; right: 12px; width: fit-content; padding: 0 4px; border-radius: 4px; background: rgba(0, 0, 0, .77); color: #fff; font-size: 13px; line-height: 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.x-card-site { margin-top: 4px; color: #71767b; font-size: 13px; line-height: 16px; }

.x-post-header { position: sticky; top: 0; z-index: 3; display: flex; align-items: center; gap: 24px; height: 53px; padding: 0 16px; background: rgba(0, 0, 0, .65); backdrop-filter: blur(12px); }
.x-post-header h2 { margin: 0; font-size: 20px; font-weight: 700; line-height: 24px; }
.x-back { display: flex; padding: 8px; margin-left: -8px; border-radius: 9999px; }
.x-back:hover { background: rgba(231, 233, 234, .1); }
.x-back .x-icon { width: 20px; height: 20px; }
.x-focal { padding: 12px 16px 0; }
.x-focal-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.x-focal-head > div { line-height: 20px; }
.x-focal .x-text { font-size: 17px; line-height: 24px; }
.x-focal-time { margin: 16px 0; color: #71767b; }
.x-focal-time b { color: #e7e9ea; }
.x-focal-actions { display: flex; justify-content: space-between; height: 48px; align-items: center; padding: 0 4px; border-top: 1px solid #2f3336; border-bottom: 1px solid #2f3336; color: #71767b; }
.x-focal-actions .x-icon { width: 22.5px; height: 22.5px; }
.x-reply-box { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-bottom: 1px solid #2f3336; }
.x-reply-box .x-placeholder { flex: 1; padding: 0 2px; }
.x-video figure { margin: 0; }
.x-video .animation-card canvas { margin: 0; border-radius: 0; }
.x-video .animation-controls { display: none; }
.x-actions { display: flex; align-items: center; margin-top: 12px; color: #71767b; }
.x-action { flex: 1; display: flex; align-items: center; gap: 4px; font-size: 13px; line-height: 16px; }
.x-action .x-icon, .x-action-end .x-icon { width: 18.75px; height: 18.75px; }
.x-action-end { display: flex; gap: 12px; }

.x-side { width: 350px; margin-left: 30px; flex-shrink: 0; }
.x-search { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 4px; height: 42px; margin: 6px 0 12px; padding: 0 12px; border: 1px solid #2f3336; border-radius: 9999px; background: #000; color: #71767b; font-size: 14px; }
.x-search .x-icon { width: 16px; height: 16px; margin-right: 4px; }
.x-card { padding: 12px 20px 16px; margin-bottom: 16px; border: 1px solid #2f3336; border-radius: 16px; }
.x-card h2 { margin: 0 0 8px; font-size: 20px; font-weight: 800; line-height: 24px; }
.x-card p { margin: 0 0 12px; }
.x-subscribe { display: inline-flex; align-items: center; height: 36px; padding: 0 16px; border-radius: 9999px; background: #1d9bf0; color: #fff; font-weight: 700; }
.x-footer { display: flex; flex-wrap: wrap; gap: 2px 12px; padding: 0 16px; color: #71767b; font-size: 13px; line-height: 16px; }

@media (max-width: 1279px) {
  .x-nav { width: 88px; }
  .x-nav-inner { align-items: center; }
  .x-nav-item span, .x-post-button, .x-account > div, .x-account > .x-icon { display: none; }
}
@media (max-width: 1004px) { .x-side { display: none; } }
@media (max-width: 699px) {
  .x-nav { display: none; }
  .x-main { border: 0; }
}
"""


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.Twitter, another spec.
        from retrieval_blog_post.overview import link_index
        from retrieval_blog_post.twitter import Twitter, publish

        parser = argparse.ArgumentParser(description="Draw the tweet thread's page.")
        parser.add_argument(
            "--publish",
            action="store_true",
            help=f"also copy it to brunborg.com's {SITE_PATH}",
        )
        args = parser.parse_args()
        twitter = Twitter()
        if twitter.val_cache_sweep.status != "done":
            raise SystemExit(
                "not computed yet: ValCacheSweep; run src/retrieval_blog_post/overview.py first"
            )
        twitter.delete("force")
        page_path = twitter.create()
        print(link_index(page_path, "twitter.html"), "->", page_path)
        if args.publish:
            print("published:", publish(page_path))

    _main()
