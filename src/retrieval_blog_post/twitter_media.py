"""The tweet thread's media as files to upload to X, drawn from its page (Twitter's) in a headless Chromium: the records
animation as an MP4, each figure as a PNG, and the blog post's link card (its og:image).

    uv run src/retrieval_blog_post/twitter_media.py        # draws them, prints their directory, and copies the card
                                                           # where the mock and brunborg.com load it
    uv run src/retrieval_blog_post/twitter.py --publish    # then, for the card in the mock on brunborg.com

The media are the TwitterMedia spec's result in final-results/; running this file draws the page and them anew (as
twitter.py does the page: their identities do not change with the code). The page is served from a temporary copy of
the site, as brunborg.com serves it. If SITE has brunborg.com's fonts (a Helvetica Neue clone for the figures, Iowan Old
Style for the card's title), Chromium draws in them, so that the files look as on a Mac; else in this machine's fonts.

The animation is stepped through frame by frame (its time slider, not real time), one loop from where the page starts
it, so that the video's first frame (X's preview of it) is the finished chart, as the page's is; FPS frames a second,
at twice its layout's size, on the card's color. A figure is its sheet as the tweet shows it (scaled to the tweet's
media width), at SCALE times that. The card is CARD_W x CARD_H: the post's title on the left as the post's page has it,
and the animation's finished chart on the right; X writes the title over its bottom left corner, under the post's."""

import base64
import functools
import http.server
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import furu
import imageio_ffmpeg
from fontTools.ttLib import TTFont
from playwright.sync_api import sync_playwright

from retrieval_blog_post.config import final_results_root
from retrieval_blog_post.overview import BLOG_CARD, BLOG_TITLE
from retrieval_blog_post.twitter import ASSETS, SITE, SITE_PATH, Twitter, publish

FONTS = SITE / "static" / "fonts"
FPS = 60
SCALE = 4  # a figure's pixels per CSS px: about 2070 px wide, at the tweet's 518 px
CARD_W, CARD_H = 1200, 630
# the figures, in the thread's order: each the tweet (of the post's replies, under the first post) it is in
FIGURES = {"how-it-works.png": 0, "hint-corpus-scaling.png": 1, "build-speed.png": 2}
CARD = Path(BLOG_CARD).name  # as brunborg.com serves it, at its root
# brunborg.com's colors: the page's background, the title's and the rule's (plum-950, plum-50, plum-800)
BACKGROUND, TITLE, RULE = "#121014", "#fff7fb", "#2d2630"


def fontconfig(directory: Path) -> Path:
    """A fontconfig file adding the bundled Helvetica Neue clone (as TTFs named Helvetica Neue, which the page's fonts
    ask for) to the system's fonts."""
    for name, style in (("HelveticaNeue", "Regular"), ("HelveticaNeue-Bold", "Bold")):
        font = TTFont(FONTS / f"{name}.woff2")
        font.flavor = None
        names = font["name"]
        for platform, encoding, language in ((3, 1, 0x409), (1, 0, 0)):
            names.setName("Helvetica Neue", 1, platform, encoding, language)
            names.setName(style, 2, platform, encoding, language)
            names.setName(f"Helvetica Neue {style}", 4, platform, encoding, language)
            names.setName(name, 6, platform, encoding, language)
        for typographic in (16, 17):
            names.removeNames(nameID=typographic)
        font.save(directory / f"{name}.ttf")
    config = directory / "fonts.conf"
    config.write_text(
        '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
        f'<include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>{directory}</dir>'
        f"<cachedir>{directory / 'cache'}</cachedir></fontconfig>"
    )
    return config


def chromium() -> str | None:
    """An installed headless Chromium (the newest), if the installed playwright's own is not there; else playwright's,
    installed if there is none."""
    shells = sorted(
        Path.home().glob(
            ".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
        ),
        key=lambda p: int(p.parts[-3].rsplit("-", 1)[1]),
    )
    if shells:
        return str(shells[-1])
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    return None


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    """`root` served on a free local port, in a thread; the server and its address."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# The focal post's animation, paused where the page starts it, its slider taking any time: its start, its length, and
# the card's color behind it.
PAUSE = """() => {
  const figure = document.querySelector(".x-focal .animation-card"), controls = figure.querySelector(".animation-controls");
  const play = controls.querySelector('[data-a="play"]'), scrub = controls.querySelector("input");
  if (play.textContent.includes("Pause")) play.click();
  scrub.step = "any";
  return [+scrub.value, +scrub.max, getComputedStyle(figure.closest(".x-media")).backgroundColor];
}"""
# The animation at time t, once drawn, as a PNG data URL on the card's color.
FRAME = """async ([t, color]) => {
  const figure = document.querySelector(".x-focal .animation-card"), scrub = figure.querySelector("input");
  scrub.value = t; scrub.dispatchEvent(new Event("input"));
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  const canvas = figure.querySelector("canvas"), out = document.createElement("canvas");
  out.width = canvas.width; out.height = canvas.height;
  const c = out.getContext("2d"); c.fillStyle = color; c.fillRect(0, 0, out.width, out.height); c.drawImage(canvas, 0, 0);
  return out.toDataURL("image/png");
}"""


# An element's box inside its border, in the viewport, 1 px further in (of the sheet's blank padding): its border, at a
# fraction of a px, shows in the edge's pixels otherwise.
INSIDE = """e => {
  const r = e.getBoundingClientRect();
  return { x: r.x + e.clientLeft + 1, y: r.y + e.clientTop + 1, width: e.clientWidth - 2, height: e.clientHeight - 2 };
}"""


def png(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1])


def video(page, frames: Path, out: Path) -> bytes:
    """The animation's loop from the page's start, as an MP4 at `out`; its first frame's PNG (the finished chart)."""
    start, duration, color = page.evaluate(PAUSE)
    count = round(duration * FPS)
    for i in range(count):
        (frames / f"{i:05d}.png").write_bytes(
            png(page.evaluate(FRAME, [(start + i / FPS) % duration, color]))
        )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames / "%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return (frames / "00000.png").read_bytes()


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def card(page, chart: bytes) -> bytes:
    """The link card: the post's title over a rule on the left, the finished chart in its card on the right."""
    iowan = FONTS / "IowanOldStyle-Roman.woff2"
    face = ""
    if iowan.exists():
        font = data_url(iowan.read_bytes(), "font/woff2")
        face = f'@font-face {{ font-family: "Iowan Old Style"; src: url("{font}") format("woff2"); }}'
    page.set_viewport_size({"width": CARD_W, "height": CARD_H})
    page.set_content(
        f"""<!doctype html><style>
{face}
body {{ margin: 0; }}
.card {{ width: {CARD_W}px; height: {CARD_H}px; display: flex; align-items: center; gap: 44px; padding: 0 44px;
  box-sizing: border-box; background: {BACKGROUND}; }}
.title {{ width: 330px; align-self: flex-start; padding-top: 56px; }}
h1 {{ margin: 0; font: 400 68px/1.05 "Iowan Old Style", serif; letter-spacing: -0.02em; color: {TITLE}; }}
.rule {{ margin-top: 28px; border-top: 1.5px solid {RULE}; }}
.chart {{ width: 740px; padding: 14px; box-sizing: border-box; border-radius: 10px; background: #fffbfd; }}
.chart img {{ display: block; width: 100%; }}
</style><div class="card"><div class="title"><h1>{BLOG_TITLE}</h1><div class="rule"></div></div>
<div class="chart"><img src="{data_url(chart, "image/png")}"></div></div>"""
    )
    page.evaluate("document.fonts.ready")
    return page.locator(".card").screenshot()


class TwitterMedia(furu.Spec[Path]):
    """The thread's media, in its data directory: records.mp4, each figure's PNG (FIGURES), and the link card (CARD)."""

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=final_results_root())

    @furu.dependency
    def twitter(self) -> Twitter:
        """The thread's page, which they are drawn from."""
        return Twitter()

    def create(self) -> Path:
        data = self.directory.data
        page_path = self.twitter.create()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "fonts").mkdir()
            (tmp / "frames").mkdir()
            publish(page_path, tmp)
            server, address = serve(tmp / "static")
            try:
                with sync_playwright() as p:
                    env = {"FONTCONFIG_FILE": str(fontconfig(tmp / "fonts"))} if FONTS.is_dir() else None
                    browser = p.chromium.launch(executable_path=chromium(), env=env)
                    page = browser.new_page(
                        viewport={"width": 1440, "height": 1000},
                        device_scale_factor=SCALE,
                    )
                    page.goto(f"{address}{SITE_PATH}#post", wait_until="networkidle")
                    page.wait_for_function("document.fonts.status === 'loaded'")
                    # first, while the animation is on screen (it draws only then), before the figures scroll to theirs
                    finished = video(page, tmp / "frames", data / "records.mp4")
                    page.wait_for_function(  # the hint-corpus chart, drawn by Vega
                        "[...document.querySelectorAll('.x-tweet .altair-chart')].every(c => c.querySelector('svg, canvas'))"
                    )
                    # each figure without X's rounded corners, inside the media's border
                    page.add_style_tag(
                        content=".x-photo { border-radius: 0 !important; }"
                    )
                    photos = page.locator(".x-tweet:visible .x-photo")
                    for name, i in FIGURES.items():
                        photos.nth(i).scroll_into_view_if_needed()
                        page.screenshot(
                            path=data / name, clip=photos.nth(i).evaluate(INSIDE)
                        )
                    (data / CARD).write_bytes(
                        card(browser.new_page(device_scale_factor=1), finished)
                    )
                    browser.close()
            finally:
                server.shutdown()
        return data


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.TwitterMedia, another spec.
        import shutil

        from retrieval_blog_post.twitter_media import CARD, TwitterMedia

        media = TwitterMedia()
        if media.twitter.val_cache_sweep.status != "done":
            raise SystemExit(
                "not computed yet: ValCacheSweep; run src/retrieval_blog_post/overview.py first"
            )
        media.twitter.delete("force")
        media.delete("force")
        data = media.create()
        print("media:", data)
        for target in (ASSETS / BLOG_CARD.removeprefix("/assets/"), SITE / "static" / CARD):
            if target.parent.is_dir():
                shutil.copy(data / CARD, target)
                print("card:", target)

    _main()
