"""The blog post's figures, under fixed names, for brunborg.com's post to import.

    uv run src/retrieval_blog_post/overview.py      # first: computes the statistics (on Slurm) the figures load
    uv run src/retrieval_blog_post/blog.py          # draws the figures and replaces the post's furu/ folder with them

The figures are drawn in overview.py (blog_svgs, blog_charts; the overview page shows them too): the diagrams svg in
Helvetica, each with a phone layout beside it, NAME-phone.svg (fixed-size-array-sizes.svg is the fixed-size array with
its pieces' sizes, for the post to show on hover); the charts JSON, report.py's Chart and BarChart, drawn by the post's
Svelte components; record-progression.json the data of the records animation (records_animation_data), which the post
plays once. They are recolored for the post (plum.py): the overview page's colors, as drawn, become its color scheme.
The post's furu/ folder is the BlogFigures result copied whole, with furu's metadata.json and provenance.json: never
edit it by hand."""

import dataclasses
import json
import shutil
from pathlib import Path

import furu

from retrieval_blog_post.config import POST, STAGING, final_results_root
from retrieval_blog_post.kgram import KGramMatchStats, KGramStats
from retrieval_blog_post.overview import (
    blog_charts,
    blog_svgs,
    records,
    records_animation_data,
)
from retrieval_blog_post.plum import PLUM_ANIMATION, recolor
from retrieval_blog_post.report import BarChart, Chart
from retrieval_blog_post.val_cache import ValCacheSweep


class BlogFigures(furu.Spec[Path]):
    """The post's figures: svg diagrams and JSON charts, named as the post imports them."""

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=final_results_root())

    @furu.dependency
    def match_stats(self) -> KGramMatchStats:
        """The n-gram statistics of the whole validation shard against the record's 370 shards."""
        return KGramMatchStats(val="val")

    @furu.dependency
    def val_cache_sweep(self) -> ValCacheSweep:
        """The #367 record validated with hint corpora of several sizes."""
        return ValCacheSweep()

    def create(self) -> Path:
        data = self.directory.data
        stats: KGramStats = self.match_stats.create()
        for name, svg in blog_svgs().items():
            (data / name).write_text(recolor(svg) + "\n", encoding="utf-8")
        for name, chart in blog_charts(stats, self.val_cache_sweep.create()).items():
            (data / name).write_text(recolor(chart_json(chart)), encoding="utf-8")
        animation = records_animation_data(records()) | {"colors": PLUM_ANIMATION}
        (data / "record-progression.json").write_text(
            json.dumps(animation, indent=1) + "\n", encoding="utf-8"
        )
        return data


def chart_json(chart: Chart | BarChart) -> str:
    """The chart's fields; x_labels (keyed by number) as [x, label] pairs, since JSON keys are strings."""
    fields = dataclasses.asdict(chart)
    if fields.get("x_labels") is not None:
        fields["x_labels"] = list(fields["x_labels"].items())
    return json.dumps(fields, indent=1, ensure_ascii=False) + "\n"


def publish(post: Path = POST) -> Path:
    """Draw the figures anew (their identity does not change with the code) and replace post/furu with them."""
    figures = BlogFigures()
    missing = [
        type(s).__name__
        for s in (figures.match_stats, figures.val_cache_sweep)
        if s.status != "done"
    ]
    if missing:
        raise SystemExit(
            f"not computed yet: {', '.join(missing)}; run src/retrieval_blog_post/overview.py first"
        )
    figures.delete("force")
    data = figures.create()
    shutil.rmtree(STAGING, ignore_errors=True)
    shutil.copytree(data, STAGING)
    # furu's record of the run beside the figures, for us (the post imports only the figures, so they are not served):
    # the spec and the dependencies it loaded, and the commit, the diff's size and the environment it was drawn in
    for record in ("metadata.json", "provenance.json"):
        shutil.copy(data.parent / record, STAGING / record)
    target = post / "furu"
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(STAGING, target)
    return target


if __name__ == "__main__":
    # Imported, not this file's own definition: run as a file, it would be __main__.BlogFigures, another spec.
    from retrieval_blog_post.blog import publish

    print(publish())
