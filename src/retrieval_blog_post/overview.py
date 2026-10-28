"""Every figure of the blog post and the tweet thread on one page, with interactive charts, and Publish: every final
result in one spec.

    uv run src/retrieval_blog_post/overview.py      # computes the results on Slurm, draws them, links ./index.html
    ./serve_html                                    # from the repo root; open the printed address

The page is the Overview spec's result in final-results/, drawn from the cached statistics of the specs it depends
on. Publish depends on it, the post's figures, the thread's page and media, and the runs the llm.c writeup quotes,
and lists them in results.json. Running this file draws the drawings and Publish anew (their identities do not change
with the code). The page's footer asks ./serve_html whether it is in the latest commit.

The post's figures (blog.py's BlogFigures) and the thread's (twitter.py's Twitter) are drawn here too, by the functions
under "The blog post's figures" and "The tweet thread's figures"; blog.py and twitter.py only write them out."""

import dataclasses
import itertools
import json
import math
import operator
import os
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import altair as alt
import furu
import numpy as np
import polars as pl

from retrieval_blog_post.combined_entry import combined_entry_svg
from retrieval_blog_post.config import final_results_root
from retrieval_blog_post.data_scaling import DataScaling, DataScalingLosses
from retrieval_blog_post.fineweb import FinewebShards, Val
from retrieval_blog_post.hints import hints_svg
from retrieval_blog_post.how_it_works import how_it_works_svg
from retrieval_blog_post.kgram import (
    RECORD_360,
    RECORD_367,
    KGramMatchStats,
    KGramStats,
)
from retrieval_blog_post.ladder import (
    MARGIN,
    RECORD,
    RUNGS,
    THREADS,
    Version,
    build_time,
    measured,
    run_estimate,
)
from retrieval_blog_post.one_pass import OnePassTime, combined
from retrieval_blog_post.one_pass import specs as one_pass_specs
from retrieval_blog_post.report import (
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
    Bar,
    BarChart,
    Chart,
    Series,
    bar_chart,
    charts_card,
    figure_card,
    line_chart,
    page,
    post_section,
    section,
    svg_card,
    write_chart,
)
from retrieval_blog_post.slot_index import (
    INK,
    line,
    rect,
    simple_index_svg,
    slot_index_figure,
    slot_index_svg,
    text,
)
from retrieval_blog_post.token_losses import (
    HINT_LENGTHS,
    NoHintsRecordTokenLosses,
    PreviousRecordTokenLosses,
    RecordTokenLosses,
)
from retrieval_blog_post.val_cache import SIZES, ValCacheLosses, ValCacheSweep

if TYPE_CHECKING:
    from retrieval_blog_post.blog import BlogFigures
    from retrieval_blog_post.twitter import Twitter
    from retrieval_blog_post.twitter_media import TwitterMedia

XTICKS = (2, 3, 4, 6, 8, 10, 12, 16, 20, 32, 48, 64)
# The data-scaling sizes shown: 75M to 1.2B. Past 1.2B the record's recipe, stretched, trains worse on more data
# (val loss 3.181 at 1.2B, 3.210 at 2.4B, 3.913 at 4.8B, 4.684 at 9.6B), so those runs say nothing about the data.
TRAINED_SIZES = SIZES[:5]
TOPS = (1, 2, 4)  # the vote accuracies shown
# The validation shards the n-gram statistics are shown for, by legend label: as is, without the documents with an
# exact copy in the training shards, and without those and Gopher's near-duplicates too (fineweb.py's Val).
VALS: dict[str, Val] = {
    "all val": "val",
    "no repeats": "val-exact-dedup",
    "no near-dups": ("val-dedup-fuzzy", 0.8),
}
VAL_COLORS = (PURPLE, BLUE, GREEN)
# seconds: the previous WR (#360, README.md's 0.665 minutes) and the new WR (#367)
NEW_RECORD = (39.9, 6.702)
STACKED_YEARS = (
    "25",
    "26",
)  # the records stacked into one bar, by the year of their date
TOP_RECORDS = 6  # the single records shown after the stack (reductions_chart)
LOG_FLOOR = 10  # (%) reductions_chart's log y axis's bottom, where its bars start
TARGET_LOSS = 3.28  # the speedrun's validation loss to reach, drawn dotted on the hint-corpus chart
PINK = "#fb64b6"  # oklch(71.8% .202 349.761): the new record's bar
# Two bright blues, alternating so that the stack's records stay apart.
STACK_COLORS = ("#3a7be6", "#5592f2")
# The stack's dates: where it stood at the start of each of these months.
STACK_DATES = ((2025, 1, "Jan '25"), (2026, 1, "Jan '26"), (2026, 9, "Sep '26"))

# The post's section its n-gram accuracy chart is in (brunborg-com's Post.svelte, #reliable-long-n-grams), as the post
# shows it: drawn above each version of the chart (reliable_sections), to see the chart as a reader does. Copied by
# hand: copy it again when the post's text changes.
POST_RELIABLE_HEADING = "Long n-gram hints are rare and reliable"
POST_RELIABLE_TEXT = [
    (
        "Web text is repetitive, with validation n-grams also appearing in training. For a 20-gram that also "
        "appears in training, the most common next token is correct 98% of the time, and 54% for 6-grams. If we "
        "record the next tokens of n-grams during training, we can pass them as hints to our model."
    )
]
POST_CHART_HEIGHT = 400  # px: about the post's LineChart's, 420 of its 1000 wide, at a 64rem card's width

# The second line of a record's label, for the records with the largest single improvements.
SHORT_NAMES = {
    92: "ANVIL",
    5: "ReLU², QK-norm",
    2: "tuned LR, RoPE",
    12: "FlexAttention",
    9: "skip connections",
    3: "Muon",
    6: "distributed Muon",
    4: "Muon tweaks",
    15: "U-net value embeds",
    8: "untied head",
    7: "torch 2.5",
    90: "FP8 QKV and MLP",
    11: "U-net skips",
    19: "FP8 head",
    13: "window warmup",
    17: "sparse value embeds",
    14: "value embeds",
    62: "bigram hash",
}


class Overview(furu.Spec[Path]):
    """The page: its index.html, with its charts beside it."""

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=final_results_root())

    @furu.dependency
    def match_stats(self) -> dict[str, dict[str, KGramMatchStats]]:
        """Per validation shard of VALS, the n-gram statistics against the record's 370 shards, and against as much
        data as the previous WR trains on."""
        return {
            label: {
                "37B tokens": KGramMatchStats(val=val),
                "329M tokens (prev WR train tokens)": KGramMatchStats(
                    train=RECORD_360, val=val
                ),
            }
            for label, val in VALS.items()
        }

    @furu.dependency
    def val_cache_sweep(self) -> ValCacheSweep:
        """The #367 record trained once, validated with hint corpora of several sizes (on a miso node's 8 GPUs)."""
        return ValCacheSweep()

    @furu.dependency
    def one_pass(self) -> list[OnePassTime]:
        """The hint index built in two stages (the record's build), in one pass, and hashed only, from the record's 37B
        tokens halved down to about 1B, each size measured in several jobs (on miso-lo workers of 800 GB or more)."""
        return one_pass_specs()

    @furu.dependency
    def data_scaling(self) -> DataScaling:
        """The record without hints, trained on the sweep's hint-corpus sizes up to TRAINED_SIZES (on a miso node's 8
        GPUs)."""
        return DataScaling(tokens=TRAINED_SIZES)

    @furu.dependency
    def token_losses(
        self,
    ) -> list[RecordTokenLosses | NoHintsRecordTokenLosses | PreviousRecordTokenLosses]:
        """The validation loss at every token: of the #367 record with its hints and with blank ones, of the record
        trained without hints, and of the previous record (#360), each one run on a miso node's 8 GPUs."""
        return [
            RecordTokenLosses(),
            NoHintsRecordTokenLosses(),
            PreviousRecordTokenLosses(),
        ]

    def create(self) -> Path:
        data = self.directory.data
        flat = [spec for specs in self.match_stats.values() for spec in specs.values()]
        made = iter(furu.create(flat))
        stats = {
            label: {name: next(made) for name in specs}
            for label, specs in self.match_stats.items()
        }
        record_s = record_build_s()
        rust = run_estimate(measured(["rust-hashmap"])["rust-hashmap"])
        body = (
            section(
                "The largest improvement ever",
                f"The new record takes the time from {NEW_RECORD[0]} s to {NEW_RECORD[1]} s, more than all the records "
                "of 2025 and 2026 together.",
                svg_card(
                    "The world record over time",
                    records_timeline_svg(records()),
                    records_timeline_caption(),
                )
                + records_animation_card(records()),
            )
            + section(
                "How do retrieval hints work?",
                "The context is looked up in the training data on the CPU, and the tokens that followed it there "
                "become a hint. On the GPU, the hint is added to the model's residual stream, which raises those "
                "tokens' predictions.",
                svg_card(
                    "Retrieval hints at a glance",
                    how_it_works_svg(),
                    "The bars are illustrative, not measured.",
                ),
            )
            + reliable_sections(data, stats["all val"]["37B tokens"])
            + section(
                "How often has the training data seen a context?",
                "For each validation position, its n-gram context is the n tokens ending at it, within its document. A match "
                "means the same n tokens occur somewhere in the training data; the vote is the most frequent tokens that "
                "followed them there (ties to the lower token id).",
                charts_card(
                    "All validation documents",
                    {
                        "Validation positions whose n-gram context occurred": write_chart(
                            data / "kgram-match.vl.json.gz",
                            line_chart(match_chart(stats["all val"])),
                        ),
                        "Next token among the n-gram's top votes, given a match": write_chart(
                            data / "kgram-vote.vl.json.gz",
                            line_chart(vote_chart(stats["all val"]["37B tokens"])),
                        ),
                    },
                    "The 329M line uses the first that many tokens of the training shards; the votes are against "
                    "the first 370 shards, the record's hint corpus.",
                )
                + charts_card(
                    "Without the validation documents repeated in the training shards",
                    {
                        "Validation positions whose n-gram context occurred": write_chart(
                            data / "kgram-match-dedup.vl.json.gz",
                            line_chart(dedup_match_chart(stats)),
                        ),
                        "Next token among the n-gram's top votes, given a match": write_chart(
                            data / "kgram-vote-dedup.vl.json.gz",
                            line_chart(dedup_vote_chart(stats)),
                        ),
                        "How many times as often the 37B tokens hold the n-gram context as the 329M": write_chart(
                            data / "kgram-match-ratio-dedup.vl.json.gz",
                            line_chart(dedup_ratio_chart(stats)),
                        ),
                    },
                    f"The validation shard as is; {dedup_caption()} The 329M lines (dashed) use the first that many "
                    "tokens of the training shards; the votes are against the first 370 shards, the record's hint "
                    "corpus. The ratio is the left chart's solid line over its dashed line of the same color.",
                )
                + charts_card(
                    "Top-1, top-2 and top-4 votes without the repeated validation documents",
                    {
                        "No repeats: next token among the n-gram's top votes, given a match": write_chart(
                            data / "kgram-vote-exact-dedup.vl.json.gz",
                            line_chart(vote_chart(stats["no repeats"]["37B tokens"])),
                        ),
                        "No near-dups: next token among the n-gram's top votes, given a match": write_chart(
                            data / "kgram-vote-fuzzy-dedup.vl.json.gz",
                            line_chart(vote_chart(stats["no near-dups"]["37B tokens"])),
                        ),
                    },
                    "As the all-validation vote chart above, for the no repeats and no near-dups validation "
                    "shards; against the first 370 shards, the record's hint corpus.",
                ),
            )
            + section(
                "How fast can the hint corpus be indexed?",
                f"Every version of the hint corpus's index build, from naive Python dicts to the record's, written "
                f"and timed on the same kind of miso-lo node ({THREADS} threads pinned one per core, 800 GB or more per job), "
                "each building about the same data: the next token of every position, keyed by its 6-token context, "
                "with its 10- and 20-token contexts. Hover over a bar for what changed.",
                figure_card(
                    f"Hint corpus indexed in {record_s:.1f} s",
                    write_chart(
                        data / "build-ladder-tokens.vl.json.gz",
                        bar_chart(ladder_tokens_chart()),
                    ),
                    f"The hint corpus each version indexes in the {record_s:.1f} s the record's build takes for its "
                    f"{_tokens(TOKENS)} tokens here, scaled from its measurements at about that size (each the median "
                    "of 3 builds in a fresh process; only the build is timed, not the setup before the record's "
                    "clock). Measured by ladder.py; the Rust versions are in cache_ladder/.",
                )
                + figure_card(
                    f"Hint corpus indexed in {record_s:.1f} s, slowest first",
                    write_chart(
                        data / "build-ladder-tokens-slowest-first.vl.json.gz",
                        bar_chart(ladder_tokens_chart(slowest_on_top=True)),
                    ),
                    "The chart above with Python on top and the record's build at the bottom. In the blog post: "
                    "build-speedups.",
                )
                + svg_card(
                    "",
                    short_ladder_svg(
                        f"Hint corpus tokens in {record_s:.1f} seconds",
                        ladder_tokens_chart(SHORT_LADDER),
                    ),
                    "The chart above with only Python, the best version auto-research found and the record's.",
                )
                + svg_card(
                    "",
                    waterfall_svg(
                        f"Hint corpus tokens in {record_s:.1f} seconds",
                        ladder_tokens_chart(SHORT_LADDER),
                    ),
                    "The same three versions, each arrow how many times as many tokens the next faster one indexes.",
                )
                + figure_card(
                    "Estimated run time of the record with each build",
                    write_chart(
                        data / "build-ladder-run.vl.json.gz",
                        bar_chart(ladder_run_chart()),
                    ),
                    "The record redone with each version's build: the largest hint corpus (in whole shards) whose "
                    f"build, started at the clock, is done by {MARGIN:.0%} of the training it gives, as the record's "
                    "370-shard build is; that training's time to the target, interpolated between record runs with "
                    "hint corpora of 0, 10.3, 25, 27.5, 30, 32.5, 35 and 37B tokens (none larger: no version's corpus "
                    "goes past 37B). Build times are each version's measurements at about that hint-corpus size (at "
                    "its slowest measured rate per token up to it), except version 2's "
                    f"{_tokens(rust['tokens'])}, which would need about 900 GB: scaled from its 8B. An estimate: below "
                    "10B tokens there are no runs between the end points.",
                ),
            )
        )
        one_pass = combined(furu.create(self.one_pass))
        body += section(
            "Two stages or one pass?",
            "The record builds its hint index in two stages: the scan writes every entry into its partition of an "
            "arena, whole cache lines at a time, then each partition is placed into its buckets while its fill counts "
            "stay in the CPU's cache. The alternative is one pass: every entry written straight into its bucket in the "
            "table as it is hashed. Both are written with AVX-512 and no branches on the data, with the same hash and the same "
            f"table, and timed as above ({THREADS} threads, one miso-lo node per size); hashing alone (reading the "
            "tokens, hashing them and computing both buckets, no inserts) is the floor of either.",
            svg_card(
                "One pass, two pass and hash only",
                gap_svg(one_pass, ONE_PASS_BUILDS),
                gap_caption(one_pass, ONE_PASS_BUILDS),
            ),
        )
        body += section(
            "How does the hint index store a 10-gram?",
            "A simplified hint index (the index of the hint corpus) that keeps 10-grams only. The hash of a 10-gram "
            "picks a partition and, in it, two buckets; the 10-gram goes in the first free slot of the emptier of "
            "the two. What the slot holds is the rest of the checksum (all of it but the partition's bits) and the "
            "next token, 64 bits in all.",
            svg_card(
                "Storing one 10-gram, as the Python version does",
                simple_index_svg(),
                "hints maps the hash of each 10-gram to the tokens that followed it. In the blog post: naive-hints.",
            )
            + svg_card(
                "Storing one 10-gram",
                slot_index_figure(),
                "hints has 2¹⁴ partitions of 2¹⁴ buckets, each 48 slots wide (enough for 10.26B training tokens "
                "at 80% load). Hover over the figure for the size of each piece in bits. In the blog post: "
                "fixed-size-array (and fixed-size-array-sizes, on hover).",
            ),
        )
        body += section(
            "How does one table hold the 6-, 10- and 20-gram?",
            "In place of three tables, one per n-gram, the hint index keeps one, keyed on the 6-gram: its hash "
            "picks the partition and gives the check. Short fingerprints of the 10-gram's and the 20-gram's hashes "
            "ride along in the same entry, so one lookup tells how long the match is.",
            svg_card(
                "One entry, three n-grams",
                combined_entry_svg(),
                "A slot holds the 64 bits after the partition, as for the 10-grams alone. In the blog post: "
                "hint-entry-layout.",
            ),
        )
        sweep = self.val_cache_sweep.create()
        scaling = self.data_scaling.create()
        body += section(
            "How much does the size of the hint corpus matter?",
            "One model, trained as the #367 record trains it (seed 0; its training hints come from the tokens it "
            "has trained on so far), validated with a hint corpus of the first N training tokens in place of the "
            "record's 370 shards. Against it, the data alone: the record trained without hints on N tokens (its "
            "model and batch, its schedule stretched to N tokens).",
            figure_card(
                "Validation loss by hint-corpus size",
                {
                    "hint corpus and training data": write_chart(
                        data / "val-cache.vl.json.gz",
                        line_chart(val_cache_chart(sweep, scaling)),
                    ),
                    "hint corpus only": write_chart(
                        data / "val-cache-only.vl.json.gz",
                        line_chart(val_cache_chart(sweep)),
                    ),
                },
                "The dashed line: the same model without hints. The green line's N counts training tokens; it stops "
                "at 1.2B, past which the stretched recipe trains worse on more data. In the blog post (hint corpus "
                "only): hint-corpus-scaling.",
            ),
        )
        body += section(
            "How does the model use a hint?",
            "The hint is added to the residual stream at three places: after the embedding, after layer 6 and before "
            "the final norm, each scaled by its own α. What is added depends on the match: its n-gram's length, how "
            "often it occurred and how pure its votes were.",
            svg_card(
                "Where a hint enters the model",
                hints_svg(),
                "σ and β are learned tables, looked up by the match's length, count and purity: an entry of σ is one "
                "number, an entry of β a 768-dim vector. In the blog post: hint-mechanism.",
            ),
        )
        losses = token_losses_by_hint(*furu.create(self.token_losses))
        body += section(
            "Which tokens do the hints help?",
            "The validation tokens grouped by the record's hint at each: none, or the longest n-gram (6, 10 or 20 "
            "tokens) whose context matched in the hint corpus. The same record validated with blank hints, and the "
            "record trained and validated without hints, show what the model alone gets on those tokens; the previous "
            "record (#360) trains 1194 steps to the record's 540.",
            figure_card(
                "Validation loss by hint length",
                write_chart(
                    data / "token-losses.vl.json.gz",
                    line_chart(token_losses_chart(losses)),
                ),
                token_losses_caption(losses),
            ),
        )
        body += blog_section() + twitter_section(data, sweep)
        # once, for every animation-card above
        body += '<script defer src="/assets/improvement-animation.js"></script>'
        path = data / "index.html"
        path.write_text(
            page(
                title="Retrieval blog post",
                eyebrow="retrieval blog post · figures",
                heading="How much does exact retrieval know?",
                lede="Exact-match statistics of the FineWeb validation set against the training data.",
                body=body,
                path=str(path.relative_to(final_results_root().parent)),
                artifact=self.object_id,
            ),
            encoding="utf-8",
        )
        return path


# The ladder's rungs on the page: the SlotIndex versions only after the per-length tables (not ladder.py's fast
# checksum index), and SlotIndex v1 ("One table") straight to the record (whose card, ABOUT's, covers every step
# since, not ladder.py's intermediate branch-free version and previous record).
LADDER: list[Version] = [
    v for v in RUNGS if v not in ("fast-checksum", "slot-branch-free", "slot-ifma")
]
TOKENS = 37e9  # the record's hint corpus
# The versions' names on the page.
NAMES: dict[Version, str] = {
    "python-dicts": "Python",
    "rust-hashmap": "Best auto-research",
    "per-length-tables": "Hash map → slots",
    "slot-v1": "All lengths together",
    "slot-balanced": "No sort, SIMD, cache local",
}
# What changed against the version before it on the page, where that is not ladder.py's rung before it.
ABOUT: dict[Version, str] = {
    "slot-balanced": "The record code. The scan fills an arena through one cache line per partition held in L2, and "
    "the finish places each partition into 2^14 buckets of the final table through staging lines, both with "
    "non-temporal stores; the inner loops are written with AVX-512 and have no branches on the data. The hash is "
    "computed with IFMA multiplies on byte-permuted token words, 16 positions at a time, one group ahead of their "
    "inserts; a scan thread done with its own pieces takes the back half of the busiest remaining run of pieces, so "
    "no thread waits on the slowest; and the finish prefetches the arena's 8192-entry blocks and places each entry "
    "with one 8-byte store into its bucket's staging line.",
}
# The short ladder's versions and their names: the start, the best version auto-research found and the record.
SHORT_LADDER: dict[Version, str] = {
    "python-dicts": "Python",
    "rust-hashmap": "Auto-research",
    RECORD: "Record",
}


DRAWINGS = ("overview", "blog_figures", "twitter", "twitter_media")  # Publish's drawings, its dependencies' names


class Publish(furu.Spec[Path]):
    """Every final result of the post, the thread and the llm.c writeup, listed in results.json."""

    def metadata(self) -> furu.Metadata:
        return furu.Metadata(storage=final_results_root())

    @furu.dependency
    def overview(self) -> Overview:
        return Overview()

    @furu.dependency
    def blog_figures(self) -> "BlogFigures":
        from retrieval_blog_post.blog import BlogFigures

        return BlogFigures()

    @furu.dependency
    def twitter(self) -> "Twitter":
        from retrieval_blog_post.twitter import Twitter

        return Twitter()

    @furu.dependency
    def twitter_media(self) -> "TwitterMedia":
        from retrieval_blog_post.twitter_media import TwitterMedia

        return TwitterMedia()

    @furu.dependency
    def llmc(self) -> dict[str, list[furu.Spec]]:
        """Record #1, 3 seeds each: llm.c's own trainer with the hints, their validation rows built on CPUs (470 steps;
        llmc_cpu_val.py), and without the hints; with the validation rows built on the GPUs (210 steps;
        llmc_onclock.py); and the PyTorch port whose recipe it runs (old_records.py)."""
        from retrieval_blog_post import llmc_cpu_val
        from retrieval_blog_post.llmc_onclock import LlmcRun
        from retrieval_blog_post.old_records import on_clock_1

        return {
            "llm.c, validation rows on CPUs": [s for s in llmc_cpu_val.BATCH if s.hints],
            "llm.c without hints": [s for s in llmc_cpu_val.BATCH if not s.hints],
            "llm.c, validation rows on GPUs": [LlmcRun(steps=210, seed=seed) for seed in (0, 1, 2)],
            "PyTorch port, validation rows on GPUs": [on_clock_1("210", seed) for seed in (0, 1, 2)],
        }

    def drawings(self) -> list[furu.Spec[Path]]:
        """Drawn from the others' cached results: cheap, and their identities do not change with the code."""
        return [getattr(self, name) for name in DRAWINGS]

    def create(self) -> Path:
        drawn = dict(zip(DRAWINGS, furu.create(self.drawings())))
        runs = {name: furu.create(specs) for name, specs in self.llmc.items()}

        def row(spec, result) -> dict:
            if isinstance(result, tuple):  # llmc_cpu_val's (val loss, seconds)
                return {"seed": spec.seed, "steps": spec.steps, "val_loss": result[0], "seconds": result[1]}
            return {"seed": spec.seed, "steps": result.steps, "val_loss": result.val_loss,
                    "seconds": result.train_time_ms / 1000}

        results = {
            # where each drawing is, under final-results/
            **{name: str(path).rpartition("final-results/")[2] for name, path in drawn.items()},
            "llmc": {
                name: [row(spec, result) for spec, result in zip(self.llmc[name], got)]
                for name, got in runs.items()
            },
        }
        data = self.directory.data
        (data / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return data


def record_build_s() -> float:
    """The record's build of its TOKENS on the ladder's nodes (ladder.build_time)."""
    return build_time(measured([RECORD])[RECORD], TOKENS)


def tokens_in(trials: list[dict], seconds: float) -> float:
    """The tokens a version builds in `seconds`: ladder.build_time inverted (it grows with the tokens), by bisection on
    a log scale."""
    lo, hi = 1e6, 1e12
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if build_time(trials, mid) <= seconds else (lo, mid)
    return lo


def blend(a: str, b: str, t: float) -> str:
    """The color `t` of the way from `a` to `b` (both #rrggbb)."""
    pa, pb = (
        (int(a[i : i + 2], 16) for i in (1, 3, 5)),
        (int(b[i : i + 2], 16) for i in (1, 3, 5)),
    )
    return "#" + "".join(
        f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb, strict=True)
    )


def ladder_tokens_chart(
    names: dict[Version, str] | None = None, slowest_on_top: bool = False
) -> BarChart:
    """The tokens each version builds in the record's build time, the record's on top (at the bottom with
    `slowest_on_top`); the versions go from the records timeline's blue to its pink as they get faster. With `names`,
    only those versions, labelled with them and not numbered (each keeps its color in the whole ladder)."""
    budget, bars, versions = record_build_s(), [], measured(LADDER)
    for i, (v, ts) in enumerate(versions.items(), 1):
        if not ts or (names is not None and v not in names):
            continue
        name = NAMES[v] if names is None else names[v]
        n = tokens_in(ts, budget)
        best = min(ts, key=lambda t: abs(t["median"] - budget))
        runs = ", ".join(f"{s:.2f}" for s in best["seconds"])
        bars.append(
            Bar(
                label=f"{i}. {name}" if names is None else name,
                value=n,
                text=_tokens(n),
                color=blend(STACK_COLORS[0], PINK, (i - 1) / (len(versions) - 1)),
                tip=(
                    f"{name} · {_tokens(n)} tokens in {budget:.1f} s",
                    ABOUT.get(v, RUNGS[v].about),
                    (
                        f"Measured: {_tokens(best['tokens'])} tokens in {runs} s ({best['node'].split('.')[0]}, "
                        f"peak {max(r['peak_rss_gib'] for r in best['runs']):.0f} GiB)."
                    ),
                ),
            )
        )
    return BarChart(
        x_label=f"hint corpus indexed in {budget:.1f} s (tokens)",
        bars=tuple(bars if slowest_on_top else reversed(bars)),
        color="#1b9aa3",
        x_domain=(0, 4e10),
        x_ticks=(0, 1e10, 2e10, 3e10, 4e10),
        axis="linear tokens",
        font_size=15,
    )


def short_ladder_svg(
    title: str,
    chart: BarChart,
    label_size: float = 20,
    left: float = 0,
    tick_size: float = 18,
    axis_size: float = 20,
) -> str:
    """`title` over `chart`'s bars (ladder_tokens_chart's, top first) as thick bars on a linear axis that ends at the
    longest, each with its tokens in bold: inside the bar, in white, if it fits, else just past its end. A row's title
    (its tooltip) is its bar's. The bars' labels are `label_size` px, the axis's tick labels `tick_size` px and its
    title `axis_size` px, each line under the bars as far below the one above it as it is large; the drawing starts
    `left` px in (the blank left of the longest label, at a larger size)."""
    # The bars' left and right, the first row's top and a row's height.
    x0, x1, y0, row = 230, 975, 56, 96
    bar_h, muted, grid = 60, "#75666e", "#ead9e1"
    y1 = y0 + row * len(chart.bars)
    top = max(b.value for b in chart.bars)
    # the baselines of the tick labels and of the axis's title, and the drawing's bottom
    tick_y = y1 + 12 + tick_size
    axis_y = tick_y + 16 + axis_size
    bottom = axis_y + 14

    def x(v: float) -> float:
        return x0 + (x1 - x0) * v / top

    o = [text(x0, 28, title, 30, "start", INK, "bold")]
    for v in range(0, int(top) + 1, 5_000_000_000):
        o.append(line(x(v), y0 - 10, x(v), y1, grid, 1.2))
        o.append(
            text(x(v), tick_y, token_count(v) if v else "0", tick_size, "middle", muted)
        )
    o.append(
        text((x0 + x1) / 2, axis_y, "tokens indexed", axis_size, "middle", "#4f4249")
    )
    for i, b in enumerate(chart.bars):
        ym = y0 + row * (i + 0.5)
        o.append(
            text(x0 - 16, ym + 0.35 * label_size, b.label, label_size, "end", "#4f4249")
        )
        o.append(line(x0 - 8, ym, x0, ym, muted, 1.2))
        w = max(x(b.value) - x0, 2)
        o.append(rect(x0, ym - bar_h / 2, w, bar_h, b.color or chart.color))
        # About 17 px a character of 28 px bold, and room on both sides.
        if w > 17 * len(b.text) + 40:
            o.append(text(x0 + w - 14, ym + 10, b.text, 28, "end", "#fff", "bold"))
        else:
            o.append(text(x0 + w + 14, ym + 10, b.text, 28, "start", INK, "bold"))
        o.append(rect(0, ym - row / 2, x1, row, "transparent", extra=svg_tip(*b.tip)))
    return (
        f"<svg viewBox='{left:g} 0 {1000 - left:g} {bottom:g}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='{escape(chart.x_label, quote=True)}'>" + "".join(o) + "</svg>"
    )


def waterfall_svg(title: str, chart: BarChart) -> str:
    """`title` over `chart`'s bars (ladder_tokens_chart's, the record's first and in its color), with no axis but the
    gridlines: each bar's tokens in bold, inside it in white if they fit, else just past its end, and an arrow up from
    each bar's tokens to the one before's, with how many times as many tokens that one has. A row's title (its
    tooltip) is its bar's."""
    # The bars' left and right, the first row's top and a row's height.
    x0, x1, y0, row = 230, 975, 60, 124
    bar_h, grid = 52, "#ead9e1"
    bars = chart.bars
    y1 = y0 + row * len(bars)
    top = max(b.value for b in bars)

    def x(v: float) -> float:
        return x0 + (x1 - x0) * v / top

    o = [text(x0, 28, title, 30, "start", INK, "bold")]
    for v in range(5_000_000_000, int(top) + 1, 5_000_000_000):
        o.append(line(x(v), y0 - 10, x(v), y1, grid, 1.2))
    o.append(line(x0, y0 - 10, x0, y1, INK, 1.4))
    labels = []  # each bar's tokens: their left, right and middle row, and whether they are inside the bar
    for i, b in enumerate(bars):
        ym, first = y0 + row * (i + 0.5), i == 0
        color = b.color or chart.color
        o.append(
            text(
                x0 - 16,
                ym + 7,
                b.label,
                20,
                "end",
                color if first else "#4f4249",
                "bold" if first else "normal",
            )
        )
        w = max(x(b.value) - x0, 2)
        o.append(rect(x0, ym - bar_h / 2, w, bar_h, color))
        # About 19 px a character of 28 px bold, and room on both sides.
        width = 19 * len(b.text)
        if inside := w > width + 40:
            o.append(text(x0 + w - 14, ym + 10, b.text, 28, "end", "#fff", "bold"))
            labels.append((x0 + w - 14 - width, x0 + w - 14, ym, inside))
        else:
            o.append(text(x0 + w + 14, ym + 10, b.text, 28, "start", INK, "bold"))
            labels.append((x0 + w + 14, x0 + w + 14 + width, ym, inside))
        o.append(rect(0, ym - row / 2, x1, row, "transparent", extra=svg_tip(*b.tip)))
    for (a, b), (u, l) in zip(itertools.pairwise(bars), itertools.pairwise(labels)):
        # One bend from the right of the lower tokens (their top right corner, if a line from their right would be
        # steep) into a head of its own under the upper's (under the bar, if inside it): it leaves flatter than the
        # straight line between them (never downward) and arrives steeper (at least 60° up), along which the head
        # points (a marker's head leaves the thick line's end showing past its tip), its handle short enough to stay
        # above the start. The factor below and right of its middle, outside the bend.
        sx, sy = (u[0] + u[1]) / 2, u[2] + (bar_h / 2 + 6 if u[3] else 23)
        px, py = l[1] + 12, l[2]
        if math.atan2(sy - py, sx - px) < -math.pi / 4:
            px, py = l[1] + 1, l[2] - 18
        head_l, head_w = 16, 8
        n = math.hypot(sx - px, sy - py)
        chord = math.atan2(sy - py, sx - px)
        out, into = min(chord + 0.25, 0.0), min(chord - 0.25, -math.pi / 3)
        ux, uy = math.cos(into), math.sin(into)
        bx, by = sx - head_l * ux, sy - head_l * uy  # the head's base
        handle = min(0.4 * n, 0.8 * (py - by) / -uy)
        c1 = (px + 0.4 * n * math.cos(out), py + 0.4 * n * math.sin(out))
        c2 = (bx - handle * ux, by - handle * uy)
        o.append(
            f"<path d='M{px:.1f},{py:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {bx:.1f},{by:.1f}' "
            f"stroke='{INK}' stroke-width='2.8' fill='none' stroke-linecap='round'/>"
        )
        o.append(
            f"<path d='M{sx:.1f},{sy:.1f} L{bx - head_w * uy:.1f},{by + head_w * ux:.1f} "
            f"L{bx + head_w * uy:.1f},{by - head_w * ux:.1f} z' fill='{INK}'/>"
        )
        mx = 0.125 * px + 0.375 * (c1[0] + c2[0]) + 0.125 * bx
        my = 0.125 * py + 0.375 * (c1[1] + c2[1]) + 0.125 * by
        o.append(text(mx + 14, my + 32, f"{a.value / b.value:.2f}×", 24, "start", INK))
    return (
        f"<svg viewBox='0 0 1000 {y1 + 10}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='{escape(title, quote=True)}'>" + "".join(o) + "</svg>"
    )


def ladder_run_chart() -> BarChart:
    """The estimated run time of the record with each version's build (ladder.run_estimate)."""
    bars = []
    for i, (v, ts) in enumerate(measured(LADDER).items(), 1):
        if not ts:
            continue
        e = run_estimate(ts)
        bars.append(
            Bar(
                label=f"{i}. {NAMES[v]}",
                value=e["run_s"],
                text=f"{e['run_s']:.1f} s",
                tip=(
                    f"{NAMES[v]} · about {e['run_s']:.1f} s",
                    (
                        f"Hint corpus: {e['shards']} shards ({_tokens(e['tokens'])} tokens), indexed by "
                        f"{e['build_s']:.1f} s after the clock starts; training with it takes about {e['run_s']:.1f} s."
                    ),
                    "",
                ),
            )
        )
    return BarChart(
        x_label="estimated seconds to the target loss",
        bars=tuple(bars),
        color="#c98500",
        x_domain=(0, 45),
        x_ticks=(0, 10, 20, 30, 40),
        rule=(40.0, "no hint corpus: 40 s"),
        axis="linear",
    )


# The one-pass section's builds, slowest first, and their colors.
ONE_PASS_BUILDS = ["one pass", "two pass", "hash only"]
ONE_PASS_COLORS = {"one pass": "#b0302a", "two pass": "#1b9aa3", "hash only": "#75666e"}


def gap_svg(results: list[dict], builds: list[str]) -> str:
    """The `builds`' median seconds by hint-corpus size (one_pass.combined's), slowest first, both axes logarithmic (the
    sizes halve, so they are evenly spaced); every point labelled with its time (above the top line's, below the
    bottom line's, above left of the others'), and at every size a double arrow between each two builds next to each
    other, with the time saved going from the slower to the faster. A point's or an arrow's title (its tooltip) has
    the numbers."""
    rs = sorted(results, key=lambda r: r["tokens"])
    colors = ONE_PASS_COLORS
    top, bottom = builds[0], builds[-1]
    x0, x1, y0, y1 = 90, 860, 24, 430  # the plot's left, right, top and bottom
    sizes = [r["tokens"] for r in rs]
    lx0, lx1 = math.log(sizes[0] / 1.3), math.log(sizes[-1] * 1.3)
    ticks = (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200)
    lo = max(t for t in ticks if t <= min(r["median"][bottom] for r in rs) / 1.7)
    hi = min(t for t in ticks if t >= max(r["median"][top] for r in rs) * 1.7)
    ly0, ly1 = math.log(lo), math.log(hi)
    r_dot = 4.5
    halo = (
        "paint-order='stroke' stroke='#fffbfd' stroke-width='4' stroke-linejoin='round'"
    )

    def x(n: float) -> float:
        return x0 + (x1 - x0) * (math.log(n) - lx0) / (lx1 - lx0)

    def y(v: float) -> float:
        return y1 - (y1 - y0) * (math.log(v) - ly0) / (ly1 - ly0)

    muted, grid = "#75666e", "#ead9e1"
    mark = "gap-" + "-".join(builds).replace(" ", "-")
    # An arrowhead 9 px long whose tip is the path's end (refX at the tip), in pixels whatever the stroke.
    head = (
        f"<defs><marker id='{mark}' viewBox='0 0 10 10' refX='10' refY='5' markerWidth='9' markerHeight='9' "
        "markerUnits='userSpaceOnUse' orient='auto-start-reverse'>"
        f"<path d='M0,0.5 L10,5 L0,9.5 z' fill='{INK}'/></marker></defs>"
    )
    o = [head]
    for v in (t for t in ticks if lo <= t <= hi):
        o.append(line(x0, y(v), x1, y(v), grid))
        o.append(text(x0 - 10, y(v) + 5, f"{v:g} s", 15, "end", muted))
    for n in sizes:
        o.append(line(x(n), y0, x(n), y1, grid))
        o.append(text(x(n), y1 + 24, token_count(n), 15, "middle", muted))
    o.append(line(x0, y1, x1, y1, INK, 1.4))
    o.append(
        text(
            (x0 + x1) / 2,
            y1 + 52,
            "tokens in the hint corpus (log scale; times on a log scale too)",
            15,
            "middle",
            "#4f4249",
        )
    )
    for b in builds:
        pts = [(x(r["tokens"]), y(r["median"][b])) for r in rs]
        o.append(
            "<polyline points='"
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            + f"' fill='none' stroke='{colors[b]}' stroke-width='2.6' stroke-linejoin='round'/>"
        )
    for slow, fast in itertools.pairwise(builds):
        for r in rs:
            a, b = r["median"][slow], r["median"][fast]
            px, ya, yb = x(r["tokens"]), y(a), y(b)
            cut = f"−{100 * (1 - b / a):.0f}%"
            # The arrow's tips touch the two dots' edges.
            o.append(
                f"<path d='M{px:.1f},{ya + r_dot:.1f} V{yb - r_dot:.1f}' stroke='{INK}' stroke-width='1.6' "
                f"fill='none' marker-start='url(#{mark})' marker-end='url(#{mark})'/>"
            )
            o.append(
                text(
                    px + 8, (ya + yb) / 2 + 6, cut, 17, "start", INK, "bold", extra=halo
                )
            )
            tip = svg_tip(
                f"{token_count(r['tokens'])} tokens",
                f"{slow}: {a:.3g} s → {fast}: {b:.3g} s",
                f"{cut} ({a / b:.1f}× faster)",
            )
            o.append(line(px, ya, px, yb, "transparent", 20, tip))
    for b in builds:
        for r in rs:
            px, py = x(r["tokens"]), y(r["median"][b])
            tip = svg_tip(
                f"{b}, {token_count(r['tokens'])} tokens", f"{r['median'][b]:.3g} s"
            )
            o.append(
                f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{r_dot}' fill='{colors[b]}'/>"
                f"<circle cx='{px:.1f}' cy='{py:.1f}' r='10' fill='transparent' {tip}/>"
            )
            # Above the top line, below the bottom one, and above left of a middle one (its line comes in from
            # below left, its arrows leave up and down).
            tx, ty, anchor = (
                (px, py - 12, "middle")
                if b == top
                else (px, py + 24, "middle")
                if b == bottom
                else (px - 9, py - 8, "end")
            )
            o.append(
                text(
                    tx, ty, f"{r['median'][b]:.3g} s", 14, anchor, colors[b], extra=halo
                )
            )
        last = rs[-1]["median"][b]
        o.append(
            text(x(sizes[-1]) + 14, y(last) + 5, b, 15, "start", colors[b], "bold")
        )
    return (
        f"<svg viewBox='0 0 1000 {y1 + 64}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='{', '.join(builds)}'>{''.join(o)}</svg>"
    )


def gap_caption(results: list[dict], builds: list[str]) -> str:
    rs = sorted(results, key=lambda r: r["tokens"])
    nodes = ", ".join(sorted({n for r in rs for n in r["nodes"]}))
    ranges = []
    for slow, fast in itertools.pairwise(builds):
        cuts = [1 - r["median"][fast] / r["median"][slow] for r in rs]
        ranges.append(
            f"from {slow} to {fast}, −{100 * min(cuts):.0f}% to −{100 * max(cuts):.0f}%"
        )
    return (
        f"Each arrow: the time saved at that size going from the build above to the one below it: {'; '.join(ranges)}. "
        "On the log time axis an arrow's length is the ratio of the two times. Each point is the median of "
        f"{rs[0]['builds']} builds ({rs[0]['jobs']} jobs of {rs[0]['builds'] // rs[0]['jobs']}, each on a worker of "
        "its own with 800 GB or more; in a job the builds run in turn, each in a fresh process); only the build is "
        "timed. The sizes are the record's 37B halved, in whole shards. The one-pass build uses plain stores, "
        "prefetches the fill counts and table lines 32 entries ahead, and loses about one entry in a million to "
        f"threads that write the same slot at once. Measured by one_pass.py on {nodes}; hover over a point or an arrow for the numbers."
    )


def _tokens(n: float) -> str:
    """3 significant digits, e.g. 83.1M, 2.74B."""
    unit, div = ("B", 1e9) if n >= 1e9 else ("M", 1e6)
    return f"{n / div:.3g}{unit}"


@dataclass(frozen=True, slots=True)
class Record:
    number: int
    seconds: float
    description: str
    date: str  # MM/DD/YY


def records() -> list[Record]:
    """The world records of README.md's track 1 table, in order; its re-timings of a record repeat its number."""
    readme = (final_results_root().parent / "README.md").read_text(encoding="utf-8")
    table = readme.split("## World record history", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(
        r"^(\d+) \| ([\d.]+) minutes \| (.*?)\s*\| (\d\d/\d\d/\d\d) \|",
        table,
        re.MULTILINE,
    )
    return [
        Record(
            int(number),
            60 * float(minutes),
            " ".join(re.sub(r"\[(.*?)\]\([^)]*\)", r"\1", description).split()),
            date,
        )
        for number, minutes, description, date in rows
    ]


def improvements(records: list[Record]) -> list[tuple[Record, Record]]:
    """Each record with the one it beat: a re-timing replaces the record beaten, rather than being a record."""
    return [
        (beaten, record)
        for beaten, record in itertools.pairwise(records)
        if record.number != beaten.number
    ]


def stacked(records: list[Record]) -> tuple[list[tuple[Record, Record]], float]:
    """The improvements of STACKED_YEARS, and their combined reduction (%): the slower re-timings do not count."""
    pairs = [p for p in improvements(records) if p[1].date.endswith(STACKED_YEARS)]
    return pairs, 100 * (1 - math.prod(r.seconds / b.seconds for b, r in pairs))


def records_chart(records: list[Record]) -> alt.TopLevelMixin:
    """Two bars on a log speedup axis: the records of STACKED_YEARS stacked (oldest at the bottom, a band per record,
    with STACK_DATES written inside), and the new record. The tooltips have the records' descriptions."""
    old, new = NEW_RECORD
    pairs, total = stacked(records)
    stack, mine = "All records since Jan '25", "Retrieval-augmented n-grams"
    speedups = list(
        itertools.accumulate(
            (b.seconds / r.seconds for b, r in pairs), operator.mul, initial=1.0
        )
    )

    def band(
        bar: str, record: str, seconds: tuple[float, float], top: float, color: str
    ) -> dict:
        return {
            "bar": bar,
            "top": top,
            "color": color,
            "record": record,
            "time": f"{seconds[0]:.4g} s → {seconds[1]:.4g} s",
            "reduction": f"−{100 * (1 - seconds[1] / seconds[0]):.1f}%",
        }

    # Every band reaches down to ×1, newest first, so that each older band covers the rest of the one before it
    # (bands that only meet leave antialiased seams).
    values = [
        band(
            stack,
            f"#{r.number} ({r.date}): {r.description}",
            (b.seconds, r.seconds),
            speedups[i + 1],
            STACK_COLORS[i % 2],
        )
        for i, (b, r) in reversed(list(enumerate(pairs)))
    ]
    values.append(band(mine, "the new record (#367)", NEW_RECORD, old / new, PINK))
    totals = [
        {
            "bar": stack,
            "top": speedups[-1],
            "text": f"−{total:.1f}%",
            "color": "#2b2025",
        },
        {
            "bar": mine,
            "top": old / new,
            "text": f"−{100 * (1 - new / old):.1f}%",
            "color": PINK,
        },
    ]
    dates = [
        {
            "bar": stack,
            "at": speedups[
                sum(
                    1
                    for _, r in pairs
                    if (int("20" + r.date[-2:]), int(r.date[:2])) < (year, month)
                )
            ],
            "text": text,
        }
        for year, month, text in STACK_DATES
    ]
    x = alt.X(
        "bar:N",
        sort=[stack, mine],
        title=None,
        axis=alt.Axis(labelAngle=0, labelFontSize=19, labelLimit=360, ticks=False),
    )
    y_scale = alt.Scale(type="log", domain=[1, 6.3])
    # Any: ty cannot follow the chart methods Altair generates (mark_*, encode)
    bars: Any = alt.Chart(alt.Data(values=values)).mark_bar(
        width=alt.RelativeBandSize(0.58)
    )
    bars = bars.encode(
        x=x,
        y=alt.Y(
            "top:Q",
            scale=y_scale,
            title="speedup",
            axis=alt.Axis(
                values=[1, 2, 3, 4, 5],
                labelExpr="'×' + datum.value",
                labelFontSize=18,
                titleFontSize=16,
            ),
        ),
        y2=alt.datum(1),
        color=alt.Color("color:N", scale=None),
        tooltip=[
            alt.Tooltip("record:N", title="record"),
            alt.Tooltip("time:N", title="time"),
            alt.Tooltip("reduction:N", title="reduction"),
        ],
    )
    text: Any = alt.Chart(alt.Data(values=totals)).mark_text(
        baseline="bottom", dy=-6, fontSize=26, fontWeight=700
    )
    text = text.encode(
        x=x,
        y=alt.Y("top:Q", scale=y_scale),
        text="text:N",
        color=alt.Color("color:N", scale=None),
    )
    # A date sits just above its spot, or just below it at the stack's top.
    labels = []
    for at_top in (False, True):
        label: Any = alt.Chart(
            alt.Data(values=[d for d in dates if (d["at"] == speedups[-1]) == at_top])
        ).mark_text(
            color="#ffffff",
            fontSize=17,
            fontWeight=600,
            baseline="top" if at_top else "bottom",
            dy=8 if at_top else -7,
        )
        labels.append(label.encode(x=x, y=alt.Y("at:Q", scale=y_scale), text="text:N"))
    return alt.layer(bars, text, *labels).properties(
        width="container", height=380, usermeta={"report": {"kind": "bar"}}
    )


def records_caption(records: list[Record]) -> str:
    pairs, total = stacked(records)
    b, last = pairs[0][0], pairs[-1][1]
    return (
        f"From README.md's record times, each over the record it beat. The stack is the {len(pairs)} records since "
        f"Jan 2025, one band per record: −{total:.1f}% (#{b.number}'s {b.seconds:.4g} s → #{last.number}'s "
        f"{last.seconds:.4g} s, without the slower re-timing of #21). Each date marks where the stack stood at the "
        "start of that month."
    )


def reductions_chart(
    records: list[Record], top: int = TOP_RECORDS, log_y: bool = False
) -> alt.TopLevelMixin:
    """A vertical bar per improvement, largest first: the new record, the stacked records of STACKED_YEARS (newest
    first, each record's segment its share of the combined reduction, in proportion to the log of its speedup), then the
    `top` largest single records, in the places of TOP_RECORDS (the others' left empty). A label's second line is a
    short name; the tooltips have the descriptions. On a log y axis the bars start at LOG_FLOOR, and a stack's segments
    keep their values, not their heights."""
    old, new = NEW_RECORD
    pairs, total = stacked(records)
    logs = sum(math.log(b.seconds / r.seconds) for b, r in pairs)
    count = f"{len(pairs)}/{len({r.number for r in records})}"  # of all the records
    largest = sorted(improvements(records), key=lambda p: p[1].seconds / p[0].seconds)[
        :TOP_RECORDS
    ]

    def segment(
        bar: str, record: str, seconds: tuple[float, float], value: float, color: str
    ) -> dict:
        return {
            "bar": bar,
            "value": value,
            "color": color,
            "record": record,
            "time": f"{seconds[0]:.4g} s → {seconds[1]:.4g} s",
            "reduction": f"−{100 * (1 - seconds[1] / seconds[0]):.1f}%",
        }

    def improvement(
        bar: str, beaten: Record, record: Record, value: float, color: str
    ) -> dict:
        name = f"#{record.number} ({record.date}): {record.description}"
        return segment(bar, name, (beaten.seconds, record.seconds), value, color)

    values = [
        segment(
            f"New record\n{old:.3g} → {new:.3g} s",
            "the new record (#367)",
            NEW_RECORD,
            100 * (1 - new / old),
            PINK,
        )
    ]
    values += [
        improvement(
            f"All of 2025–26\n{count} records",
            b,
            r,
            total * math.log(b.seconds / r.seconds) / logs,
            STACK_COLORS[i % 2],
        )
        for i, (b, r) in enumerate(reversed(pairs))
    ]
    values += [
        improvement(
            f"#{r.number} (20{r.date[-2:]})\n{SHORT_NAMES.get(r.number, '')}".strip(),
            b,
            r,
            100 * (1 - r.seconds / b.seconds),
            BLUE,
        )
        for b, r in largest
    ]
    slots = list(dict.fromkeys(v["bar"] for v in values))
    labels = slots[: 2 + top]
    values = [v for v in values if v["bar"] in labels]
    # Stacked here, not by Vega-Lite, whose stacks start at 0 (off a log axis): the stack's records newest first,
    # from the axis.
    ends: dict[str, float] = {}
    for value in values:
        start = ends.get(value["bar"], 0)
        ends[value["bar"]] = value["end"] = start + value["value"]
        value["start"] = max(start, LOG_FLOOR) if log_y else start
    totals = [
        {"bar": bar, "total": t, "text": f"−{t:.1f}%"}
        for bar in labels
        for t in [sum(v["value"] for v in values if v["bar"] == bar)]
    ]
    x = alt.X(
        "bar:N",
        title=None,
        # a label's lines, split at its newline
        axis=alt.Axis(
            values=labels,
            labelAngle=0,
            labelLimit=120,
            ticks=False,
            labelExpr="split(datum.label, '\\n')",
        ),
        scale=alt.Scale(domain=slots),
    )
    y_scale = (
        alt.Scale(type="log", domain=[LOG_FLOOR, 100])
        if log_y
        else alt.Scale(domain=[0, 100])
    )
    # Any: ty cannot follow the chart methods Altair generates (mark_*, encode)
    bars: Any = alt.Chart(alt.Data(values=values)).mark_bar(
        stroke="#fffbfd",
        strokeWidth=0.4,  # the cards' --surface (report.py)
    )
    bars = bars.encode(
        x=x,
        y=alt.Y(
            "end:Q",
            scale=y_scale,
            title="time reduction over the record beaten (%)",
        ),
        y2="start:Q",
        color=alt.Color("color:N", scale=None),
        tooltip=[
            alt.Tooltip("record:N", title="record"),
            alt.Tooltip("time:N", title="time"),
            alt.Tooltip("reduction:N", title="reduction"),
        ],
    )
    text: Any = alt.Chart(alt.Data(values=totals)).mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        color="#4f4249",
        fontSize=11,  # report.py's --ink-2
    )
    text = text.encode(x=x, y=alt.Y("total:Q", scale=y_scale), text="text:N")
    return alt.layer(bars, text).properties(
        width="container", height=340, usermeta={"report": {"kind": "bar"}}
    )


def reductions_caption(records: list[Record]) -> str:
    pairs, total = stacked(records)
    b, last = pairs[0][0], pairs[-1][1]
    return (
        f"From README.md's record times, each over the record it beat. The stacked bar is −{total:.1f}% "
        f"(#{b.number}'s {b.seconds:.4g} s → #{last.number}'s {last.seconds:.4g} s, without the slower re-timings "
        f"of #21); each record's segment is its share, in proportion to the log of its speedup. Then the "
        f"{TOP_RECORDS} largest single records. On the log axis the bars start at {LOG_FLOOR}%, and the stacked "
        "bar's lower segments (the newest records) look larger than their shares."
    )


def svg_tip(title: str, *lines: str) -> str:
    """An svg_card element's tooltip (report-charts.js): its title, then a paragraph per line."""
    return f"data-tip-title='{escape(title)}' data-tip='{escape(chr(10).join(lines))}'"


def record_tip(beaten: Record, record: Record) -> str:
    return svg_tip(*record_tip_lines(beaten, record))


def record_tip_lines(beaten: Record, record: Record) -> tuple[str, ...]:
    """A record's tooltip: its title, then its paragraphs. A re-timing (the same record again) has no reduction."""
    if record.number == beaten.number:
        return (
            f"#{record.number} re-timed ({record.date})",
            record.description,
            f"{beaten.seconds:.4g} s → {record.seconds:.4g} s",
        )
    return (
        f"#{record.number} ({record.date})",
        record.description,
        f"{beaten.seconds:.4g} s → {record.seconds:.4g} s (−{100 * (1 - record.seconds / beaten.seconds):.1f}%)",
    )


def new_record_tip() -> str:
    return svg_tip(*new_record_tip_lines())


def new_record_tip_lines() -> tuple[str, ...]:
    old, new = NEW_RECORD
    return (
        "Retrieval-augmented n-grams (#367)",
        f"{old:.4g} s → {new:.4g} s (−{100 * (1 - new / old):.1f}%)",
    )


# The ruler: record times from RULER[0] s down to RULER[1] s, on a log axis from x = 50 to 950.
RULER = (240, 6)
RULER_TICKS = (200, 150, 100, 70, 50, 30, 20, 15, 10, 7)  # seconds
RULER_Y, RULER_H = 100, 54  # the bar's top and height


def ruler_x(seconds: float) -> float:
    return 50 + 900 * math.log(RULER[0] / seconds) / math.log(RULER[0] / RULER[1])


def curly(x1: float, x2: float, y: float, h: float = 18) -> str:
    """A curly brace over [x1, x2], its ends down at y, its point up at y - h."""
    m, r, c = (x1 + x2) / 2, min(10, abs(x2 - x1) / 4), y - h / 2
    return (
        f"<path d='M{x1:.1f},{y} Q{x1:.1f},{c} {x1 + r:.1f},{c} L{m - r:.1f},{c} Q{m:.1f},{c} {m:.1f},{y - h} "
        f"Q{m:.1f},{c} {m + r:.1f},{c} L{x2 - r:.1f},{c} Q{x2:.1f},{c} {x2:.1f},{y}' stroke='{INK}' "
        "stroke-width='1.5' fill='none' stroke-linecap='round'/>"
    )


def records_ruler_svg(records: list[Record]) -> str:
    """The record times on a log ruler: a band per record of STACKED_YEARS (its own times, so #21's slower re-timing
    shows as a sliver of overlap), then the new record's span; a brace over each with its reduction, and dates under.
    A band's title (its tooltip) is its record."""
    old, new = NEW_RECORD
    pairs, _ = stacked(records)
    first, last = pairs[0][0], pairs[-1][1]
    end_of_25 = [r for _, r in pairs if r.date.endswith("25")][-1]
    y, h, x = RULER_Y, RULER_H, ruler_x
    axis = y + h + 46
    o = []
    for t in RULER_TICKS:
        o.append(line(x(t), y - 8, x(t), axis, "#ead9e1"))
        o.append(text(x(t), axis + 23, f"{t} s", 18, "middle", "#75666e"))
    o.append(line(50, axis, 964, axis, INK, 1.4))
    o.append(
        f"<path d='M957,{axis - 6} L966,{axis} L957,{axis + 6}' stroke='{INK}' stroke-width='1.4' fill='none'/>"
    )
    o.append(text(966, axis + 52, "record time", 16, "end", "#4f4249"))
    for i, (b, r) in enumerate(pairs):
        o.append(
            rect(
                x(b.seconds),
                y,
                max(x(r.seconds) - x(b.seconds), 0.6),
                h,
                STACK_COLORS[i % 2],
                extra=record_tip(b, r),
            )
        )
    o.append(rect(x(old), y, x(new) - x(old), h, PINK, extra=new_record_tip()))
    o.append(
        text(
            (x(old) + x(new)) / 2,
            y + h / 2 + 7,
            "retrieval-augmented n-gram hints",
            19,
            "middle",
            "#ffffff",
            "bold",
        )
    )
    for t, label in (
        (first.seconds, "Jan '25"),
        (end_of_25.seconds, "end of '25"),
        (old, "Sep '26"),  # where the pink starts, as it covers the bands' last 1%
        (new, "Oct '26"),
    ):
        o.append(line(x(t), y + h, x(t), y + h + 14, INK, 1.4))
        o.append(text(x(t), y + h + 35, label, 18, "middle", "#75666e"))
    for (a, b), reduction, color in (
        (
            (x(first.seconds), x(old) - 2),
            1 - last.seconds / first.seconds,
            INK,
        ),
        ((x(old) + 2, x(new)), 1 - new / old, PINK),
    ):
        o.append(curly(a, b, y - 12))
        o.append(
            text(
                (a + b) / 2,
                y - 40,
                f"−{100 * reduction:.1f}%",
                26,
                "middle",
                color,
                "bold",
            )
        )
    return (
        f"<svg viewBox='0 0 1000 {axis + 64}' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='The record time on a log axis'>{''.join(o)}</svg>"
    )


def records_ruler_caption(records: list[Record]) -> str:
    pairs, total = stacked(records)
    first, last = pairs[0][0], pairs[-1][1]
    return (
        f"README.md's record times: the records since Jan 2025 take #{first.number}'s {first.seconds:.4g} s to "
        f"#{last.number}'s {last.seconds:.4g} s (−{100 * (1 - last.seconds / first.seconds):.1f}%, less than the "
        f"−{total:.1f}% above, as here the slower re-timing of #21 counts), and the new record the previous WR's "
        f"{NEW_RECORD[0]} s to {NEW_RECORD[1]} s. Equal lengths are equal speedups. Hover over a band for its record."
    )


# The timeline: dates from TIMELINE_DATES[0] to [1], on x = 100 to 820; record times from TIMELINE_SECONDS[0] s down
# to [1] s, on a log axis from y = 30 to 400.
TIMELINE_DATES = (date(2024, 12, 15), date(2026, 10, 25))
TIMELINE_SECONDS = (250, 5.5)
NEW_RECORD_DATE = date(2026, 10, 1)


def timeline_x(day: date) -> float:
    (d0, d1) = TIMELINE_DATES
    return 100 + 720 * (day - d0).days / (d1 - d0).days


def timeline_y(seconds: float) -> float:
    (s0, s1) = TIMELINE_SECONDS
    return 30 + 370 * math.log(s0 / seconds) / math.log(s0 / s1)


def curly_right(x: float, y1: float, y2: float, w: float = 16) -> str:
    """A curly brace beside [y1, y2], its ends left at x, its point right at x + w."""
    m, r, c = (y1 + y2) / 2, min(10, abs(y2 - y1) / 4), x + w / 2
    return (
        f"<path d='M{x:.1f},{y1:.1f} Q{c:.1f},{y1:.1f} {c:.1f},{y1 + r:.1f} L{c:.1f},{m - r:.1f} Q{c:.1f},{m:.1f} "
        f"{x + w:.1f},{m:.1f} Q{c:.1f},{m:.1f} {c:.1f},{m + r:.1f} L{c:.1f},{y2 - r:.1f} Q{c:.1f},{y2:.1f} "
        f"{x:.1f},{y2:.1f}' stroke='{INK}' stroke-width='1.5' fill='none' stroke-linecap='round'/>"
    )


def records_timeline_svg(records: list[Record]) -> str:
    """The world record over time on a log axis: a step per record of STACKED_YEARS (its own times, so #21's slower
    re-timing steps up), labelled, then the new record's drop; on the right, a brace over each with its reduction, as
    in the animation's finished chart. A dot's title (its tooltip) is its record."""
    old, new = NEW_RECORD
    pairs, _ = stacked(records)
    first, last = pairs[0][0], pairs[-1][1]
    x, y = timeline_x, timeline_y
    blue, muted = STACK_COLORS[0], "#75666e"

    def day(r: Record) -> date:
        month, d, year = map(int, r.date.split("/"))
        return date(2000 + year, month, d)

    o = []
    for t in (200, 100, 50, 20, 10):
        o.append(line(100, y(t), 820, y(t), "#ead9e1"))
        o.append(text(88, y(t) + 6, f"{t} s", 18, "end", muted))
    d0, d1 = TIMELINE_DATES
    for year, month in itertools.product((2025, 2026), (1, 4, 7, 10)):
        if date(year, month, 1) > d1:
            continue
        mx = x(date(year, month, 1))
        o.append(line(mx, 30, mx, 400, "#ead9e1"))
        name = date(year, month, 1).strftime("%b '%y")
        o.append(text(mx, 428, name, 18, "middle", muted))
    o.append(line(100, 400, 820, 400, INK, 1.4))
    # The blue line's label, under it and centered over the plot: its cap height (15 px) at 240 s, as in the animation's
    # finished chart
    o.append(
        text(
            460,
            y(240) + 15,
            "all improvements since 2025",
            21,
            "middle",
            blue,
            "bold",
        )
    )
    points = [(x(d0), y(first.seconds))]
    for b, r in pairs:
        points += [(x(day(r)), y(b.seconds)), (x(day(r)), y(r.seconds))]
    points.append((x(NEW_RECORD_DATE), y(last.seconds)))
    o.append(
        "<polyline points='"
        + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        + f"' fill='none' stroke='{blue}' stroke-width='2.4' stroke-linejoin='round'/>"
    )
    for _, r in pairs:
        o.append(
            f"<circle cx='{x(day(r)):.1f}' cy='{y(r.seconds):.1f}' r='3.6' fill='{blue}' stroke='#fffbfd' "
            "stroke-width='1'/>"
        )
    xm = x(NEW_RECORD_DATE)
    o.append(line(xm, y(old), xm, y(new), PINK, 5, "stroke-linecap='round'"))
    o.append(
        f"<circle cx='{xm:.1f}' cy='{y(new):.1f}' r='6.5' fill='{PINK}' stroke='#fffbfd' stroke-width='1.5'/>"
    )
    # Wider targets for the tooltips than the dots and the drop themselves
    for b, r in pairs:
        o.append(
            f"<circle cx='{x(day(r)):.1f}' cy='{y(r.seconds):.1f}' r='10' fill='transparent' "
            f"{record_tip(b, r)}/>"
        )
    o.append(
        line(xm, y(old), xm, y(new), "transparent", 18, new_record_tip())
        + f"<circle cx='{xm:.1f}' cy='{y(new):.1f}' r='12' fill='transparent' {new_record_tip()}/>"
    )
    for i, name in enumerate(("retrieval-augmented", "n-gram hints")):
        o.append(
            text(
                xm - 14,
                y(17) - 2 + 26 * i,
                name,
                21,
                "end",
                PINK,
                "bold",
                # a halo of the card's background, over the gridlines
                extra="paint-order='stroke' stroke='#fffbfd' stroke-width='4' stroke-linejoin='round'",
            )
        )
    bx = 840
    for (y1, y2), reduction, color in (
        ((y(first.seconds), y(old) - 2), 1 - last.seconds / first.seconds, blue),
        ((y(old) + 2, y(new)), 1 - new / old, PINK),
    ):
        o.append(curly_right(bx, y1, y2))
        o.append(
            text(
                bx + 26,
                (y1 + y2) / 2 + 9,
                f"−{100 * reduction:.1f}%",
                26,
                "start",
                color,
                "bold",
            )
        )
    return (
        "<svg viewBox='0 0 1000 446' xmlns='http://www.w3.org/2000/svg' role='img' "
        f"aria-label='The world record over time'>{''.join(o)}</svg>"
    )


def records_timeline_caption() -> str:
    return (
        "README.md's record times, a step per record since Jan 2025; the small step up in May 2025 is the slower "
        "re-timing of #21. Hover over a dot for its record."
    )


def records_animation_data(records: list[Record]) -> dict[str, Any]:
    """The animation's data: README.md's rows for the records of STACKED_YEARS (from the first record they beat,
    re-timings included), then the new record's drop from NEW_RECORD's re-timed previous record, on NEW_RECORD_DATE;
    and the tooltips of the dots (each row's but the first, which has none) and of the drop."""
    pairs, _ = stacked(records)
    rows = records[records.index(pairs[0][0]) : records.index(pairs[-1][1]) + 1]
    return {
        "rows": [[r.number, round(r.seconds, 3), r.date] for r in rows],
        "previousRetimed": NEW_RECORD[0],
        "newSeconds": NEW_RECORD[1],
        "newDate": NEW_RECORD_DATE.strftime("%m/%d/%y"),
        "tips": [record_tip_lines(b, r) for b, r in itertools.pairwise(rows)],
        "newTip": new_record_tip_lines(),
    }


def animation_json(records: list[Record]) -> str:
    """records_animation_data as an animation-card's inline JSON."""
    return json.dumps(records_animation_data(records), separators=(",", ":")).replace(
        "</", "<\\/"
    )


ANIMATION_CAPTION = (
    "The same records, drawn as a line that starts fast and slows down all the way to the bottom of the new record's "
    "drop; the y axis follows the drop down, then zooms out to the whole chart. It plays while on screen and loops; "
    "<b>⬇ Video</b> saves one loop at 1920×1080 (MP4 in recent Chrome, else WebM). In the blog post: "
    "record-progression (its data)."
)


def records_animation_card(
    records: list[Record],
    title: str = "The world record over time, animated",
    caption: str = ANIMATION_CAPTION,
    layout: str = "",
    width: int = 0,
) -> str:
    """The world record over time, animated by assets/improvement-animation.js (loaded once, after the cards) from the
    card's JSON (records_animation_data); in its `layout` (phone or tweet), `width` px wide, if given."""
    attrs = f' data-layout="{layout}"' if layout else ""
    style = f' style="max-width: {width}px; margin: 8px auto 4px"' if width else ""
    return (
        f'<figure class="card chart-card animation-card"{attrs}><div class="chart-head"><p class="chart-title">'
        f'{escape(title)}</p></div><canvas role="img" aria-label="{escape(title, quote=True)}"{style}></canvas>'
        f'<script type="application/json">{animation_json(records)}</script>'
        f'<figcaption class="chart-caption">{caption}</figcaption></figure>'
    )


def dedup_caption() -> str:
    """What the deduplicated validation shards of VALS leave out, with the share of the shard's tokens each drops."""
    (full,) = FinewebShards(shards="val").create()

    def dropped(val: Val) -> str:
        (kept,) = FinewebShards(shards=val).create()
        return f"{1 - len(kept.tokens) / len(full.tokens):.1%} of its tokens"

    _, amount = VALS["no near-dups"]
    return (
        "no repeats drops its documents that occur exactly in the training shards "
        f"({dropped(VALS['no repeats'])}), and no near-dups those and the documents with a word 13-gram Jaccard "
        f"similarity of at least {amount} with a training document, as Gopher deduplicates "
        f"({dropped(VALS['no near-dups'])})."
    )


def dedup_match_chart(stats: dict[str, dict[str, KGramStats]]) -> Chart:
    """The match rate per k, a line per validation shard of VALS (its color) and training-data amount (dashed: the
    previous WR's)."""
    series = []
    for label, color in zip(VALS, VAL_COLORS):
        for dashed, name in (
            (False, "37B tokens"),
            (True, "329M tokens (prev WR train tokens)"),
        ):
            s = stats[label][name]
            rate = 100 * np.array(s.matched) / np.array(s.eligible)
            series.append(
                Series(
                    f"{name.split()[0]} · {label}",
                    tuple(zip(s.ks, rate.tolist())),
                    color,
                    dashed=dashed,
                )
            )
    # in two legend columns: the 37B lines on the left, the 329M lines beside them
    return _chart(
        "val positions whose n-gram context occurred (%)", series, legend_columns=2
    )


def dedup_ratio_chart(stats: dict[str, dict[str, KGramStats]]) -> Chart:
    """The match rate against the 370 shards over that against the previous WR's training tokens per k, a line per
    validation shard of VALS."""
    series = []
    for label, color in zip(VALS, VAL_COLORS):
        big, small = (
            stats[label][name]
            for name in ("37B tokens", "329M tokens (prev WR train tokens)")
        )
        ratio = (np.array(big.matched) / np.array(big.eligible)) / (
            np.array(small.matched) / np.array(small.eligible)
        )
        series.append(Series(label, tuple(zip(big.ks, ratio.tolist())), color))
    return Chart(
        x_label="n-gram",
        y_label="37B match rate ÷ 329M match rate",
        series=tuple(series),
        log_x=True,
        x_ticks=XTICKS,
        y_domain=(0, 20),
        x_prefix="n = ",
        tooltip_decimals=1,
        value_suffix="×",
        font_size=15,
    )


def dedup_vote_chart(stats: dict[str, dict[str, KGramStats]]) -> Chart:
    """The top-1 and top-4 vote accuracy given a match against the 370 shards per k, a line per validation shard of
    VALS (its color) and n (dashed: top-4)."""
    series = []
    for label, color in zip(VALS, VAL_COLORS):
        for dashed, n in ((False, 1), (True, 4)):
            s = stats[label]["37B tokens"]
            correct = np.array([row[s.tops.index(n)] for row in s.correct])
            series.append(
                Series(
                    f"top-{n} · {label}",
                    tuple(zip(s.ks, (100 * correct / np.array(s.matched)).tolist())),
                    color,
                    dashed=dashed,
                )
            )
    return _chart("next token right, given a match (%)", series, legend_columns=2)


def match_chart(stats: dict[str, KGramStats]) -> Chart:
    """The match rate per k, a line per training-data amount."""
    series = []
    for (name, s), color in zip(stats.items(), [PURPLE, ORANGE]):
        rate = 100 * np.array(s.matched) / np.array(s.eligible)
        series.append(Series(name, tuple(zip(s.ks, rate.tolist())), color))
    return _chart("val positions whose n-gram context occurred (%)", series)


def vote_chart(stats: KGramStats, tops: tuple[int, ...] = TOPS) -> Chart:
    """The top-n vote accuracy given a match per k, a line per n of tops (some of TOPS)."""
    matched = np.array(stats.matched)
    series = []
    for n, color in zip(tops, [PURPLE, BLUE, GREEN]):
        correct = np.array([row[stats.tops.index(n)] for row in stats.correct])
        series.append(
            Series(
                f"top-{n}",
                tuple(zip(stats.ks, (100 * correct / matched).tolist())),
                color,
            )
        )
    return _chart("next token right, given a match (%)", series)


def coverage_series(stats: KGramStats) -> Series:
    """The share of validation positions with a hint per k: those whose n-gram context occurred in the training data, of
    those with a valid n-gram context (match_chart's rate, as the post's "11% at 37B" for 20-grams)."""
    rate = 100 * np.array(stats.matched) / np.array(stats.eligible)
    return Series("val tokens with a hint", tuple(zip(stats.ks, rate.tolist())), BLUE)


def vote_coverage_chart(stats: KGramStats) -> Chart:
    """The share of validation tokens that get a hint (coverage_series), and on a right axis the top vote's accuracy
    given a match per k (vote_chart's top-1 line): the post's n-gram accuracy chart."""
    top1 = vote_chart(stats, tops=(1,))
    hinted = coverage_series(stats)
    return dataclasses.replace(
        top1,
        y_label=f"{hinted.label} (%)",
        series=(hinted,),
        right=dataclasses.replace(
            top1.series[0], label="next token right, given a match"
        ),
        right_label=top1.y_label,
    )


def reliable_sections(data: Path, stats: KGramStats) -> str:
    """The post's section on the n-gram accuracy chart, twice: with the top vote only, and with the share of validation
    tokens that get a hint on the left axis and the accuracy on the right (as in the post)."""
    top1 = vote_chart(stats, tops=(1,))
    twin = vote_coverage_chart(stats)
    return post_section(
        "Top vote only",
        POST_RELIABLE_HEADING,
        POST_RELIABLE_TEXT,
        figure_card(
            "",
            write_chart(
                data / "post-kgram-accuracy.vl.json.gz",
                line_chart(top1, height=POST_CHART_HEIGHT),
            ),
            label=top1.y_label,
        ),
    ) + post_section(
        "As in the post: with the share of tokens that get a hint",
        POST_RELIABLE_HEADING,
        POST_RELIABLE_TEXT,
        figure_card(
            "",
            write_chart(
                data / "post-kgram-accuracy-coverage.vl.json.gz",
                line_chart(twin, height=POST_CHART_HEIGHT),
            ),
            label=f"{twin.y_label}, and {twin.right_label}",
        ),
    )


def val_cache_chart(
    sweep: ValCacheLosses,
    scaling: DataScalingLosses | None = None,
    target: bool = False,
    reach: float = 1.0,
) -> Chart:
    """The validation loss per hint-corpus size, and per training-data size without hints (`scaling`, if given), with
    no hints as a dashed line and, if `target`, the target loss as a dotted one, both from the smallest size to `reach`
    times the largest (past it, the falling line ends clearly under them), on a log axis ticked at the hint-corpus
    sizes."""
    tokens = sweep.tokens
    ends = (tokens[0], tokens[-1] * reach)
    trained = (
        ()
        if scaling is None
        else (
            Series(
                "training data",
                # at the nominal sizes: the runs train on whole steps, within 0.1% of them
                tuple((n, r.val_loss) for n, r in zip(scaling.tokens, scaling.losses)),
                GREEN,
            ),
        )
    )
    return Chart(
        x_label=(
            "hint corpus tokens N"
            if scaling is None
            else "hint corpus or training tokens N"
        ),
        y_label="val loss",
        series=(
            Series(
                f"hint corpus ({_tokens(RECORD_367[1])} transformer tokens)",
                tuple(zip(tokens, sweep.losses)),
                BLUE,
            ),
            *trained,
            Series(
                "no hint corpus",
                tuple((x, sweep.no_hints) for x in ends),
                ORANGE,
                dashed=True,
            ),
            *(
                (
                    Series(
                        f"target loss ({TARGET_LOSS})",
                        tuple((x, TARGET_LOSS) for x in ends),
                        INK,
                        dotted=True,
                        dots=False,
                    ),
                )
                if target
                else ()
            ),
        ),
        log_x=True,
        x_ticks=tokens,
        x_labels={n: token_count(n) for n in tokens},
        font_size=15,
    )


# The token-loss chart's lines: a token_losses_by_hint column each, its label and its color (the record's blue and
# the no-hints orange and green of the hint-corpus chart).
TOKEN_LOSS_SERIES = (
    ("loss", "record (#367)", BLUE, False),
    ("blank_hints", "record, blank hints", ORANGE, True),
    ("no_hints", "record trained without hints", GREEN, False),
    ("previous", "previous record (#360)", PURPLE, False),
)


def token_losses_by_hint(
    record: pl.LazyFrame, no_hints: pl.LazyFrame, previous: pl.LazyFrame
) -> pl.DataFrame:
    """Per hint length (HINT_LENGTHS, 0: no hint), the tokens and each TOKEN_LOSS_SERIES column's mean loss."""
    return (
        record.join(no_hints.select("position", no_hints="loss"), on="position")
        .join(previous.select("position", previous="loss"), on="position")
        .group_by("hint_length")
        .agg(
            tokens=pl.len(),
            loss=pl.col("loss").mean(),
            blank_hints=pl.col("loss_without_hints").mean(),
            no_hints=pl.col("no_hints").mean(),
            previous=pl.col("previous").mean(),
        )
        .sort("hint_length")
        .collect()
    )


def token_losses_chart(losses: pl.DataFrame) -> Chart:
    """The mean validation loss per hint length, a line per TOKEN_LOSS_SERIES, the lengths evenly spaced."""
    xs = tuple(range(len(HINT_LENGTHS)))
    if losses["hint_length"].to_list() != HINT_LENGTHS.tolist():
        raise ValueError(f"hint lengths {losses['hint_length'].to_list()}")
    return Chart(
        x_label="the hint's match length (tokens)",
        y_label="val loss",
        series=tuple(
            Series(label, tuple(zip(xs, losses[column].to_list())), color, dashed)
            for column, label, color, dashed in TOKEN_LOSS_SERIES
        ),
        x_ticks=xs,
        x_labels={x: str(n) if n else "no hint" for x, n in zip(xs, HINT_LENGTHS)},
        y_domain=(0, 4),
        font_size=15,
    )


def token_losses_caption(losses: pl.DataFrame) -> str:
    total = losses["tokens"].sum()
    shares = ", ".join(
        f"{f'{n}-token' if n else 'no'} hint {100 * t / total:.1f}%"
        for n, t in zip(losses["hint_length"], losses["tokens"])
    )
    return (
        f"The mean loss of the {total:,} validation targets of FineWeb-100B in each group ({shares}). Seed 0 for "
        "every run; the blank-hints line is the record's own trained model, validated again with every hint blank. "
        "The previous record runs with its loader's BOS-index race fixed (record_360/README.md)."
    )


def token_count(n: int) -> str:
    """`n` tokens in millions or billions, e.g. 75M, 1.2B."""
    return f"{n / 1e9:g}B" if n >= 1e9 else f"{n / 1e6:g}M"


def _chart(
    y_label: str, series: list[Series], legend_columns: int | None = None
) -> Chart:
    return Chart(
        x_label="n-gram",
        y_label=y_label,
        series=tuple(series),
        log_x=True,
        x_ticks=XTICKS,
        y_domain=(0, 100),
        x_prefix="n = ",
        tooltip_decimals=1,
        value_suffix="%",
        font_size=15,
        legend_columns=legend_columns,
    )


# The blog post's figures (blog.py's BlogFigures writes them). The diagrams are cropped to what they draw (CROP), so
# that each fills the post's figure card as the charts do, and each has a phone layout, NAME-phone.svg, for the post to
# show on narrow screens: narrower and taller, its text at about its size in CSS pixels. Each diagram's crop, (x, y,
# width, height) in its own units: its ink (getBBox in Chromium, with Nimbus Sans, which has Helvetica's metrics, for
# the text), and 3 units around it. Measure again when a diagram's drawing changes.
CROP = {
    "hint-mechanism.svg": (26, 71, 860, 470),
    # with DejaVu Sans's (Chromium's fallback, a fifth wider), so that its labels' run to the right edge fits in any font
    "hint-mechanism-phone.svg": (26, 61, 330, 640),
    "naive-hints.svg": (124, 16, 686, 152),
    "naive-hints-phone.svg": (6, 6, 332, 134),
    "fixed-size-array.svg": (96, 10, 808, 328),
    "fixed-size-array-sizes.svg": (96, 10, 808, 328),  # the same as the drawing it is swapped with
    "fixed-size-array-phone.svg": (6, 6, 341, 336),
    "fixed-size-array-sizes-phone.svg": (6, 6, 341, 336),
    "hint-entry-layout.svg": (28, 4, 904, 97),
    "hint-entry-layout-phone.svg": (6, 4, 338, 118),
}


def crop(name: str, svg: str) -> str:
    """svg cropped to CROP[name]."""
    x, y, w, h = CROP[name]
    root, n = re.subn(
        r"viewBox='0 0 \d+ \d+'",
        f"viewBox='{x} {y} {w} {h}'",
        svg,
        count=1,
    )
    assert n == 1, name
    return root


def phone_ids(svg: str) -> str:
    """svg with its ids, and the references to them, prefixed with phone-: a url(#...) finds the first element of
    its id on the page, and the desktop layout's, hidden on phones, may come first."""
    return re.sub(r"(id='|url\(#)", r"\1phone-", svg)


def blog_svgs() -> dict[str, str]:
    """The post's diagrams by file name, cropped, each with its phone layout (its ids its own: phone_ids)."""
    svgs = {
        "hint-mechanism.svg": hints_svg(),
        "naive-hints.svg": simple_index_svg(),
        "fixed-size-array.svg": slot_index_svg(),
        "fixed-size-array-sizes.svg": slot_index_svg(sizes=True),  # shown on hover
        "hint-entry-layout.svg": combined_entry_svg(),
        # on phones
        "hint-mechanism-phone.svg": phone_ids(hints_svg(phone=True)),
        "naive-hints-phone.svg": phone_ids(simple_index_svg(phone=True)),
        "fixed-size-array-phone.svg": phone_ids(slot_index_svg(phone=True)),
        "fixed-size-array-sizes-phone.svg": phone_ids(
            slot_index_svg(sizes=True, phone=True)
        ),
        "hint-entry-layout-phone.svg": phone_ids(combined_entry_svg(phone=True)),
    }
    return {name: crop(name, svg) for name, svg in svgs.items()}


def blog_charts(
    stats: KGramStats, sweep: ValCacheLosses
) -> dict[str, Chart | BarChart]:
    """The post's charts by file name, drawn by its Svelte components."""
    return {
        "kgram-accuracy.json": vote_coverage_chart(stats),
        "hint-corpus-scaling.json": val_cache_chart(sweep, target=True, reach=2**0.08),
        "build-speedups.json": ladder_tokens_chart(slowest_on_top=True),
    }


def blog_section() -> str:
    """What only the post has: the diagrams' and the animation's phone layouts (the rest is marked in its caption)."""
    cells = "".join(
        f"<div style='width: {CROP[name][2]}px; max-width: 100%'><p class='chart-subtitle'>{name}</p>{svg}</div>"
        for name, svg in blog_svgs().items()
        if name.endswith("-phone.svg")
    )
    return section(
        "Blog post",
        "The post's figures (BlogFigures, blog.py) are the ones above marked “In the blog post”, and the n-gram chart "
        "“As in the post”, recolored and the diagrams cropped. Here, what only the post has: their phone layouts.",
        svg_card(
            "The diagrams on phones",
            "<div style='display: flex; flex-wrap: wrap; gap: 32px; justify-content: center; align-items: flex-start'>"
            f"{cells}</div>",
            "Each at its size on a phone, a unit a CSS px; the post shows them in place of the others on narrow "
            "screens.",
        )
        + records_animation_card(
            records(),
            "The world record over time, animated, on phones",
            "The animation in its phone layout, as the post's record-animation.js draws record-progression on narrow "
            "screens, 350 px wide.",
            layout="phone",
            width=350,
        ),
    )


# The tweet thread's figures (twitter.py's Twitter shows them in its mock of X, twitter_media.py's TwitterMedia records
# them as records.mp4 and PNGs).
BLOG_TITLE = "N-gram hinted language modeling"
# the post's card on X: its og:image (TwitterMedia's, copied here and to brunborg.com's static/n-gram-hinted-lm-card.png)
BLOG_CARD = "/assets/twitter/n-gram-hinted-lm-card.png"
# px: the hint-corpus chart's width and padding, and its plot's height: about 1.8 times as wide as it is tall in all,
# its text about as large as the tweet's (15 px) at a tweet's 518 px, with room for every x label
CHART_SHEET, CHART_PAD, CHART_HEIGHT = 480, 6, 200


def tweet_chart(chart: Chart) -> Chart:
    """`chart` for a CHART_SHEET px wide sheet: 14 px text, every other x tick labelled (ending at the last; all keep
    their grid lines and points), the x axis titled without its N, the hint corpus's line labelled without its
    transformer's tokens, the levels (the target loss and no hint corpus) without points at their ends, its legend's
    entries one above the other in the plot's bottom left, below the falling line, its axis titles close to their
    labels, and its x axis ending just past the data."""
    ticks = chart.x_ticks or ()
    hinted, *rest = chart.series
    return dataclasses.replace(
        chart,
        x_label="hint corpus tokens",
        series=(
            dataclasses.replace(hinted, label="hint corpus"),
            *(dataclasses.replace(s, dots=False) for s in rest),
        ),
        x_tick_labels=ticks[len(ticks) % 2 == 0 :: 2],
        font_size=14,
        legend_columns=1,
        legend_orient="bottom-left",
        title_padding=4,
        x_padding=12,
    )


def chart_embed(spec: Path, label: str) -> str:
    """A Vega-Lite chart next to the page, drawn by report-charts.js."""
    name = escape(spec.name, quote=True)
    return f'<div class="altair-chart" data-spec="{name}" role="img" aria-label="{escape(label, quote=True)}"></div>'


def tweet_hint_chart(path: Path, sweep: ValCacheLosses) -> str:
    """The thread's hint-corpus chart (tweet_chart), written to `path`."""
    # reach: a dot past the data, the chart being narrower than the blog's
    spec = line_chart(tweet_chart(val_cache_chart(sweep, target=True, reach=2**0.15)), height=CHART_HEIGHT)
    return chart_embed(write_chart(path, spec), "Validation loss by hint-corpus size")


def tweet_build_svg() -> str:
    """The thread's build-speed bars: short_ladder_svg's, the record's as My code, with larger labels."""
    return short_ladder_svg(
        f"Hint corpus tokens in {record_build_s():.1f} seconds",
        ladder_tokens_chart({**SHORT_LADDER, RECORD: "My code"}),
        label_size=28,
        left=16,
        tick_size=24,
        axis_size=26,
    )


def twitter_section(data: Path, sweep: ValCacheLosses) -> str:
    """The thread's figures, in its order."""
    sheet = (
        f"<div style='width: {CHART_SHEET}px; max-width: 100%; margin: 0 auto; padding: {CHART_PAD}px'>"
        f"{tweet_hint_chart(data / 'val-cache-tweet.vl.json.gz', sweep)}</div>"
    )
    return section(
        "Twitter thread",
        "The thread's figures, in the order of its tweets, as twitter.html shows them; TwitterMedia (twitter_media.py) "
        "records them from it as records.mp4 and PNGs.",
        records_animation_card(
            records(),
            "The world record over time, animated, as in the first tweet",
            "The animation in its tweet layout (the phone one, shorter), at a tweet's 518 px: records.mp4.",
            layout="tweet",
            width=518,
        )
        + svg_card(
            "Retrieval hints at a glance, stacked",
            how_it_works_svg(stacked=True),
            "The diagram of the same name above, the context over the training data: how-it-works.png.",
        )
        + '<figure class="card chart-card"><div class="chart-head"><p class="chart-title">Validation loss by '
        f'hint-corpus size</p></div>{sheet}<figcaption class="chart-caption">The hint corpus only chart above, '
        f"{CHART_SHEET} px wide, with the target loss: hint-corpus-scaling.png.</figcaption></figure>"
        + svg_card(
            "",
            tweet_build_svg(),
            "The three versions above, the record's as My code: build-speed.png.",
        )
        + svg_card(
            "The blog post's link card",
            f"<img src='{BLOG_CARD}' alt='{BLOG_TITLE}' style='display: block; width: 100%'>",
            "In the last tweet: TwitterMedia's n-gram-hinted-lm-card.png, as copied to assets/twitter/ (its og:image).",
        ),
    )


def link_index(page_path: Path, name: str = "index.html") -> Path:
    """Point ./`name` (in the checkout final-results/ is in) at `page_path`, with a relative symlink."""
    index = final_results_root().parent / name
    index.unlink(missing_ok=True)
    index.symlink_to(os.path.relpath(page_path, index.parent))
    return index


if __name__ == "__main__":

    def _main() -> None:
        # Imported, not this file's own definitions: run as a file, they would be __main__.Publish, another spec.
        from retrieval_blog_post.ladder import EXTRA, RUNGS, BuildTime, search
        from retrieval_blog_post.overview import Publish, link_index
        from retrieval_blog_post.config import create

        publish = Publish()
        overview = publish.overview
        # Everything the drawings and results.json load, on Slurm workers (the cached results load): the statistics,
        # the record runs, the llm.c runs and the build timings (the search's, round by round, and EXTRA's).
        create(
            [
                *(
                    spec
                    for specs in overview.match_stats.values()
                    for spec in specs.values()
                ),
                overview.val_cache_sweep,
                *overview.one_pass,
                overview.data_scaling,
                *overview.token_losses,
                *(spec for specs in publish.llmc.values() for spec in specs),
            ],
            job_name="publish",
        )
        search(list(RUNGS))
        create(
            [BuildTime(version=v, tokens=n) for v, ns in EXTRA.items() for n in ns],
            job_name="ladder",
        )
        # Then the drawings here, always anew: they are cheap, and their identities do not change with the code.
        for spec in (*publish.drawings(), publish):
            spec.delete("force")
        print("results:", publish.create())
        pages = {"index.html": publish.overview, "twitter.html": publish.twitter}
        for name, page in pages.items():
            path = page.load_existing()
            print(link_index(path, name), "->", path)

    _main()
