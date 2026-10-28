from typing import Any

import numpy as np

def kgram_counts(
    files: list[str],
    tokens: np.ndarray,
    k: int,
    top: int,
    memory: int,
    train_tokens: int | None = None,
    filter: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]: ...

def kgram_query_counts(
    files: list[str],
    tokens: np.ndarray,
    k: int,
    top: int,
    cnt_top: int,
    memory: int,
    breaks: list[int] = ...,
    filter: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]: ...

def kgram_stats(
    files: list[str],
    tokens: np.ndarray,
    k: int,
    depth: int,
    memory: int,
    train_tokens: int | None = None,
    filter: np.ndarray | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray]: ...

def train_hints(
    files: list[str],
    levels: list[int],
    entries: int,
    buckets: list[int] | None = None,
    slots: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

def train_counts(
    files: list[str], levels: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def val_overlap(
    val: str, train: list[str], n: int, offsets: np.ndarray, bytes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

class CountTable:
    """Exact next-token counts of every k-context (k of `levels`) in the training shards `files` (count_table.rs):
    create before the clock, build on it, query the validation tokens (inputs + the last target) after it."""
    @staticmethod
    def create(
        files: list[str], levels: list[int], threads: int, cpus: list[int], huge_pages: bool = False
    ) -> CountTable: ...
    def build(self) -> None: ...
    def query_counts(
        self, tokens: np.ndarray, top: int = 1024
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def query_rows(
        self, tokens: np.ndarray, row_levels: list[int], buckets: list[int], entries: int = 8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def sizes(self) -> list[tuple[int, int, int, int]]: ...
    def release(self) -> None: ...
    @property
    def levels(self) -> list[int]: ...
    @property
    def tokens(self) -> int: ...

class OnlineRows:
    """A run's training stream (`tokens`, its parts' starts `bounds` + its length) and when each position is trained
    (`spans` [m, 3]: (lo, hi, step), inputs [lo, hi) of `step`, hi its last target), for its online hint and count
    rows under the step rule (online.rs): a query at step s counts only occurrences whose tokens were all trained
    before s. hints/counts write the rows of the queries trained in steps [step_lo, step_hi) into stream-sized arrays
    (index = stream position), leaving the rest untouched. rule="segment" is train_hints/train_counts's rule (tests)."""
    def __init__(
        self, tokens: np.ndarray, bounds: list[int], spans: np.ndarray, threads: int, cpus: list[int]
    ) -> None: ...
    def prefault(self) -> None: ...
    def hints(
        self,
        levels: list[int],
        buckets: list[int],
        entries: int,
        step_lo: int,
        step_hi: int,
        cell: np.ndarray,
        next: np.ndarray,
        weights: np.ndarray,
        rule: str = "step",
    ) -> int: ...
    def counts(
        self,
        levels: list[int],
        step_lo: int,
        step_hi: int,
        tot: np.ndarray,
        dist: np.ndarray,
        cnt: np.ndarray,
        rule: str = "step",
    ) -> None: ...
    def prepare(self, levels: list[int], buckets: list[int], entries: int, count_levels: list[int]) -> None:
        """Before the clock: advance()'s state for this configuration, allocated and faulted in."""
    def advance(
        self,
        step_hi: int,
        levels: list[int],
        buckets: list[int],
        entries: int,
        count_levels: list[int],
        cell: np.ndarray | None = None,
        next: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        tot: np.ndarray | None = None,
        dist: np.ndarray | None = None,
        cnt: np.ndarray | None = None,
    ) -> int:
        """Incremental: the rows hints(levels, ...) + counts(count_levels, ...) write for steps [done, step_hi), with
        the counted occurrences kept between calls (cost per new step, not per prefix); returns the queries hinted."""
    @property
    def done(self) -> int: ...
    def __len__(self) -> int: ...

# the rest of the module, untyped
def __getattr__(name: str) -> Any: ...

def val_scan(
    files: list[str],
    tokens: np.ndarray,
    levels: list[int],
    short: int,
    short_files: int,
    threads: int,
    top: int = 1024,
    keep: int = 8,
    filter_bits: int = 28,
    scale: float = 0.0,
    growth: list[float] = ...,
    rare_below: int = 0,
    rare_bits: int = 16,
    anchor: int = 0,
    filter2_bits: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[str, float]],
]: ...

def val_hint_rows(
    ks: list[int],
    levels: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    buckets: list[int],
    entries: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
