"""
fasteval.py - score MANY Omaha hands at once with numpy.

Why: evaluator.omaha_score scores one hand with 100 Python dict lookups
(roughly 20 microseconds). A simulation needs two of those per trial, so
Python alone gives only ~50 trials per millisecond. Doing hundreds of
trials as ONE batch of array operations is many times faster. (Same idea
as a GPU: do the same small job on lots of data at once.)

The trick that avoids sorting cards:
  * A five-card hand's strength depends only on WHICH RANKS it has (with
    repeats) and whether it is a flush.
  * Give each rank r the weight 5**r. Adding the weights of five cards gives
    a number that is different for every multiset of ranks (each rank
    appears at most 4 times, so base 5 never "carries").
  * So score = TABLE[rank_key + FLUSH_OFFSET * is_flush], one array lookup.
  * Omaha uses exactly 2 hole + 3 board cards, so we add a "pair key" and a
    "triple key" with broadcasting to get all 10 x 10 = 100 hands at once.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations, combinations_with_replacement

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import _score_ranks  # noqa: E402  (the official scoring logic)

NUM_CARDS = 36
POW5 = 5 ** np.arange(9, dtype=np.int64)            # weight of each rank
KEYSPACE = 5 ** 9                                    # 1,953,125 possible keys

RANK = np.arange(NUM_CARDS) >> 2
SUIT = np.arange(NUM_CARDS) & 3
CARD_KEY = POW5[RANK].astype(np.int32)               # rank weight of each card

PAIRS = np.array(list(combinations(range(5), 2)))    # (10, 2) hole-card picks
TRIPLES = np.array(list(combinations(range(5), 3)))  # (10, 3) board-card picks

NO_SUIT_PAIR = 8     # "this pair is not suited" (never equals a triple's code)
NO_SUIT_TRIPLE = 9   # "this triple is not one suit"


def _build_table() -> np.ndarray:
    """Score every possible (rank multiset, flush?) combination once."""
    table = np.zeros(2 * KEYSPACE, dtype=np.int32)
    for ranks in combinations_with_replacement(range(9), 5):
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue                                 # five of a kind: impossible
        key = int(sum(POW5[r] for r in ranks))
        desc = tuple(sorted(ranks, reverse=True))
        table[key] = _score_ranks(desc, False)
        if len(set(ranks)) == 5:                     # only distinct ranks can be a flush
            table[KEYSPACE + key] = _score_ranks(desc, True)
    return table


TABLE = _build_table()


def _pair_parts(holes: np.ndarray):
    """For (N,5) hole cards -> (N,10) pair keys and (N,10) pair suit codes."""
    a, b = holes[:, PAIRS[:, 0]], holes[:, PAIRS[:, 1]]
    key = CARD_KEY[a] + CARD_KEY[b]
    sa, sb = SUIT[a], SUIT[b]
    suit = np.where(sa == sb, sa, NO_SUIT_PAIR)
    return key, suit


def _triple_parts(boards: np.ndarray):
    """For (N,5) boards -> (N,10) triple keys and (N,10) triple suit codes."""
    a, b, c = (boards[:, TRIPLES[:, i]] for i in range(3))
    key = CARD_KEY[a] + CARD_KEY[b] + CARD_KEY[c]
    sa, sb, sc = SUIT[a], SUIT[b], SUIT[c]
    suit = np.where((sa == sb) & (sb == sc), sa, NO_SUIT_TRIPLE)
    return key, suit


def score_from_parts(pkey, psuit, tkey, tsuit) -> np.ndarray:
    """Best of the 100 two-plus-three hands for each row -> (N,) scores."""
    key = pkey[:, :, None] + tkey[:, None, :]                  # (N,10,10)
    flush = psuit[:, :, None] == tsuit[:, None, :]
    return TABLE[key + flush * KEYSPACE].max(axis=(1, 2))


def batch_omaha_score(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Like evaluator.omaha_score, but for N hands at once. (N,5),(N,5) -> (N,)."""
    holes = np.asarray(holes)
    boards = np.asarray(boards)
    return score_from_parts(*_pair_parts(holes), *_triple_parts(boards))


def sample_unseen(rng: np.random.Generator, dead, n: int, k: int) -> np.ndarray:
    """n random draws of k distinct cards each, avoiding the `dead` cards.

    Returns an (n, k) int array. Works by giving every live card a random
    number and taking the k smallest in each row (a partial shuffle).
    """
    live = np.setdiff1d(np.arange(NUM_CARDS), np.asarray(dead, dtype=np.int64))
    noise = rng.random((n, live.size))
    pick = np.argpartition(noise, k - 1, axis=1)[:, :k] if k < live.size else np.argsort(noise, axis=1)
    return live[pick]
