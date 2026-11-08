"""
equity.py - Monte Carlo: "how often do I win against what this opponent probably holds?"

One call deals N imaginary opponent hands and N imaginary rest-of-boards,
scores them all at once (fasteval), and returns per-sample arrays. The
strategy then weights those samples by how likely the opponent is to hold
each hand given how they've bet (see RangeSpec).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cards import PREFLOP_PCT, colex_many
from .fasteval import (CARD_KEY, SUIT, TABLE, KEYSPACE, NO_SUIT_TRIPLE, _pair_parts,
                       _triple_parts, sample_unseen, score_from_parts)
from itertools import combinations


@dataclass
class RangeSpec:
    """How narrow the opponent's range is on each street.

    keep[s] = fraction of hands the opponent would have played the way they
    did on street s (1.0 = no information). floor = weight kept for hands
    outside that fraction (bluffs and draws are never impossible).
    """
    keep: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    floor: float = 0.2


@dataclass
class Sample:
    win: np.ndarray        # (N,) 1 win, 0.5 tie, 0 loss
    weight: np.ndarray     # (N,) how likely the opponent holds that hand
    strength: np.ndarray   # (N,) opponent's strength percentile right now (0..1)


def _known_board_score(holes: np.ndarray, known) -> np.ndarray:
    """Best 2+3 hand using only the KNOWN board cards (3, 4 or 5 of them)."""
    known = list(known)
    trip = np.array(list(combinations(known, 3)))                 # (T,3)
    tkey = CARD_KEY[trip].sum(axis=1)                             # (T,)
    ts = SUIT[trip]
    tsuit = np.where((ts[:, 0] == ts[:, 1]) & (ts[:, 1] == ts[:, 2]), ts[:, 0], NO_SUIT_TRIPLE)
    pkey, psuit = _pair_parts(holes)                              # (N,10)
    key = pkey[:, :, None] + tkey[None, None, :]
    flush = psuit[:, :, None] == tsuit[None, None, :]
    return TABLE[key + flush * KEYSPACE].max(axis=(1, 2))


def _pct(values: np.ndarray) -> np.ndarray:
    """Percentile rank of each value within the array (ties share a rank)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    # Give tied values the same (average) rank so ties are treated fairly.
    sorted_vals = values[order]
    _, first, counts = np.unique(sorted_vals, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return (ranks + 0.5) / len(values)


def simulate(hole, board, n: int, rng: np.random.Generator, spec: RangeSpec) -> Sample:
    hole = np.asarray(hole, dtype=np.int64)
    board = list(board)
    k = len(board)
    dealt = sample_unseen(rng, list(hole) + board, n, 5 + (5 - k))
    opp = dealt[:, :5]
    if k < 5:
        full = np.concatenate([np.tile(np.asarray(board, dtype=np.int64), (n, 1)), dealt[:, 5:]], axis=1) \
            if k else dealt[:, 5:]
    else:
        full = np.tile(np.asarray(board, dtype=np.int64), (n, 1))

    # Final showdown result for every sample.
    tparts = _triple_parts(full)
    me = score_from_parts(*_pair_parts(np.tile(hole, (n, 1))), *tparts)
    op = score_from_parts(*_pair_parts(opp), *tparts)
    win = (me > op) + 0.5 * (me == op)

    # Opponent strength on each street they've already played, for range weighting.
    pf = PREFLOP_PCT[colex_many(opp)].astype(np.float64)
    weight = np.where(pf >= 1.0 - spec.keep[0], 1.0, spec.floor)
    strength = pf
    for street, nb in ((1, 3), (2, 4), (3, 5)):
        if k < nb:
            break
        made = _pct(_known_board_score(opp, board[:nb]).astype(np.float64))
        # A made hand isn't everything early on: blend in preflop strength for draws.
        mix = {1: 0.7, 2: 0.85, 3: 1.0}[street]
        strength = mix * made + (1 - mix) * pf
        if spec.keep[street] < 1.0:
            # Percentile WITHIN the hands that survived earlier streets.
            weight = weight * np.where(_weighted_pct(strength, weight) >= 1.0 - spec.keep[street],
                                       1.0, spec.floor)
    return Sample(win=win, weight=weight, strength=_weighted_pct(strength, weight))


def _weighted_pct(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Percentile of each value, counting each sample by its weight."""
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weight[order])
    total = cw[-1] if len(cw) and cw[-1] > 0 else 1.0
    out = np.empty(len(values))
    out[order] = (cw - 0.5 * weight[order]) / total
    return out


def equity(s: Sample) -> float:
    """Chance we win against the weighted range."""
    tw = s.weight.sum()
    return float((s.win * s.weight).sum() / tw) if tw > 0 else 0.5


def equity_when_called(s: Sample, fold_frac: float) -> float:
    """Chance we win if the opponent folds their weakest `fold_frac` and continues."""
    keep = s.weight * (s.strength >= fold_frac)
    tw = keep.sum()
    if tw <= 0:
        return equity(s)
    return float((s.win * keep).sum() / tw)


def strength_of(hole, known, rng: np.random.Generator, n: int = 150) -> float:
    """Percentile (0..1) of one hand among random hands, on the known board.
    Same blend as simulate(): preflop strength early, made hand later."""
    pf = float(PREFLOP_PCT[colex_many(np.asarray([hole]))[0]])
    if len(known) < 3:
        return pf
    others = sample_unseen(rng, list(hole) + list(known), n, 5)
    theirs = _known_board_score(others, known)
    mine = _known_board_score(np.asarray([hole]), known)[0]
    made = ((theirs < mine).sum() + 0.5 * (theirs == mine).sum()) / n
    mix = {3: 0.7, 4: 0.85, 5: 1.0}[len(known)]
    return float(mix * made + (1 - mix) * pf)
