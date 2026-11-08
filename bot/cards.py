"""
cards.py - the precomputed preflop table and helpers to look hands up in it.

PREFLOP_EQUITY[i]  chance hand i beats a random hand (0..1)
PREFLOP_PCT[i]     what fraction of all starting hands are weaker (0..1)
                   e.g. 0.90 means "top 10% hand"

i is the hand's colex index: a unique number 0..376,991 for each sorted
set of five cards, computed without any dictionary.
"""

from __future__ import annotations

import os
from math import comb

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preflop_equity.npy")

BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(36)], dtype=np.int64)
_BINOM_PY = [[comb(n, k) for k in range(6)] for n in range(36)]

PREFLOP_EQUITY = np.load(_DATA).astype(np.float32) / 65535.0
_order = np.argsort(PREFLOP_EQUITY, kind="stable")
PREFLOP_PCT = np.empty_like(PREFLOP_EQUITY)
PREFLOP_PCT[_order] = (np.arange(len(_order)) + 0.5) / len(_order)
del _order


def colex_one(hole) -> int:
    """Index of one hand (any order) in the preflop tables."""
    c = sorted(hole)
    b = _BINOM_PY
    return b[c[0]][1] + b[c[1]][2] + b[c[2]][3] + b[c[3]][4] + b[c[4]][5]


def colex_many(holes: np.ndarray) -> np.ndarray:
    """Indexes of N hands at once. holes: (N,5) int array, any order."""
    s = np.sort(holes, axis=1)
    return (BINOM[s[:, 0], 1] + BINOM[s[:, 1], 2] + BINOM[s[:, 2], 3]
            + BINOM[s[:, 3], 4] + BINOM[s[:, 4], 5])


def preflop_equity(hole) -> float:
    return float(PREFLOP_EQUITY[colex_one(hole)])


def preflop_pct(hole) -> float:
    return float(PREFLOP_PCT[colex_one(hole)])
