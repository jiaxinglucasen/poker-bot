"""
build_preflop.py - precompute how strong EVERY starting hand is (run once, offline).

There are C(36,5) = 376,992 possible five-card starting hands. For each one we
estimate its "equity": the chance it wins at showdown against a random hand
on a random board (ties count half).

Speed-ups:
  * Suit symmetry: As Ks Qh Jh Tc plays exactly like Ah Kh Qs Js Td, so we
    group hands into classes and simulate each class once.
  * Batched numpy scoring (bot/fasteval.py).

Output: bot/data/preflop_equity.npy, a uint16 array indexed by the hand's
"colex index" (see bot/cards.py). Load it in the bot for an instant lookup.

    python tools/build_preflop.py --trials 3000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations, permutations
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bot.fasteval import batch_omaha_score  # noqa: E402

OUT = os.path.join(ROOT, "bot", "data", "preflop_equity.npy")
BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(36)], dtype=np.int64)


def colex(sorted_hands: np.ndarray) -> np.ndarray:
    """Unique index 0..376,991 for each sorted 5-card hand (rows ascending)."""
    return sum(BINOM[sorted_hands[:, i], i + 1] for i in range(5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3000, help="simulations per hand class")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    hands = np.array(list(combinations(range(36), 5)), dtype=np.int64)   # sorted rows
    idx = colex(hands)
    assert np.array_equal(np.sort(idx), np.arange(len(hands)))

    # Canonical class of each hand = smallest colex index over all 24 suit relabelings.
    rank, suit = hands >> 2, hands & 3
    canon = np.full(len(hands), np.iinfo(np.int64).max)
    for perm in permutations(range(4)):
        relabeled = np.sort(rank * 4 + np.array(perm)[suit], axis=1)
        canon = np.minimum(canon, colex(relabeled))
    classes, inverse = np.unique(canon, return_inverse=True)
    print(f"{len(hands)} hands -> {len(classes)} suit classes")

    # Representative hand for each class: the first hand that maps to it.
    rep = np.zeros(len(classes), dtype=np.int64)
    rep[inverse[::-1]] = np.arange(len(hands))[::-1]
    reps = hands[rep]

    equity = np.zeros(len(classes))
    chunk = 200                                   # classes per batch
    all_cards = np.arange(36)
    for start in range(0, len(classes), chunk):
        mine = reps[start:start + chunk]           # (C,5)
        c = len(mine)
        mine_rep = np.repeat(mine, args.trials, axis=0)              # (C*T,5)
        # Deal 10 other cards (5 opponent + 5 board) avoiding my cards.
        mask = np.ones((c, 36), dtype=bool)
        mask[np.arange(c)[:, None], mine] = False
        live = np.array([all_cards[m] for m in mask])                # (C,31)
        live_rep = np.repeat(live, args.trials, axis=0)
        noise = rng.random(live_rep.shape)
        pick = np.argpartition(noise, 9, axis=1)[:, :10]
        dealt = np.take_along_axis(live_rep, pick, axis=1)
        opp, board = dealt[:, :5], dealt[:, 5:]
        me_s = batch_omaha_score(mine_rep, board)
        op_s = batch_omaha_score(opp, board)
        win = (me_s > op_s) + 0.5 * (me_s == op_s)
        equity[start:start + c] = win.reshape(c, args.trials).mean(axis=1)
        if start % (chunk * 20) == 0:
            print(f"  {start}/{len(classes)} classes  {time.time() - t0:.0f}s")

    table = np.zeros(len(hands), dtype=np.uint16)
    table[idx] = np.round(equity[inverse] * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, table)
    print(f"saved {OUT}  ({time.time() - t0:.0f}s)")
    print(f"equity range {equity.min():.3f} .. {equity.max():.3f}, median {np.median(equity):.3f}")


if __name__ == "__main__":
    main()
