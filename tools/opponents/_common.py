"""Shared helpers for the practice opponents (simulator-only bots)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
from game import Action, Street
from bot.cards import preflop_pct
from bot.equity import RangeSpec, simulate, equity

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_RNG = np.random.default_rng(7)

def raw_equity(obs, n=300):
    """Equity vs a RANDOM hand (what most student bots compute)."""
    if not obs.board:
        from bot.cards import preflop_equity
        return preflop_equity(obs.hole_cards)
    return equity(simulate(obs.hole_cards, obs.board, n, _RNG, RangeSpec()))

def pick(obs, *prefs):
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL
