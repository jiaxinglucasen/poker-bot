"""Very tight: only plays top-20% starting hands, then bets hard. Folds everything else."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

def new_half(config, memory): pass
def on_hand_end(result, memory): pass

def choose_action(obs, memory):
    if obs.street == Street.PREFLOP:
        if preflop_pct(obs.hole_cards) > 0.80:
            return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
        return pick(obs, Action.FOLD) if obs.to_call else Action.CHECK_CALL
    eq = raw_equity(obs)
    if eq > 0.6:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if obs.to_call == 0 or eq > 0.45:
        return Action.CHECK_CALL
    return pick(obs, Action.FOLD)
