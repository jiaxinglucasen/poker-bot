"""Hyper-aggressive: pot-raises most of the time, folds trash only when facing big bets."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
import random
_r = random.Random(3)

def new_half(config, memory): pass
def on_hand_end(result, memory): pass

def choose_action(obs, memory):
    eq = raw_equity(obs, n=150)
    if eq > 0.35 or _r.random() < 0.5:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if obs.to_call > obs.pot * 0.6:
        return pick(obs, Action.FOLD)
    return Action.CHECK_CALL
