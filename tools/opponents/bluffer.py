"""Tricky: plays decent hands normally, but bluffs 35% of the time when checked to."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *
import random
_r = random.Random(11)

def new_half(config, memory): pass
def on_hand_end(result, memory): pass

def choose_action(obs, memory):
    eq = raw_equity(obs)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq > 0.65:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if obs.to_call == 0 and _r.random() < 0.35:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if obs.to_call == 0 or eq > price:
        return Action.CHECK_CALL
    return pick(obs, Action.FOLD)
