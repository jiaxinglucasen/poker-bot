"""A solid 'typical strong student' bot: equity vs random hand + pot odds, never bluffs."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import *

def new_half(config, memory): pass
def on_hand_end(result, memory): pass

def choose_action(obs, memory):
    eq = raw_equity(obs)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if eq > 0.68:
        return pick(obs, Action.RAISE_POT, Action.CHECK_CALL)
    if eq > 0.58:
        return pick(obs, Action.RAISE_HALF, Action.CHECK_CALL)
    if obs.to_call == 0 or eq > price + 0.05:
        return Action.CHECK_CALL
    return pick(obs, Action.FOLD)
