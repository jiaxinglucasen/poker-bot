"""
strategy.py - the brain: pick the action with the highest expected value (EV).

For every legal action we estimate how many chips it wins on average,
counting from now (chips already in the pot are "sunk"):

  FOLD        0
  CHECK/CALL  equity * (pot + to_call) * realize - to_call
  RAISE       P(opp folds) * pot
              + P(opp calls) * (equity_when_called * final_pot * realize - our_chips_added)

"equity"  = how often we win vs the opponent's likely hands (Monte Carlo).
"realize" = a discount when more betting is still to come (we won't always
            get to see the showdown, and being out of position is worse).

Every number we can't compute exactly is a tunable PARAM below.
"""

from __future__ import annotations

import time

import numpy as np

from game import Action, Street

from . import opponent as opp_model
from .equity import RangeSpec, equity, equity_when_called, simulate, strength_of

PARAMS = {
    # How much of our equity we actually collect if more streets remain.
    "realize_ip": [0.95, 0.97, 0.98, 1.0],    # in position (button), per street
    "realize_oop": [0.86, 0.90, 0.94, 1.0],   # out of position
    "realize_raise": 0.97,                    # after our raise gets called
    "range_floor": 0.2,                       # weight for hands "outside" opp's range
    "adaptive_floor": True,                   # learn that weight from showdowns instead
    "half_fold_scale": 0.75,                  # half-pot raises get fewer folds
    "fold_cap": 0.85,                         # never assume more folds than this
    "call_margin": 0.0,                       # extra chips (in pots) needed to call
    "min_trials": 120,
    "max_trials": 900,
    "bank_target": 1.5,                       # seconds we plan to spend per half (of 8);
                                              # the rest is safety for HTTP overhead and
                                              # a slower tournament machine
}

# Local tuning only: tools/tune.py passes trial settings through this
# environment variable. In the tournament it isn't set, so PARAMS stay as above.
import json as _json, os as _os
PARAMS.update(_json.loads(_os.environ.get("BOT_PARAMS", "{}")))

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def new_half(memory: dict) -> None:
    memory["opp"] = opp_model.new_stats()
    memory["used"] = 0.0           # seconds spent in choose_action this half
    memory["decisions"] = 0
    memory["ms_per_trial"] = 0.004
    memory["rng"] = np.random.default_rng(20261107)


def _range_spec(obs, st) -> RangeSpec:
    """Turn the opponent's actions THIS hand into 'what fraction of hands
    would play like this' on each street."""
    keep = [1.0, 1.0, 1.0, 1.0]
    totals = [1, 2] if obs.is_button else [2, 1]     # [me, opp] incl. blinds
    street = 0
    for rec in obs.action_history:
        s = int(rec.street)
        if s != street:
            street, totals = s, [0, 0]
        if not rec.is_me:
            facing = totals[0] > totals[1]
            if rec.action in RAISES:
                keep[s] *= opp_model.raise_rate(st, s)
            elif facing and rec.action == Action.CHECK_CALL:
                keep[s] *= opp_model.continue_rate(st, s)
        totals[0 if rec.is_me else 1] = rec.street_total
    keep = [max(k, 0.03) for k in keep]
    floor = opp_model.range_floor(st) if PARAMS["adaptive_floor"] else PARAMS["range_floor"]
    return RangeSpec(keep=keep, floor=floor)


def _trials(obs, memory) -> int:
    """How many simulations we can afford right now without running out of time."""
    hands_left = max(1, 1000 - obs.hand_index)
    done = memory["decisions"]
    per_hand = (done / (obs.hand_index + 1)) if obs.hand_index > 20 else 3.0
    left_s = PARAMS["bank_target"] - memory["used"]
    if left_s <= 0.2:
        return 40
    # Each request also pays fixed costs we can't see (HTTP, JSON): leave room.
    budget_ms = 1000 * left_s / (hands_left * max(per_hand, 1.0)) - 0.3
    # Spend more thinking on big pots (they matter more).
    budget_ms *= min(2.5, 0.6 + obs.pot / 40)
    n = int((budget_ms - 0.15) / memory["ms_per_trial"])
    return int(min(PARAMS["max_trials"], max(PARAMS["min_trials"], n)))


def decide(obs, memory) -> Action:
    st = memory["opp"]
    opp_model.observe(st, obs)
    st["last_hist"] = obs.action_history

    legal = obs.legal_actions
    street = int(obs.street)
    n = _trials(obs, memory)
    spec = _range_spec(obs, st)

    t_sim = time.perf_counter()
    sample = simulate(obs.hole_cards, obs.board, n, memory["rng"], spec)
    took = time.perf_counter() - t_sim
    memory["ms_per_trial"] = 0.8 * memory["ms_per_trial"] + 0.2 * (1000 * took / n)

    eq = equity(sample)
    pot, call = obs.pot, obs.to_call
    all_in_call = call >= obs.my_stack
    if street == Street.RIVER or all_in_call or obs.opp_stack == 0:
        realize = 1.0
    else:
        realize = (PARAMS["realize_ip"] if obs.is_button else PARAMS["realize_oop"])[street]

    ev = {}
    if Action.FOLD in legal:
        ev[Action.FOLD] = 0.0
    ev[Action.CHECK_CALL] = eq * (pot + min(call, obs.my_stack)) * realize - min(call, obs.my_stack) \
        - PARAMS["call_margin"] * pot * (call > 0)

    f_base = min(PARAMS["fold_cap"], opp_model.fold_to_us(st, opp_model.current_kind(st)))
    for a in RAISES:
        if a not in legal:
            continue
        target = obs.raise_to[a]
        mine = target - obs.my_bet                        # chips we add
        their = max(0, target - obs.opp_bet)              # chips they must add to call
        their = min(their, obs.opp_stack)
        f = f_base * (PARAMS["half_fold_scale"] if a == Action.RAISE_HALF else 1.0)
        f = max(0.0, min(f, 0.95))
        eq_c = equity_when_called(sample, f)
        final_pot = pot + mine + their
        r = 1.0 if (street == Street.RIVER or mine >= obs.my_stack) else PARAMS["realize_raise"]
        ev[a] = f * pot + (1 - f) * (eq_c * final_pot * r - mine)

    best = max(ev, key=ev.get)
    opp_model.note_my_action(st, best, street)
    memory["decisions"] += 1
    memory["last"] = {"eq": eq, "n": n, "ev": {a.name: round(v, 2) for a, v in ev.items()}}
    return best


def warmup() -> None:
    """Run a few throwaway simulations so numpy's first-call setup cost is
    paid at server start instead of during a real decision."""
    rng = np.random.default_rng(0)
    for board in ([], [0, 5, 10], [0, 5, 10, 15], [0, 5, 10, 15, 20]):
        spec = RangeSpec(keep=[0.5, 0.5, 0.5, 0.5])
        simulate([30, 31, 32, 33, 34], board, 200, rng, spec)


def hand_over(result, memory) -> None:
    st = memory["opp"]
    opp_model.hand_over(st, result)
    opp_model.record_showdown(st, result, lambda hole, known: strength_of(hole, known, memory["rng"]))
    st["last_hist"] = ()
