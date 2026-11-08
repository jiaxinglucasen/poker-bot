"""
fastsim.py - play duplicate matches between bots IN ONE PROCESS (no HTTP).

Much faster than judge_engine.py, so it's the tool for testing and tuning.
It uses the real game.py rules and the same scoring as the judge
(paired bb/deal, 95% confidence interval, win/chop/loss).

A "bot" is any .py file with new_half / choose_action / on_hand_end
(agent.py and every baseline bot qualify).

    python tools/fastsim.py agent.py baseline-bots/baseline_random.py
    python tools/fastsim.py agent.py tools/opponents/tight_aggressive.py --deals 500 --seeds 1 2 3

Like the tournament, each half gets a FRESH copy of each bot (the module is
re-imported), so no global state leaks between halves.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import statistics
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from game import BIG_BLIND, AgentConfig, Game, button_for_deal, default_action, make_deal  # noqa: E402

TIME_BANK = 8.0
MAX_ERRORS = 5


def load_bot(path: str):
    """Import a bot file as a brand-new module (fresh globals every time)."""
    name = f"bot_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Seat:
    """One bot for one half: its memory, time used and errors."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.bot = load_bot(path)
        self.memory: dict = {}
        self.time_used = 0.0
        self.errors = 0
        self.decisions = 0
        self.slowest = 0.0
        self.first_error = None
        self.pending = []           # finished hands waiting to be delivered
        self.bot.new_half(AgentConfig(), self.memory)

    @property
    def usable(self) -> bool:
        return self.errors < MAX_ERRORS and self.time_used < TIME_BANK

    def _deliver(self) -> None:
        for res in self.pending:
            try:
                self.bot.on_hand_end(res, self.memory)
            except Exception as exc:     # counts as an error, like the judge
                self._error(f"on_hand_end: {type(exc).__name__}: {exc}")
        self.pending = []

    def _error(self, msg: str) -> None:
        self.errors += 1
        if self.first_error is None:
            self.first_error = msg

    def act(self, obs):
        if not self.usable:
            return default_action(obs.to_call)
        start = time.perf_counter()
        self._deliver()                 # judge sends finished hands with /act
        try:
            action = self.bot.choose_action(obs, self.memory)
        except Exception as exc:
            action = None
            self._error(f"choose_action: {type(exc).__name__}: {exc}")
        took = time.perf_counter() - start
        self.decisions += 1
        self.slowest = max(self.slowest, took)
        self.time_used += max(0.0, took - 0.001)     # first ms is free
        if took > 1.0:
            self._error(f"slow decision: {took:.2f}s")
            return default_action(obs.to_call)
        if action not in obs.legal_actions:
            if action is not None:
                self._error(f"illegal action {action!r}")
            return default_action(obs.to_call)
        return action


def play_half(path_a: str, path_b: str, seed, deals: int, half: int):
    """Return (nets for bot A per deal, Seat A, Seat B)."""
    a, b = Seat(path_a), Seat(path_b)
    seats = (a, b) if half == 1 else (b, a)          # seats[i] sits in seat i
    a_seat = 0 if half == 1 else 1
    nets = []
    for d in range(deals):
        game = Game(make_deal(str(seed), d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            s = game.to_act
            game.apply(seats[s].act(game.observation(s)))
        nets.append(game.result().nets[a_seat])
        for s in (0, 1):
            seats[s].pending.append(game.hand_result(s))
    for s in seats:
        if s.usable:
            s._deliver()
    return nets, a, b


def play_match(path_a: str, path_b: str, seed=1, deals: int = 1000):
    n1, a1, b1 = play_half(path_a, path_b, seed, deals, half=1)
    n2, a2, b2 = play_half(path_a, path_b, seed, deals, half=2)
    paired = [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    mean = sum(paired) / len(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    forfeit_a = a1.errors >= MAX_ERRORS or a2.errors >= MAX_ERRORS
    forfeit_b = b1.errors >= MAX_ERRORS or b2.errors >= MAX_ERRORS
    if forfeit_a and forfeit_b:
        verdict = "BOTH_FORFEIT"
    elif forfeit_a:
        verdict = "LOSS (forfeit)"
    elif forfeit_b:
        verdict = "WIN (opp forfeit)"
    elif lo > 0:
        verdict = "WIN"
    elif hi < 0:
        verdict = "LOSS"
    else:
        verdict = "CHOP"
    return {
        "mean": mean, "lo": lo, "hi": hi, "verdict": verdict,
        "a_halves": (a1, a2), "b_halves": (b1, b2),
    }


def describe(seat: Seat) -> str:
    avg_ms = 1000 * seat.time_used / max(1, seat.decisions)
    text = (f"time {seat.time_used:5.2f}s/8s  decisions {seat.decisions:5d}  "
            f"avg {avg_ms:5.2f}ms  slowest {1000 * seat.slowest:6.1f}ms  errors {seat.errors}")
    if seat.first_error:
        text += f"\n      first error: {seat.first_error}"
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bot_a")
    p.add_argument("bot_b")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seeds", nargs="+", default=["1"])
    p.add_argument("--quiet", action="store_true", help="one line per match")
    args = p.parse_args()

    for seed in args.seeds:
        r = play_match(args.bot_a, args.bot_b, seed, args.deals)
        name_b = os.path.basename(args.bot_b)
        print(f"seed {seed:>4} vs {name_b:32s} {r['mean']:+7.3f} bb/deal  "
              f"CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  {r['verdict']}")
        if not args.quiet:
            for i, s in enumerate(r["a_halves"], 1):
                print(f"    A half {i}: {describe(s)}")
            for i, s in enumerate(r["b_halves"], 1):
                print(f"    B half {i}: {describe(s)}")


if __name__ == "__main__":
    main()
