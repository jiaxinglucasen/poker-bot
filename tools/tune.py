"""
tune.py - run our bot against a whole field of opponents and report one score.

    python tools/tune.py                                  # current settings
    python tools/tune.py --params '{"range_floor": 0.1}'  # try a change
    python tools/tune.py --deals 300 --seeds 1 2 3

Uses both CPU cores. "points" is computed like the tournament
(win 1, chop 0.5, loss 0); "avg bb" breaks ties and shows the margin.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

FIELD = [
    "baseline-bots/baseline_random.py",
    "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_half.py",
    "baseline-bots/baseline_always_pot.py",
    "tools/opponents/equity_bot.py",
    "tools/opponents/nit.py",
    "tools/opponents/maniac.py",
    "tools/opponents/bluffer.py",
]
POINTS = {"WIN": 1.0, "CHOP": 0.5, "LOSS": 0.0}


def run_one(job):
    me, opp, seed, deals, params = job
    os.environ["BOT_PARAMS"] = params
    os.chdir(ROOT)
    from fastsim import play_match
    r = play_match(me, opp, seed, deals)
    t = max(s.time_used for s in r["a_halves"])
    errs = sum(s.errors for s in r["a_halves"])
    return opp, seed, r["mean"], r["lo"], r["verdict"].split()[0], t, errs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot", default="agent.py")
    p.add_argument("--params", default="{}")
    p.add_argument("--deals", type=int, default=400)
    p.add_argument("--seeds", nargs="+", default=["1", "2"])
    p.add_argument("--field", nargs="*", default=None)
    a = p.parse_args()
    field = a.field or FIELD
    jobs = [(a.bot, o, s, a.deals, a.params) for o in field for s in a.seeds]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        res = list(ex.map(run_one, jobs))
    pts = bb = 0.0
    for opp, seed, mean, lo, verdict, t, errs in res:
        pts += POINTS.get(verdict, 0.0)
        bb += mean
        print(f"  {os.path.basename(opp):28s} seed {seed:>3}  {mean:+7.3f}  {verdict:5s}  "
              f"time {t:.2f}s  errors {errs}")
    print(f"params {a.params}\n  points {pts}/{len(res)}   avg bb/deal {bb / len(res):+.3f}")


if __name__ == "__main__":
    main()
