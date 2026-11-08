# How our poker bot works

A short guide for teammates. The bot picks whichever action wins the most
chips on average, using a fast simulation of how often we win and a running
model of how this particular opponent plays.

## Files

| File | What it does |
|---|---|
| `agent.py` | The starter server. Our three functions just call into `bot/` and never let an error escape. |
| `bot/fasteval.py` | Scores hundreds of hands at once with numpy (~75x faster than `omaha_score` one at a time). |
| `bot/cards.py` | Loads the preflop table and looks a hand up in it instantly. |
| `bot/data/preflop_equity.npy` | Strength of all 376,992 starting hands, computed offline. |
| `bot/equity.py` | Monte Carlo: deals imaginary opponent hands and boards, weighted by what the opponent's betting tells us. |
| `bot/opponent.py` | Counts how often this opponent raises, calls, and folds to our bets. |
| `bot/strategy.py` | Turns all that into an expected-value number for each legal action and picks the best. Also manages the time bank. |
| `tools/fastsim.py` | Plays full duplicate matches in one process (no HTTP), with the same scoring as the judge. |
| `tools/tune.py` | Runs our bot against a whole field of opponents and gives one score. |
| `tools/build_preflop.py` | Rebuilds the preflop table (a few minutes). |
| `tools/opponents/` | Practice opponents: equity_bot, nit, maniac, bluffer. |

## One decision, step by step

1. **Update opponent stats** from any new actions in `obs.action_history`.
2. **Decide how many simulations we can afford**, from the time we have left
   and how many decisions are still to come. Bigger pots get more.
3. **Describe the opponent's range**: if they raised preflop and they raise
   30% of the time, they probably hold a top-30% hand. Same idea on later
   streets, using the strength of their made hand on the board.
4. **Simulate** N opponent hands and run-outs, all scored in one batch.
5. **Compute expected value** for each legal action:
   - Fold: 0
   - Check/call: equity x (pot + call) x realize - call
   - Raise: P(they fold) x pot + P(they call) x (equity-when-called x final pot - our chips)
6. **Return the best one.** Any exception falls back to a safe check/call/fold.

## Two bugs worth knowing about

1. **Re-raising into strong hands.** The tight practice bot folds about 70% of
   the time when we bet, so the bot learned "this player folds a lot" and
   re-raised whenever it raised. But it almost never folds once it has
   raised itself. Fix: track folds to our bets and folds to our re-raises
   separately. Result vs the tight bot went from -1.2 to about +0.8 bb/deal.
2. **One bluff setting can't fit everyone.** A low "bluff allowance" was best
   against the tight bot and a high one against the bluffer. Fix: after each
   showdown, check whether the opponent's raises came from hands inside
   their usual range, and learn the allowance per opponent.

## Results (1000-deal duplicate matches, two seeds each)

| Opponent | bb/deal | Verdict |
|---|---|---|
| random | +9.6 to +11.3 | win, win |
| calling station | +13.0 | win, win |
| always half-pot | +6.4 to +7.3 | win, win |
| always pot | +3.4 to +5.0 | win, win |
| equity bot (pot odds, no bluffs) | +1.7 to +2.5 | win, win |
| nit (top 20% only) | +0.5 to +1.6 | win, chop |
| maniac | +4.7 to +5.3 | win, win |
| bluffer | +3.1 to +4.2 | win, win |

A copy of the bot against itself chops, as it should. With the real HTTP
judge, the bot used about 1.5 to 2 s of its 8 s time bank per half, with 0 errors.

## Testing

```
python tools/fastsim.py agent.py baseline-bots/baseline_random.py --deals 1000
python tools/tune.py --seeds 1 2 3
python tools/tune.py --params '{"range_floor": 0.1}'
```

Then a real HTTP match, which also measures network overhead:

```
python agent.py
python baseline-bots/baseline_random.py
python judge_engine.py
```

Always check that `errors` is 0 and time stays well under 8 s per half.
