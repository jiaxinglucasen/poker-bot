"""
opponent.py - learn how THIS opponent plays, from what we can observe.

We watch every opponent action in obs.action_history and count, per street:
  * how often they raise when they act         -> how wide their raising range is
  * how often they continue when facing a bet  -> how wide their calling range is
  * how often they fold to OUR bets/raises     -> how often our bluffs work

Fold-to-us is tracked in two separate buckets, because they behave very
differently:
  BET      we bet/raise and they had NOT raised yet this street
  RERAISE  we raise over THEIR raise (they've already shown strength)
A tight player folds a lot to bets but almost never to re-raises, and
mixing the two made the bot re-raise into strong hands.

Every rate starts from a "prior" (a sensible guess worth a few fake
observations), so the first hands don't swing wildly. As real hands come
in, the observed behaviour takes over.
"""

from __future__ import annotations

from game import Action

PRIOR_N = 12.0
PRIOR_RAISE = (0.30, 0.30, 0.30, 0.30)      # per street
PRIOR_CONTINUE = (0.65, 0.60, 0.60, 0.55)   # when facing a bet
PRIOR_FOLD = (0.35, 0.20)                   # BET, RERAISE
BET, RERAISE = 0, 1
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def new_stats() -> dict:
    return {
        "acts": [0, 0, 0, 0], "raises": [0, 0, 0, 0],
        "faced": [0, 0, 0, 0], "continued": [0, 0, 0, 0],
        "aggr": [0, 0], "folded": [0, 0],       # our bets/raises by kind, and their folds
        "hand": -1, "seen": 0,
        "street": 0, "street_totals": [0, 0],
        "opp_raised_street": False,
        "open_kind": None,     # kind of our latest bet/raise still awaiting their answer
        "pending": None,       # (kind or None, street) of the action we just returned
        "last_hist": (),       # this hand's history as of our latest decision
        "sd_raises": 0,        # opponent raises we later saw at showdown...
        "sd_bluffs": 0,        # ...and how many were with a hand outside their usual range
    }


def _rate(hits, n, prior, prior_n=PRIOR_N):
    return (hits + prior * prior_n) / (n + prior_n)


def raise_rate(st: dict, street: int) -> float:
    return _rate(st["raises"][street], st["acts"][street], PRIOR_RAISE[street])


def continue_rate(st: dict, street: int) -> float:
    return _rate(st["continued"][street], st["faced"][street], PRIOR_CONTINUE[street])


def fold_to_us(st: dict, kind: int) -> float:
    return _rate(st["folded"][kind], st["aggr"][kind], PRIOR_FOLD[kind], prior_n=15.0)


def current_kind(st: dict) -> int:
    """If we raise right now, is it a plain bet/raise or a re-raise over them?"""
    return RERAISE if st["opp_raised_street"] else BET


def _start_hand(st: dict, obs) -> None:
    st["hand"] = obs.hand_index
    st["seen"] = 0
    st["street"] = 0
    st["street_totals"] = [1, 2] if obs.is_button else [2, 1]   # [me, opp] blinds
    st["opp_raised_street"] = False
    st["open_kind"] = None
    st["pending"] = None


def observe(st: dict, obs) -> None:
    """Process any actions we haven't seen yet in this hand."""
    if st["hand"] != obs.hand_index:
        _start_hand(st, obs)
    hist = obs.action_history
    for rec in hist[st["seen"]:]:
        street = int(rec.street)
        if street != st["street"]:
            st["street"] = street
            st["street_totals"] = [0, 0]
            st["opp_raised_street"] = False
        if rec.is_me:
            if rec.action in RAISES:
                st["open_kind"] = RERAISE if st["opp_raised_street"] else BET
            st["street_totals"][0] = rec.street_total
            continue
        # An opponent action.
        facing = st["street_totals"][0] > st["street_totals"][1]
        st["acts"][street] += 1
        if rec.action in RAISES:
            st["raises"][street] += 1
            st["opp_raised_street"] = True
        if facing:
            st["faced"][street] += 1
            if rec.action != Action.FOLD:
                st["continued"][street] += 1
        if st["open_kind"] is not None:
            st["aggr"][st["open_kind"]] += 1          # they answered our bet without folding
            st["open_kind"] = None
        st["street_totals"][1] = rec.street_total
    st["seen"] = len(hist)


PRIOR_BLUFF = 0.30     # share of raises that come from "outside the range" hands
TYPICAL_KEEP = 0.30


def bluff_share(st: dict) -> float:
    return _rate(st["sd_bluffs"], st["sd_raises"], PRIOR_BLUFF, prior_n=8.0)


def range_floor(st: dict) -> float:
    """Weight for hands outside the opponent's usual range, learned from showdowns.

    If a share b of their raises are 'outside' hands, and 'inside' hands are a
    fraction k of all hands, the matching weight is  b*k / ((1-b)*(1-k)).
    """
    b = min(bluff_share(st), 0.8)
    k = TYPICAL_KEEP
    return max(0.03, min(0.6, b * k / ((1 - b) * (1 - k))))


def record_showdown(st: dict, result, strength_at) -> None:
    """After a showdown we see their cards: were their raises 'in range'?

    strength_at(hole, board_cards) -> percentile 0..1 of that hand on that board.
    """
    if not result.went_to_showdown or result.opp_hole is None:
        return
    board = list(result.board)
    for rec in st.get("last_hist", ()):
        if rec.is_me or rec.action not in RAISES:
            continue
        s = int(rec.street)
        known = board[:(0, 3, 4, 5)[s]]
        strength = strength_at(result.opp_hole, known)
        st["sd_raises"] += 1
        if strength < 1.0 - raise_rate(st, s):
            st["sd_bluffs"] += 1


def note_my_action(st: dict, action, street) -> None:
    """Remember what we just returned. If the hand ends before our next turn,
    hand_over() uses this to work out the opponent's final answer."""
    kind = current_kind(st) if action in RAISES else None
    st["pending"] = (kind, int(street))


def hand_over(st: dict, result) -> None:
    """Fill in the opponent's LAST action, which history never shows us."""
    if st["hand"] != result.hand_index or not st.get("pending"):
        return                           # we never acted this hand
    kind, street = st["pending"]
    st["pending"] = None
    if kind is None:
        return                           # our last action wasn't a bet/raise
    opp_folded = (not result.went_to_showdown) and result.winner == "me"
    st["aggr"][kind] += 1
    st["faced"][street] += 1
    st["acts"][street] += 1
    if opp_folded:
        st["folded"][kind] += 1
    else:
        st["continued"][street] += 1
