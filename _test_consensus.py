# -*- coding: utf-8 -*-
"""Unit tests for the PROMPT 17 multi-engine consensus filter (core/consensus.py).

Pure / headless: no TensorFlow, no GUI. Run with: python _test_consensus.py
"""
from core.consensus import (
    top_numbers_from_dist,
    top_numbers_from_preds,
    tally_votes,
    consensus_numbers,
    build_votes,
    apply_consensus_filter,
)

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# --------------------------------------------------------------------------- #
# top_numbers_*
# --------------------------------------------------------------------------- #
def test_top_from_dist():
    dist = {1: 0.5, 2: 0.3, 5: 0.15, 8: 0.05}
    check(top_numbers_from_dist(dist, top_n=2) == [1, 2], "top-2 by prob")
    check(top_numbers_from_dist(dist, top_n=3) == [1, 2, 5], "top-3 by prob")
    check(top_numbers_from_dist({}, top_n=3) == [], "empty dist -> []")
    check(top_numbers_from_dist(None, top_n=3) == [], "None dist -> []")
    check(top_numbers_from_dist(dist, top_n=3, min_prob=0.2) == [1, 2],
          "min_prob cuts low votes")


def test_top_from_preds():
    preds = [
        {"number": 5, "confidence": 0.1},
        {"number": 1, "confidence": 0.6},
        {"number": 2, "confidence": 0.3},
    ]
    check(top_numbers_from_preds(preds, top_n=2) == [1, 2], "preds top-2 sorted")
    check(top_numbers_from_preds(preds, top_n=3, min_confidence=0.25) == [1, 2],
          "min_confidence filters")


# --------------------------------------------------------------------------- #
# tally / consensus_numbers / build_votes
# --------------------------------------------------------------------------- #
def test_tally_and_consensus():
    engine_top = {
        "physics": [1, 2, 5],
        "bayes": [1, 2, 8],
        "markov": [1, 5, 10],
    }
    votes = tally_votes(engine_top)
    check(votes[1]["count"] == 3, "number 1 has 3 votes")
    check(votes[2]["count"] == 2, "number 2 has 2 votes")
    check(votes[5]["count"] == 2, "number 5 has 2 votes")
    check(votes[10]["count"] == 1, "number 10 has 1 vote")
    check(set(votes[1]["voters"]) == {"physics", "bayes", "markov"},
          "voters recorded for 1")
    allowed = consensus_numbers(votes, min_agree=2)
    check(allowed == {1, 2, 5}, "min_agree=2 allows 1,2,5")
    check(consensus_numbers(votes, min_agree=3) == {1}, "min_agree=3 allows only 1")


def test_tally_dedup():
    # A single engine voting the same number twice counts once.
    votes = tally_votes({"a": [1, 1, 2]})
    check(votes[1]["count"] == 1, "duplicate vote within an engine counts once")


def test_build_votes_skips_empty():
    dists = {
        "physics": {1: 0.5, 2: 0.3, 5: 0.2},
        "bayes": {1: 0.6, 2: 0.25, 8: 0.15},
        "markov": None,
        "lstm": {},
    }
    votes, n = build_votes(dists, top_n=3)
    check(n == 2, "only 2 engines have usable distributions")
    check(votes[1]["count"] == 2, "1 voted by both present engines")


# --------------------------------------------------------------------------- #
# apply_consensus_filter
# --------------------------------------------------------------------------- #
def _allocs():
    return [
        {"number": 1, "token_bet": 10, "is_positive_ev": True},
        {"number": 8, "token_bet": 5, "is_positive_ev": True},
        {"number": 2, "token_bet": 0, "is_positive_ev": False},
    ]


def test_filter_blocks_low_consensus():
    dists = {
        "physics": {1: 0.5, 2: 0.3, 5: 0.2},
        "bayes": {1: 0.6, 2: 0.25, 5: 0.15},
        "markov": {1: 0.4, 2: 0.35, 5: 0.25},
    }  # 8 is ranked by nobody
    allocs = _allocs()
    out, info = apply_consensus_filter(allocs, dists, min_agree=2, top_n=3)
    check(info["applied"] is True, "filter applied with 3 engines")
    check(info["n_engines"] == 3, "n_engines = 3")
    a1 = next(a for a in out if a["number"] == 1)
    a8 = next(a for a in out if a["number"] == 8)
    check(a1["token_bet"] == 10, "consensus number 1 keeps its stake")
    check(a8["token_bet"] == 0, "number 8 (no consensus) is zeroed")
    check(a8.get("consensus_blocked") is True, "blocked flag set on 8")
    check(a8["is_positive_ev"] is False, "blocked bet marked -EV")
    check(8 in info["blocked"], "8 listed as blocked")
    check(a1["consensus_votes"] == 3, "vote count annotated on 1")


def test_filter_annotates_when_passing():
    dists = {
        "physics": {1: 0.5, 8: 0.3, 5: 0.2},
        "bayes": {1: 0.6, 8: 0.25, 5: 0.15},
    }
    allocs = _allocs()
    out, info = apply_consensus_filter(allocs, dists, min_agree=2, top_n=3)
    a8 = next(a for a in out if a["number"] == 8)
    check(a8["token_bet"] == 5, "8 now has consensus -> stake kept")
    check(a8["consensus_votes"] == 2, "8 annotated with 2 votes")


def test_filter_fails_open_few_engines():
    dists = {"physics": {1: 0.5, 2: 0.3, 5: 0.2}}  # only 1 engine
    allocs = _allocs()
    out, info = apply_consensus_filter(allocs, dists, min_agree=2, top_n=3)
    check(info["applied"] is False, "not applied with < min_agree engines")
    a8 = next(a for a in out if a["number"] == 8)
    check(a8["token_bet"] == 5, "fail-open keeps stakes untouched")
    check(a8["consensus_votes"] == 0, "still annotated (0 votes)")


def test_filter_disabled():
    dists = {
        "physics": {1: 0.5, 2: 0.3, 5: 0.2},
        "bayes": {1: 0.6, 2: 0.25, 5: 0.15},
    }
    allocs = _allocs()
    out, info = apply_consensus_filter(allocs, dists, min_agree=2, top_n=3,
                                       enabled=False)
    check(info["applied"] is False, "disabled -> not applied")
    a8 = next(a for a in out if a["number"] == 8)
    check(a8["token_bet"] == 5, "disabled keeps stakes")
    check("consensus_votes" in a8, "disabled still annotates")


def test_filter_does_not_touch_zero_bets():
    dists = {
        "physics": {5: 0.5, 10: 0.3, 15: 0.2},
        "bayes": {5: 0.6, 10: 0.25, 15: 0.15},
    }
    allocs = _allocs()  # number 2 already token_bet=0
    out, info = apply_consensus_filter(allocs, dists, min_agree=2, top_n=3)
    a2 = next(a for a in out if a["number"] == 2)
    check(a2["token_bet"] == 0, "already-zero stake stays zero")
    check(2 not in info["blocked"], "zero-stake not counted as a fresh block")


def test_filter_min_agree_three():
    dists = {
        "physics": {1: 0.5, 8: 0.3, 5: 0.2},
        "bayes": {1: 0.6, 8: 0.25, 5: 0.15},
        "markov": {1: 0.4, 2: 0.35, 5: 0.25},
    }  # 8 has 2 votes, 1 has 3
    allocs = _allocs()
    out, info = apply_consensus_filter(allocs, dists, min_agree=3, top_n=3)
    a1 = next(a for a in out if a["number"] == 1)
    a8 = next(a for a in out if a["number"] == 8)
    check(a1["token_bet"] == 10, "1 has 3 votes -> kept at min_agree=3")
    check(a8["token_bet"] == 0, "8 has only 2 votes -> blocked at min_agree=3")


if __name__ == "__main__":
    for fn in [
        test_top_from_dist,
        test_top_from_preds,
        test_tally_and_consensus,
        test_tally_dedup,
        test_build_votes_skips_empty,
        test_filter_blocks_low_consensus,
        test_filter_annotates_when_passing,
        test_filter_fails_open_few_engines,
        test_filter_disabled,
        test_filter_does_not_touch_zero_bets,
        test_filter_min_agree_three,
    ]:
        fn()
    total = PASS + FAIL
    print(f"\n_test_consensus: {PASS}/{total} checks passed"
          + ("" if FAIL == 0 else f"  ({FAIL} FAILED)"))
    raise SystemExit(1 if FAIL else 0)
