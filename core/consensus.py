# -*- coding: utf-8 -*-
"""PROMPT 17: multi-engine consensus filter.

IDEA: a single model's confidence is weak evidence. If FOUR independent models
(physics / Bayesian frequency / Markov / LSTM) are each asked for their top
picks, a number that several of them INDEPENDENTLY rank highly is a sturdier
signal than one model shouting alone. This filter only lets a stake stand on a
number when at least K engines agree it is a top pick; everything else is zeroed
(SKIP), exactly like the reality-check gate.

HONEST NOTE: on a fair, memoryless wheel even unanimous agreement does not make
a number "due". Consensus reduces single-model overconfidence / noise; it does
NOT create an edge that the wheel's payouts don't already give. It is a
risk filter, not a profit engine.

Pure & GUI-agnostic: stdlib only. Works off normalized {number: prob} dicts
(e.g. ContinuousLearningEngine.model_distributions) and the sized allocation
list produced by the active engine.
"""


def top_numbers_from_dist(dist, top_n=3, min_prob=0.0):
    """Top-N numbers (by probability) from a {number: prob} distribution."""
    if not dist:
        return []
    items = sorted(dist.items(), key=lambda kv: (kv[1] or 0.0), reverse=True)
    out = []
    for num, prob in items[:top_n]:
        if (prob or 0.0) >= min_prob:
            try:
                out.append(int(num))
            except (TypeError, ValueError):
                continue
    return out


def top_numbers_from_preds(predictions, top_n=3, number_key="number",
                           conf_key="confidence", min_confidence=0.0):
    """Top-N numbers from an engine.predict_next() style list."""
    rows = sorted(predictions, key=lambda p: (p.get(conf_key, 0) or 0.0),
                  reverse=True)
    out = []
    for p in rows[:top_n]:
        if (p.get(conf_key, 0) or 0.0) >= min_confidence:
            num = p.get(number_key)
            if num is not None:
                try:
                    out.append(int(num))
                except (TypeError, ValueError):
                    continue
    return out


def tally_votes(engine_top):
    """Count how many engines vote for each number.

    engine_top: dict engine_name -> list[int] of that engine's top picks.
    Returns {number: {"count": int, "voters": [names]}}.
    """
    votes = {}
    for name, nums in engine_top.items():
        for num in set(nums):
            v = votes.setdefault(int(num), {"count": 0, "voters": []})
            v["count"] += 1
            v["voters"].append(name)
    return votes


def consensus_numbers(votes, min_agree=2):
    """Set of numbers backed by at least ``min_agree`` engines."""
    return {num for num, v in votes.items() if v["count"] >= min_agree}


def build_votes(engine_distributions, top_n=3, min_prob=0.0):
    """Turn {engine: {number: prob}} into (votes, n_engines).

    Engines with an empty/None distribution are skipped (they cannot vote).
    """
    engine_top = {}
    for name, dist in engine_distributions.items():
        if not dist:
            continue
        engine_top[name] = top_numbers_from_dist(dist, top_n=top_n,
                                                 min_prob=min_prob)
    return tally_votes(engine_top), len(engine_top)


def apply_consensus_filter(predictions, engine_distributions, min_agree=2,
                           top_n=3, min_prob=0.0, enabled=True,
                           number_key="number", bet_key="token_bet"):
    """Zero stakes on numbers fewer than ``min_agree`` engines rank in top-N.

    Every prediction is annotated with ``consensus_votes`` (int) and
    ``consensus_voters`` (list). Bets that are filtered out also get
    ``consensus_blocked = True`` and ``is_positive_ev = False``.

    FAILS OPEN (annotate only, no filtering) when disabled or when fewer than
    ``min_agree`` engines are available -- we never block bets just because
    models are missing.

    Returns ``(predictions, info)`` where info carries n_engines / min_agree /
    allowed / blocked / votes / applied.
    """
    votes, n_engines = build_votes(engine_distributions, top_n=top_n,
                                   min_prob=min_prob)
    info = {
        "n_engines": n_engines,
        "min_agree": min_agree,
        "top_n": top_n,
        "votes": votes,
        "allowed": set(),
        "blocked": [],
        "applied": False,
    }

    def _annotate(p):
        num = p.get(number_key)
        try:
            num = int(num) if num is not None else None
        except (TypeError, ValueError):
            num = None
        v = votes.get(num, {"count": 0, "voters": []})
        p["consensus_votes"] = v["count"]
        p["consensus_voters"] = list(v["voters"])
        return num

    if not enabled or n_engines < min_agree:
        for p in predictions:
            _annotate(p)
        return predictions, info

    allowed = consensus_numbers(votes, min_agree=min_agree)
    info["allowed"] = allowed
    info["applied"] = True
    for p in predictions:
        num = _annotate(p)
        if (p.get(bet_key, 0) or 0) > 0 and num not in allowed:
            info["blocked"].append(num)
            p[bet_key] = 0
            p["is_positive_ev"] = False
            p["consensus_blocked"] = True
    return predictions, info
