"""PROMPT 14: the manual % input as a proper Dirichlet prior.

The old code blended the user's manual probabilities into the active engine
with a HARD-CODED weight of 0.35 (``0.65*engine + 0.35*manual``). That weight
ignored how much real data the engine had seen, so a manual hunch counted the
same after 3 spins as after 300.

A Dirichlet prior fixes this honestly. We treat the manual distribution as
``strength`` pseudo-observations (alpha = manual_prob * strength). The engine's
prediction acts like ``data_count`` real observations. The posterior mean is

    posterior_n = (manual_prob_n * strength + engine_prob_n * data_count)
                  / (strength + data_count)

which, for a fixed number n, is the convex blend

    posterior_n = (1 - w) * engine_n + w * manual_n,   w = strength / (strength + data_count)

Properties (all desirable, none hand-tuned):
  * strength = 0  -> w = 0      -> pure engine (manual ignored).
  * data_count = 0 (cold start) -> w = 1 -> pure manual prior.
  * strength == data_count      -> w = 0.5 (equal say).
  * as data grows, w -> 0: evidence eventually overrules any hunch.
  * at the default strength 27 with ~50 spins, w ~= 0.35 -> backwards-compatible
    with the old fixed weight, but now it MOVES with the data.

HONEST NOTE: a prior cannot manufacture signal on a fair, memoryless wheel. The
benefit is purely cold-start regularisation and letting the user encode a
belief that *fades automatically* as real data accumulates.
"""


def dirichlet_prior_weight(strength, data_count):
    """Manual-prior weight w = strength / (strength + data_count), in [0, 1]."""
    s = max(0.0, float(strength))
    n = max(0.0, float(data_count))
    denom = s + n
    if denom <= 0:
        return 0.0
    return s / denom


def blend_confidence(engine_conf, manual_conf, weight):
    """Convex blend of one number's engine vs manual confidence."""
    w = min(1.0, max(0.0, float(weight)))
    return (1.0 - w) * float(engine_conf) + w * float(manual_conf)


def has_manual_input(manual_pct):
    """True when the user supplied a non-empty manual distribution."""
    if not manual_pct:
        return False
    try:
        return sum(float(v) for v in manual_pct.values()) > 0
    except (TypeError, ValueError):
        return False


def apply_manual_prior(predictions, manual_pct, strength, data_count,
                       number_key="number", conf_key="confidence"):
    """Blend the manual Dirichlet prior into a list of prediction dicts.

    Mutates each dict's confidence in place using the data-dependent weight and
    returns ``(predictions, weight)``. No-op (weight unchanged but applied as 0)
    when the user supplied no manual input.
    """
    w = dirichlet_prior_weight(strength, data_count)
    if w <= 0 or not has_manual_input(manual_pct):
        return predictions, w
    for p in predictions:
        manual_conf = float(manual_pct.get(p.get(number_key), 0.0) or 0.0) / 100.0
        p[conf_key] = blend_confidence(p.get(conf_key, 0.0), manual_conf, w)
    return predictions, w


def clamp_strength(value, lo, hi, default):
    """Clamp a slider value into [lo, hi]; fall back to default on bad input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(min(hi, max(lo, v)))
