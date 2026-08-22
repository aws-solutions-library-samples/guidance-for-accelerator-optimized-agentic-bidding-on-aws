"""Bounded epsilon-greedy exploration for the Yield Optimizer.

The genesis model (and any future model that has converged to a stable
"no change" recommendation) can settle into floor_multiplier==1.0,
margin_value==0.0 for every input -- which business-rules.md BR-3/BR-4's
no-op-avoidance logic deliberately never turns into a mutation. That
means the container never generates outcome data for itself to learn
from: a real cold-start deadlock (no mutation -> no
DealYieldOutcomeEvent -> no training data -> the model never learns to
recommend anything else), not a bug in the no-op logic itself.

Exploration breaks that deadlock the way any epsilon-greedy bandit does:
with probability epsilon, apply a small, bounded, real random
perturbation around the model's own prediction instead of using it
verbatim, so real variation exists to train the next model version on.
Continues to apply (at the same small epsilon) once a real trained model
exists too -- this is standard explore/exploit behavior for a live
bandit, not a genesis-only workaround.

This is disclosed, not fabricated: the caller (app.py) marks any response
containing an explored deal with a ":explore" model_version suffix, so
DealYieldOutcomeEvents and later analyses can distinguish "the model's
own prediction" from "an exploratory probe" -- training on mislabeled
exploration data (as if it were a confident recommendation) would bias
the learned policy, so the distinction has to survive into the emitted
event, not just exist here.

Deliberately independent of any specific model architecture -- this
module only perturbs the two scalar outputs (floor_multiplier,
margin_value) already returned by
triton_inference.predict_yield_adjustment(), regardless of what produced
them.
"""

from __future__ import annotations

import random

# Hard safety clamps applied AFTER perturbation, independent of the
# configured exploration bound -- protects against a pathological
# combination of a bad base prediction plus a misconfigured (too-large)
# exploration bound ever producing an absurd floor/margin. These are NOT
# the same as BR-7's bidfloor>=0 clamp (applied downstream in app.py on
# the final adjusted_bidfloor); this clamps the multiplier/value itself.
FLOOR_MULTIPLIER_MIN = 0.5
FLOOR_MULTIPLIER_MAX = 1.5
MARGIN_VALUE_MIN = -0.5
MARGIN_VALUE_MAX = 0.5


def apply_exploration(
    floor_multiplier: float,
    margin_value: float,
    *,
    rng: random.Random,
    epsilon: float,
    floor_bound: float,
    margin_bound: float,
) -> tuple[float, float, bool]:
    """Possibly perturbs (floor_multiplier, margin_value) for one deal.

    With probability ``epsilon``, returns a bounded-random perturbation of
    each input (independently drawn, but from the SAME explore/exploit
    coin flip -- one flip per deal, not two) instead of the input
    unchanged. Floor and margin either both explore together or neither
    does for a given deal, since they describe the same yield-optimization
    decision even though ADJUST_DEAL_FLOOR/ADJUST_DEAL_MARGIN remain
    independent, atomic mutations per BR-5.

    Args:
        floor_multiplier: The model's own floor_multiplier prediction.
        margin_value: The model's own margin_value prediction.
        rng: Caller-supplied ``random.Random`` instance -- real entropy in
            production, a seeded instance in tests (dependency injection,
            not a global monkeypatch).
        epsilon: Probability in [0, 1] of exploring this deal. epsilon<=0
            always returns the inputs unchanged.
        floor_bound: Exploration perturbs floor_multiplier by
            Uniform(-floor_bound, +floor_bound) before clamping.
        margin_bound: Exploration perturbs margin_value by
            Uniform(-margin_bound, +margin_bound) before clamping.

    Returns:
        (floor_multiplier, margin_value, explored). ``explored`` is True
        iff a perturbation was actually applied, so the caller can
        honestly disclose it -- never silently blend an exploratory probe
        into what looks like a model recommendation.
    """
    if epsilon <= 0.0 or rng.random() >= epsilon:
        return floor_multiplier, margin_value, False

    explored_floor = floor_multiplier + rng.uniform(-floor_bound, floor_bound)
    explored_floor = max(FLOOR_MULTIPLIER_MIN, min(FLOOR_MULTIPLIER_MAX, explored_floor))

    explored_margin = margin_value + rng.uniform(-margin_bound, margin_bound)
    explored_margin = max(MARGIN_VALUE_MIN, min(MARGIN_VALUE_MAX, explored_margin))

    return explored_floor, explored_margin, True
