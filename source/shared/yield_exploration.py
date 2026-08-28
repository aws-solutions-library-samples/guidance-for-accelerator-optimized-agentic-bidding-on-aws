"""Bounded epsilon-greedy exploration shared by the two Yield Optimizer containers.

The genesis model (and any future model that has converged to a stable
"no change" recommendation) can settle into floor_multiplier==1.0 /
margin_value==0.0 for every input -- which business-rules.md BR-3/BR-4's
no-op-avoidance logic deliberately never turns into a mutation. That means the
container never generates outcome data for itself to learn from: a real
cold-start deadlock (no mutation -> no DealYieldOutcomeEvent -> no training
data -> the model never learns to recommend anything else), not a bug in the
no-op logic itself.

Exploration breaks that deadlock the way any epsilon-greedy bandit does: with
probability epsilon, apply a small, bounded, real random perturbation around
the model's own prediction instead of using it verbatim, so real variation
exists to train the next model version on. It continues to apply (at the same
small epsilon) once a real trained model exists -- standard explore/exploit
behavior for a live bandit, not a genesis-only workaround.

This is disclosed, not fabricated: the caller marks any response containing an
explored deal with a ":explore" suffix on model_version, so
DealYieldOutcomeEvents and later analyses can distinguish "the model's own
prediction" from "an exploratory probe". Training on mislabeled exploration
data (as if it were a confident recommendation) would bias the learned policy,
so the distinction has to survive into the emitted event, not just exist here.

SINGLE-SCALAR BY DESIGN (changed at the container split). Before the split,
one container held both outputs and perturbed them from ONE coin flip, so floor
and margin always explored together or not at all. Two independent containers
handle two independent requests with two independent RNGs, so a shared flip is
structurally impossible -- each model now decides on its own. That is the more
faithful reading of BR-5 anyway (ADJUST_DEAL_FLOOR and ADJUST_DEAL_MARGIN are
independent, atomic mutations, never gated on each other); the previous
"same decision, so same flip" coupling was the weaker justification. Practical
consequence for anyone reading training data: for a given deal you may now see
an explored floor alongside an unexplored margin. Both are still individually
disclosed via their own event's model_version suffix, so nothing is
mislabeled -- the correlation between the two simply no longer exists.

Deliberately independent of any specific model architecture: this module only
perturbs a scalar prediction, regardless of what produced it.
"""

from __future__ import annotations

import random

# Hard safety clamps applied AFTER perturbation, independent of the configured
# exploration bound -- protects against a pathological combination of a bad
# base prediction plus a misconfigured (too-large) exploration bound ever
# producing an absurd floor/margin. These are NOT the same as BR-7's
# bidfloor>=0 clamp (applied downstream on the final adjusted_bidfloor); these
# clamp the multiplier/value itself.
#
# Each container imports only the pair that applies to its own output.
FLOOR_MULTIPLIER_MIN = 0.5
FLOOR_MULTIPLIER_MAX = 1.5
MARGIN_VALUE_MIN = -0.5
MARGIN_VALUE_MAX = 0.5


def parse_explore_override(model_params: dict | None) -> bool | None:
    """Read ``ext.model_params.explore`` without ever guessing an intent.

    Returns True/False only when the caller supplied a real bool. Anything
    else -- absent, null, or an unexpected type such as the string "true"
    arriving from a hand-written client -- returns None, meaning "no
    override", so ``resolve_effective_epsilon`` falls back to the load-test
    signal. Coercing a stray string to True here would silently arm
    exploration on traffic the caller never asked to perturb.
    """
    override = (model_params or {}).get("explore")
    if isinstance(override, bool):
        return override
    return None


def resolve_effective_epsilon(
    is_load_test: bool,
    explore_override: bool | None,
    configured_epsilon: float,
) -> float:
    """Decide whether exploration is armed for this specific request.

    Two independent ways a caller can arm it:
    - ``is_load_test``: the orchestrator's load-test invocation path set the
      X-Load-Test header (see shared/load_test_context.py) -- structurally
      load-test-only, never live auction traffic.
    - ``explore_override``: the caller set ``ext.model_params.explore``
      explicitly (True or False). The demo Scenario card's Explore toggle uses
      this, defaulting to True so a single scenario Send against an untrained
      model can also produce a real mutation, not just a load test.

    An explicit ``explore_override=False`` always wins, so a user who has since
    trained a real model can force exploration off and see that model's
    unperturbed prediction. Otherwise either signal being "on" arms epsilon.
    Neither signal set (real production traffic hitting this same endpoint,
    with no override and no load-test header) always resolves to 0.0 -- this is
    what keeps exploration off by default for traffic the container cannot
    otherwise identify as load-test or demo-originated, REGARDLESS of how
    the epsilon env var is configured.
    """
    if explore_override is False:
        return 0.0
    if is_load_test or explore_override is True:
        return configured_epsilon
    return 0.0


def apply_exploration(
    value: float,
    *,
    rng: random.Random,
    epsilon: float,
    bound: float,
    clamp_min: float,
    clamp_max: float,
) -> tuple[float, bool]:
    """Possibly perturbs one scalar prediction for one deal.

    With probability ``epsilon``, returns a bounded-random perturbation of
    ``value`` (then clamped to [clamp_min, clamp_max]) instead of ``value``
    unchanged.

    Args:
        value: The model's own scalar prediction (a floor_multiplier or a
            margin_value).
        rng: Caller-supplied ``random.Random`` instance -- real entropy in
            production, a seeded instance in tests (dependency injection, not
            a global monkeypatch).
        epsilon: Probability in [0, 1] of exploring this deal. epsilon<=0
            always returns the input unchanged.
        bound: Perturbs ``value`` by Uniform(-bound, +bound) before clamping.
        clamp_min: Hard lower safety clamp applied after perturbation.
        clamp_max: Hard upper safety clamp applied after perturbation.

    Returns:
        (value, explored). ``explored`` is True iff a perturbation was actually
        applied, so the caller can disclose it -- never silently blend an
        exploratory probe into what looks like a model recommendation.
    """
    if epsilon <= 0.0 or rng.random() >= epsilon:
        return value, False

    explored = value + rng.uniform(-bound, bound)
    explored = max(clamp_min, min(clamp_max, explored))
    return explored, True
