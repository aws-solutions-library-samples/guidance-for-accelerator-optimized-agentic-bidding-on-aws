"""NeMo-RL reward function for bid outcome reinforcement learning.

Computes a scalar reward in [-1.0, 1.0] for each bid outcome, used by the
SageMaker training pipeline (NeMo-RL integration) to optimize ROI:
maximize revenue while minimizing cost.

Reward signal semantics:
- Positive reward for profitable wins (high revenue relative to cost)
- Negative reward for overpayment (cost exceeds revenue)
- Negative reward for losing high-value impressions (bid too low)
- Small positive reward for correctly avoiding low-value impressions

Requirements: 3.3
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.feedback_models import BidShadingOutcomeRecord


# ---------------------------------------------------------------------------
# CTR estimation heuristic
# ---------------------------------------------------------------------------

_DEVICE_CTR: dict[str, float] = {
    "mobile": 0.025,
    "desktop": 0.015,
    "tablet": 0.02,
}

_DEFAULT_CTR: float = 0.02


def _estimate_ctr(outcome: BidShadingOutcomeRecord) -> float:
    """Estimate click-through rate based on device type.

    Uses a simple device-type-based baseline CTR heuristic. In production
    this would be replaced by the model's own CTR prediction, but for
    reward computation a rough proxy is sufficient.

    Returns:
        Estimated CTR in [0, 1].
    """
    return _DEVICE_CTR.get(outcome.device_type.lower(), _DEFAULT_CTR)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------


def compute_rl_reward(outcome: BidShadingOutcomeRecord) -> float:
    """Compute reinforcement learning reward for a bid outcome.

    Used by NeMo-RL during SageMaker training to shape the policy towards
    profitable bidding: maximizing revenue while minimizing cost.

    Args:
        outcome: A validated BidShadingOutcomeRecord from the training dataset.

    Returns:
        A scalar reward in [-1.0, 1.0].
    """
    if not outcome.won:
        # Lost the auction — evaluate whether this was a good or bad decision
        potential_value = outcome.conversion_value_estimate * _estimate_ctr(outcome)

        if potential_value > outcome.shaded_price:
            # We bid too low for a valuable impression — penalize proportionally
            loss_ratio = (potential_value - outcome.shaded_price) / potential_value
            return -0.3 * min(1.0, loss_ratio)
        else:
            # Correct decision: impression wasn't worth more than our bid
            return 0.1  # Small reward for cost avoidance

    # Won the auction — compute ROI-based reward
    cost = outcome.price_paid if outcome.price_paid is not None else outcome.shaded_price
    revenue = 0.0

    if outcome.conversion and outcome.conversion_value is not None:
        revenue = outcome.conversion_value
    elif outcome.click:
        # Partial credit for click without conversion
        revenue = outcome.conversion_value_estimate * 0.1
    elif outcome.impression:
        # Minimal credit for impression only
        revenue = outcome.conversion_value_estimate * 0.01

    # ROI signal: (revenue - cost) / cost, bounded to prevent extreme values
    roi = (revenue - cost) / max(cost, 0.01)

    # Scale ROI into reward range
    reward = max(-1.0, min(1.0, roi * 0.5))

    # Bonus for winning at low cost (efficient shading)
    savings_ratio = 1.0 - (cost / outcome.original_price) if outcome.original_price > 0 else 0.0
    reward += 0.2 * savings_ratio

    # Final clamp to ensure [-1.0, 1.0] bounds
    return max(-1.0, min(1.0, reward))
