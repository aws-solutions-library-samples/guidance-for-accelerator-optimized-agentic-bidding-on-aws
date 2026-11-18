"""NeMo-RL reward function for bid outcome reinforcement learning (tensor version).

Computes vectorized rewards from bid outcomes for REINFORCE policy gradient.
Used inside the SageMaker training container alongside NeMo Framework.

Reward types:
- "roi": Optimizes return on investment (revenue - cost) / cost
- "ctr": Optimizes click-through rate prediction accuracy
- "revenue": Optimizes raw revenue per bid
"""

from __future__ import annotations

import torch


def compute_reward(
    scores: torch.Tensor,
    outcomes: torch.Tensor,
    reward_type: str = "roi",
) -> torch.Tensor:
    """Compute per-sample rewards from model scores and bid outcomes.

    Args:
        scores: Model predictions [batch_size] — used as bid price signals.
        outcomes: Outcome tensor [batch_size, 3] — columns: [win, price_paid, revenue].
        reward_type: One of "roi", "ctr", "revenue".

    Returns:
        Reward tensor [batch_size] in approximately [-1.0, 1.0].
    """
    wins = outcomes[:, 0]         # 1.0 if won, 0.0 if lost
    price_paid = outcomes[:, 1]   # Actual cost
    revenue = outcomes[:, 2]      # Revenue generated

    if reward_type == "roi":
        return _roi_reward(wins, price_paid, revenue, scores)
    elif reward_type == "ctr":
        return _ctr_reward(wins, scores)
    elif reward_type == "revenue":
        return _revenue_reward(wins, revenue, price_paid)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def _roi_reward(
    wins: torch.Tensor,
    price_paid: torch.Tensor,
    revenue: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """ROI-based reward: maximize (revenue - cost) / cost for wins.

    - Profitable wins: positive reward proportional to ROI
    - Unprofitable wins (overpaid): negative reward
    - Losses on high-value opportunities: small negative
    - Correct avoidance of low-value: small positive
    """
    cost = torch.clamp(price_paid, min=0.01)
    roi = (revenue - cost) / cost

    # Wins: reward based on ROI, clamped
    win_reward = torch.clamp(roi * 0.5, min=-1.0, max=1.0)

    # Losses: penalize missing high-value opportunities
    # Use score as proxy for estimated value — high score + loss = missed opportunity
    loss_penalty = -0.2 * torch.clamp(scores, min=0.0, max=1.0)

    # Combine: wins get ROI reward, losses get penalty
    reward = wins * win_reward + (1.0 - wins) * loss_penalty

    return torch.clamp(reward, min=-1.0, max=1.0)


def _ctr_reward(wins: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """CTR prediction reward: correct predictions get positive reward.

    Reward is higher when score aligns with outcome (high score + win, low score + loss).
    """
    # Binary cross-entropy style reward (not loss — higher is better)
    correct = wins * scores + (1.0 - wins) * (1.0 - scores)
    return 2.0 * correct - 1.0  # Map [0, 1] → [-1, 1]


def _revenue_reward(
    wins: torch.Tensor,
    revenue: torch.Tensor,
    price_paid: torch.Tensor,
) -> torch.Tensor:
    """Revenue reward: maximize revenue per bid, penalize high cost."""
    # Normalize revenue to roughly [-1, 1] using a soft cap
    net = revenue - price_paid
    cap = torch.clamp(torch.abs(net).mean(), min=1.0)  # adaptive scale
    return torch.clamp(net / cap, min=-1.0, max=1.0)
