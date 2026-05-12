import numpy as np
from typing import List, Optional


def compute_metrics(
    rewards: List[float],
    lengths: List[float],
    crashes: List[int],
    mean_speeds: Optional[List[float]] = None,
) -> dict:
    """
    计算评估指标，供 evaluator.py / compare_methods.py 调用。
    - mean_speeds: 每个 episode 的平均速度（可选）
    """
    metrics = {
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) else 0.0,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "std_length": float(np.std(lengths)) if len(lengths) else 0.0,
        "crash_rate": float(np.mean(crashes)) if len(crashes) else 0.0,
        "n_episodes": int(len(rewards)),
        "success_rate": float(1.0 - np.mean(crashes)) if len(crashes) else 0.0,
    }

    if mean_speeds:
        metrics.update(
            {
                "mean_speed": float(np.mean(mean_speeds)),
                "std_speed": float(np.std(mean_speeds)),
            }
        )

    return metrics
