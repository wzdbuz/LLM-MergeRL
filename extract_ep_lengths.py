"""
从各方法的 evaluations.npz 中提取平均回合步长
取最后10次评估的ep_lengths，计算均值，汇总各方法3次训练的均值±标准差
运行方式：在项目根目录下执行 python extract_ep_lengths.py
"""

import os
import numpy as np


def latest_subdir(path: str):
    if not os.path.exists(path):
        return None
    subdirs = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def get_ep_length_median_last10(npz_path: str) -> float:
    """取最后10次评估的ep_lengths，展平后取中位数"""
    data = np.load(npz_path)
    ep_lengths = data["ep_lengths"]  # shape: (n_evals, 6)
    last10 = ep_lengths[-10:]        # 最后10次评估，shape: (10, 6)
    flat = last10.flatten()          # 展平为60个值
    return float(np.median(flat))


# 各方法对应的log目录和seed列表
METHODS = {
    "PPO Baseline":  ("ppo_baseline",  [0, 1, 2]),
    "LLM-state":     ("ppo_llm_state", [0, 1, 2]),
    "LLM-reward":    ("ppo_llm_reward",[0, 1, 2]),
    "DQN Baseline":  ("dqn_baseline",  [0, 1, 2]),
}

print(f"{'方法':<16} {'seed0':>8} {'seed1':>8} {'seed2':>8} {'均值':>8} {'标准差':>8}")
print("-" * 60)

for method_name, (prefix, seeds) in METHODS.items():
    medians = []
    for seed in seeds:
        log_root = os.path.join("results", "logs", f"{prefix}_seed{seed}")
        latest = latest_subdir(log_root)
        if latest is None:
            medians.append(None)
            continue
        npz_path = os.path.join(latest, "evaluations.npz")
        if not os.path.exists(npz_path):
            medians.append(None)
            continue
        val = get_ep_length_median_last10(npz_path)
        medians.append(val)

    valid = [v for v in medians if v is not None]
    if not valid:
        print(f"{method_name:<16} {'缺失':>8}")
        continue

    seed_strs = [f"{v:>8.2f}" if v is not None else f"{'缺失':>8}" for v in medians]
    mean_val = float(np.mean(valid))
    std_val  = float(np.std(valid))
    print(f"{method_name:<16} {''.join(seed_strs)} {mean_val:>8.2f} {std_val:>8.3f}")