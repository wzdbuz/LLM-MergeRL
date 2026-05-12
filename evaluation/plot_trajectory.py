"""
轨迹可视化：对比 PPO Baseline 与 LLM-reward 的自车行驶轨迹
在项目根目录下运行：python plot_trajectory.py
输出：results/figures/trajectory_comparison.svg
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from config.config import get_config
from env.highway_wrapper import make_env

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 配置 ──────────────────────────────────────────────────────────────────────

EVAL_SEEDS = [137, 256, 891]
N_EPISODES = len(EVAL_SEEDS)
SAVE_PATH  = "results/figures/trajectory_comparison.svg"

METHODS = {
    "PPO Baseline": {
        "model_path": "results/checkpoints/ppo_baseline_seed0/best_model",
        "mode":       "baseline",
        "color":      "#378ADD",
        "ls":         "-",
    },
    "LLM-reward": {
        "model_path": "results/checkpoints/ppo_llm_reward_seed0/best_model",
        "mode":       "llm_reward",
        "color":      "#E24B4A",
        "ls":         "-",
    },
}

# ── 轨迹采集 ──────────────────────────────────────────────────────────────────

def collect_trajectory(model_path, mode, seed, use_fake_llm=True):
    env_config, _, _ = get_config("baseline")
    env = make_env(env_config, seed=seed, mode=mode, use_fake_llm=use_fake_llm)
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)

    positions, lanes = [], []
    crashed = False
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        try:
            vehicle = env.unwrapped.vehicle
            x, y = float(vehicle.position[0]), float(vehicle.position[1])
            positions.append((x, y))
            lane_idx = vehicle.lane_index[2] if vehicle.lane_index else 0
            lanes.append(lane_idx)
        except Exception:
            pass
        if info.get("crashed", False):
            crashed = True

    env.close()
    return positions, lanes, crashed

# ── 采集所有轨迹 ──────────────────────────────────────────────────────────────

print("Collecting trajectories...")
all_trajs = {name: [] for name in METHODS}

for name, cfg in METHODS.items():
    for seed in EVAL_SEEDS:
        print(f"  {name} seed={seed} ...")
        pos, lanes, crashed = collect_trajectory(cfg["model_path"], cfg["mode"], seed)
        all_trajs[name].append({"positions": pos, "lanes": lanes,
                                 "crashed": crashed, "seed": seed})

# ── 绘图 ──────────────────────────────────────────────────────────────────────

print("Plotting...")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

fig, axes = plt.subplots(N_EPISODES, 1, figsize=(14, 3.5 * N_EPISODES))
if N_EPISODES == 1:
    axes = [axes]

for ep_idx, seed in enumerate(EVAL_SEEDS):
    ax = axes[ep_idx]

    # 道路背景
    ax.axhspan(-2, 6,  color="#f5f5f5", zorder=0)
    ax.axhspan(6,  12, color="#eeeeee", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axhline(4, color="#cccccc", linewidth=0.8, zorder=1)
    ax.axhline(8, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # 轨迹
    for name, cfg in METHODS.items():
        traj      = all_trajs[name][ep_idx]
        positions = traj["positions"]
        crashed   = traj["crashed"]
        if not positions:
            continue

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        ax.plot(xs, ys, color=cfg["color"], linestyle=cfg["ls"],
                linewidth=2.0, label=name, zorder=3, alpha=0.85)
        ax.scatter(xs[0], ys[0], color=cfg["color"], s=60, marker="o", zorder=5)
        end_marker = "x" if crashed else "s"
        ax.scatter(xs[-1], ys[-1], color=cfg["color"], s=80,
                   marker=end_marker, zorder=5,
                   linewidths=2 if crashed else 1)

    # x轴范围
    all_xs = []
    for name in METHODS:
        pos = all_trajs[name][ep_idx]["positions"]
        if pos:
            all_xs.extend([p[0] for p in pos])
    if all_xs:
        ax.set_xlim(min(all_xs) - 10, max(all_xs) + 10)

    ax.set_ylim(-4, 14)
    ax.set_xlabel("纵向位置 (m)", fontsize=12)
    ax.set_ylabel("横向位置 (m)", fontsize=12)
    ax.set_title(f"回合 {ep_idx + 1}  (seed={seed})", fontsize=15, fontweight="bold")
    ax.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=cfg["color"], linewidth=2, label=name)
            for name, cfg in METHODS.items()
        ] + [
            plt.Line2D([0], [0], marker="o", color="gray", linestyle="none",
                       markersize=6, label="起点"),
            plt.Line2D([0], [0], marker="s", color="gray", linestyle="none",
                       markersize=6, label="终点"),
        ],
        fontsize=9, loc="upper left", framealpha=0.9
    )

plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches="tight", format="svg")
plt.close()
print(f"Saved: {SAVE_PATH}")