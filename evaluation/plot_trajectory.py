"""
轨迹可视化：对比 PPO Baseline 与 LLM-reward 的自车行驶轨迹
左列PPO Baseline，右列LLM-reward，每行一个episode
在项目根目录下运行：python plot_trajectory.py
输出：results/figures/trajectory_comparison.svg
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from config.config import get_config
from env.highway_wrapper import make_env

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 配置 ──────────────────────────────────────────────────────────────────────

EVAL_SEEDS = [7, 256, 891]
N_EPISODES = len(EVAL_SEEDS)
SAVE_PATH  = "results/figures/trajectory_comparison.svg"

METHODS = {
    "PPO Baseline": {
        "model_path": "results/checkpoints/ppo_baseline_seed0/best_model",
        "mode":       "baseline",
        "color":      "#378ADD",
    },
    "LLM-reward": {
        "model_path": "results/checkpoints/ppo_llm_reward_seed0/best_model",
        "mode":       "llm_reward",
        "color":      "#E24B4A",
    },
}

METHOD_NAMES = list(METHODS.keys())

# ── 轨迹采集 ──────────────────────────────────────────────────────────────────

def collect_trajectory(model_path, mode, seed, use_fake_llm=True):
    env_config, _, _ = get_config("baseline")
    env = make_env(env_config, seed=seed, mode=mode, use_fake_llm=use_fake_llm)
    model = PPO.load(model_path)
    obs, _ = env.reset(seed=seed)

    positions = []
    crashed = False
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        try:
            vehicle = env.unwrapped.vehicle
            x, y = float(vehicle.position[0]), float(vehicle.position[1])
            positions.append((x, y))
        except Exception:
            pass
        if info.get("crashed", False):
            crashed = True

    env.close()
    return positions, crashed

# ── 采集所有轨迹 ──────────────────────────────────────────────────────────────

print("Collecting trajectories...")
all_trajs = {name: [] for name in METHODS}

for name, cfg in METHODS.items():
    for seed in EVAL_SEEDS:
        print(f"  {name} seed={seed} ...")
        pos, crashed = collect_trajectory(cfg["model_path"], cfg["mode"], seed)
        all_trajs[name].append({"positions": pos, "crashed": crashed, "seed": seed})

# ── 绘图（左右两列）──────────────────────────────────────────────────────────

print("Plotting...")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

fig, axes = plt.subplots(N_EPISODES, 2, figsize=(16, 3.5 * N_EPISODES))

for ep_idx, seed in enumerate(EVAL_SEEDS):
    for col_idx, name in enumerate(METHOD_NAMES):
        ax = axes[ep_idx][col_idx]
        cfg = METHODS[name]
        traj = all_trajs[name][ep_idx]
        positions = traj["positions"]
        crashed = traj["crashed"]

        # 道路背景
        ax.axhspan(-2, 6,  color="#f5f5f5", zorder=0)
        ax.axhspan(6,  12, color="#eeeeee", zorder=0)
        ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
        ax.axhline(4, color="#cccccc", linewidth=0.8, zorder=1)
        ax.axhline(8, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            ax.plot(xs, ys, color=cfg["color"], linestyle="-",
                    linewidth=2.5, zorder=3, alpha=0.9)
            ax.scatter(xs[0], ys[0], color=cfg["color"], s=60,
                       marker="o", zorder=5)
            end_marker = "x" if crashed else "s"
            ax.scatter(xs[-1], ys[-1], color=cfg["color"], s=80,
                       marker=end_marker, zorder=5,
                       linewidths=2 if crashed else 1)

            ax.set_xlim(min(xs) - 10, max(xs) + 10)
        else:
            ax.set_xlim(50, 300)

        ax.set_ylim(-4, 14)
        ax.set_xlabel("纵向位置 (m)", fontsize=10)
        ax.set_ylabel("横向位置 (m)", fontsize=10)

        # 标题：第一行显示方法名，每行显示seed
        if ep_idx == 0:
            ax.set_title(f"{name}\n回合 {ep_idx+1}  (seed={seed})",
                        fontsize=11, fontweight="bold",
                        color=cfg["color"])
        else:
            ax.set_title(f"回合 {ep_idx+1}  (seed={seed})",
                        fontsize=11, fontweight="bold")

        ax.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 图例
        legend_handles = [
            plt.Line2D([0], [0], color=cfg["color"], linewidth=2.5, label=name),
            plt.Line2D([0], [0], marker="o", color="gray", linestyle="none",
                       markersize=6, label="起点"),
            plt.Line2D([0], [0], marker="s", color="gray", linestyle="none",
                       markersize=6, label="正常终止"),
            plt.Line2D([0], [0], marker="x", color="gray", linestyle="none",
                       markersize=8, markeredgewidth=2, label="碰撞终止"),
        ]
        ax.legend(handles=legend_handles, fontsize=8,
                  loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches="tight", format="svg")
plt.close()
print(f"Saved: {SAVE_PATH}")