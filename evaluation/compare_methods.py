"""
多seed训练曲线对比绘图脚本
在项目根目录下运行：python evaluation/compare_methods.py
"""

import os
import csv
import sys
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 配置 ──────────────────────────────────────────────────────────────────────

METHODS = {
    "PPO Baseline": ("ppo_baseline",   [0, 1, 2], "#378ADD", "-"),
    "LLM-state":    ("ppo_llm_state",  [0, 1, 2], "#1D9E75", "-"),
    "LLM-reward":   ("ppo_llm_reward", [0, 1, 2], "#E24B4A", "-"),
}

SAVE_PATH             = "results/figures"
CONVERGENCE_THRESHOLD = 15.0

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def latest_subdir(path):
    if not os.path.exists(path):
        return None
    subdirs = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None


def load_progress(progress_path):
    train_steps, train_rewards, ev_steps, ev_values = [], [], [], []
    with open(progress_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("time/total_timesteps") and row.get("rollout/ep_rew_mean"):
                train_steps.append(int(row["time/total_timesteps"]))
                train_rewards.append(float(row["rollout/ep_rew_mean"]))
            if row.get("time/total_timesteps") and row.get("train/explained_variance"):
                ev_steps.append(int(row["time/total_timesteps"]))
                ev_values.append(float(row["train/explained_variance"]))
    return (np.array(train_steps), np.array(train_rewards),
            np.array(ev_steps),    np.array(ev_values))


def interpolate_to_common(steps, values, common_steps):
    return np.interp(common_steps, steps, values)


def find_convergence_step(steps, rewards, threshold):
    for s, r in zip(steps, rewards):
        if r >= threshold:
            return s
    return None


def style_ax(ax, title, xlabel, ylabel):
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, color="#e0e0e0", linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"
        )
    )

# ── 数据加载 ──────────────────────────────────────────────────────────────────

print("Loading training data...")
all_data          = {}
convergence_table = {}

for method_name, (prefix, seeds, color, ls) in METHODS.items():
    train_steps_list, train_rewards_list = [], []
    ev_steps_list,    ev_values_list     = [], []
    conv_steps = []

    for seed in seeds:
        log_root = os.path.join("results", "logs", f"{prefix}_seed{seed}")
        latest   = latest_subdir(log_root)
        if latest is None:
            print(f"  [skip] {prefix}_seed{seed}: no log dir")
            continue

        progress_path = os.path.join(latest, "progress.csv")
        if not os.path.exists(progress_path):
            print(f"  [skip] {prefix}_seed{seed}: no progress.csv")
            continue

        ts, rw, es, ev = load_progress(progress_path)
        if len(ts) == 0:
            continue

        train_steps_list.append(ts)
        train_rewards_list.append(rw)
        if len(es) > 0:
            ev_steps_list.append(es)
            ev_values_list.append(ev)

        c = find_convergence_step(ts, rw, CONVERGENCE_THRESHOLD)
        if c is not None:
            conv_steps.append(c)
        print(f"  {prefix}_seed{seed}: {len(ts)} points, "
              f"convergence={'N/A' if c is None else f'{c}steps'}")

    all_data[method_name] = {
        "train_steps":   train_steps_list,
        "train_rewards": train_rewards_list,
        "ev_steps":      ev_steps_list,
        "ev_values":     ev_values_list,
        "color": color,
        "ls":    ls,
    }
    convergence_table[method_name] = conv_steps

# ── 图1：训练奖励曲线 ──────────────────────────────────────────────────────────

print("\nPlotting training reward curves...")
os.makedirs(SAVE_PATH, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))

for method_name, d in all_data.items():
    if not d["train_steps"]:
        continue
    color, ls = d["color"], d["ls"]

    max_step = max(s[-1] for s in d["train_steps"])
    common   = np.linspace(0, max_step, 500)
    interped = np.array([interpolate_to_common(s, r, common)
                         for s, r in zip(d["train_steps"], d["train_rewards"])])

    mean = interped.mean(axis=0)
    std  = interped.std(axis=0)

    ax.plot(common, mean, color=color, linestyle=ls, linewidth=2, label=method_name)
    ax.fill_between(common, mean - std, mean + std, color=color, alpha=0.12)

style_ax(ax, "", "训练步数", "回合奖励")
ax.legend(fontsize=12, framealpha=0.9)
plt.tight_layout()
out = os.path.join(SAVE_PATH, "training_reward_curves.svg")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── 图2：EV曲线 ───────────────────────────────────────────────────────────────

print("Plotting EV curves...")
fig, ax = plt.subplots(figsize=(10, 4))

for method_name, d in all_data.items():
    if not d["ev_steps"]:
        continue
    color, ls = d["color"], d["ls"]

    max_step = max(s[-1] for s in d["ev_steps"])
    common   = np.linspace(0, max_step, 500)
    interped = np.array([interpolate_to_common(s, v, common)
                         for s, v in zip(d["ev_steps"], d["ev_values"])])

    mean = interped.mean(axis=0)
    std  = interped.std(axis=0)

    ax.plot(common, mean, color=color, linestyle=ls, linewidth=2, label=method_name)
    ax.fill_between(common, mean - std, mean + std, color=color, alpha=0.12)

ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--")
ax.axhline(0.0, color="#cccccc", linewidth=0.8, linestyle="--")
ax.set_ylim(-0.1, 1.05)
style_ax(ax, "",
         "训练步数", "解释方差（EV）")
ax.legend(fontsize=12, framealpha=0.9, loc="lower right")
plt.tight_layout()
out = os.path.join(SAVE_PATH, "ev_curves.svg")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── 收敛步数汇总 ──────────────────────────────────────────────────────────────

print(f"\n收敛步数汇总（首次达到奖励 >= {CONVERGENCE_THRESHOLD}）：")
print(f"{'方法':<18} {'seed0':>10} {'seed1':>10} {'seed2':>10} {'均值':>10} {'标准差':>8}")
print("-" * 70)

for method_name, steps in convergence_table.items():
    if not steps:
        print(f"{method_name:<18} {'N/A':>10}")
        continue
    seed_strs = []
    for i in range(3):
        seed_strs.append(f"{steps[i]:>10}" if i < len(steps) else f"{'N/A':>10}")
    mean_c = np.mean(steps)
    std_c  = np.std(steps)
    print(f"{method_name:<18} {''.join(seed_strs)} {mean_c:>10.0f} {std_c:>8.0f}")

print("\nDone. Figures saved to:", SAVE_PATH)