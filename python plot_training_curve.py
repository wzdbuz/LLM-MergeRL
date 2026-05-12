"""
PPO Baseline Seed0 训练曲线可视化
使用方法：python plot_training_curve.py
输出：training_curve_seed0.png（保存在同目录下）
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 1. 读取数据 ────────────────────────────────────────────────
PROGRESS_CSV = "results/logs/ppo_baseline_seed2/20260508_122906/progress.csv"

train_steps, train_rewards = [], []
ev_steps, ev_values = [], []
eval_steps, eval_rewards = [], []

# eval行没有timesteps，用前一个训练行的timesteps补充
last_timestep = None

with open(PROGRESS_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 训练行（有total_timesteps和ep_rew_mean）
        if row["time/total_timesteps"] and row["rollout/ep_rew_mean"]:
            last_timestep = int(row["time/total_timesteps"])
            train_steps.append(last_timestep)
            train_rewards.append(float(row["rollout/ep_rew_mean"]))

        # EV行（有explained_variance）
        if row["train/explained_variance"] and row["time/total_timesteps"]:
            ev_steps.append(int(row["time/total_timesteps"]))
            ev_values.append(float(row["train/explained_variance"]))

        # eval行（有eval/mean_reward但没有timesteps）
        if row["eval/mean_reward"] and not row["time/total_timesteps"] and last_timestep:
            eval_steps.append(last_timestep)
            eval_rewards.append(float(row["eval/mean_reward"]))

# ── 2. 平滑函数（指数移动平均）────────────────────────────────
def smooth(values, weight=0.85):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed

# ── 3. 画图 ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.patch.set_facecolor("#0f1117")

colors = {
    "raw":    "#3a6ea5",
    "smooth": "#5bc4f5",
    "eval":   "#f5a623",
    "ev":     "#7ed321",
    "grid":   "#2a2d3a",
    "text":   "#c8ccd4",
}

def style_ax(ax, title):
    ax.set_facecolor("#161b27")
    ax.set_title(title, color=colors["text"], fontsize=13, pad=10, fontweight="bold")
    ax.tick_params(colors=colors["text"], labelsize=9)
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors["grid"])
    ax.grid(True, color=colors["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}k"))

# ── 子图1：训练奖励曲线 ─────────────────────────────────────────
ax1 = axes[0]
style_ax(ax1, "Training Reward — PPO Baseline (seed=0)")

train_steps_arr = np.array(train_steps)
train_rewards_arr = np.array(train_rewards)
smoothed = smooth(train_rewards_arr, weight=0.9)

ax1.plot(train_steps_arr, train_rewards_arr,
         color=colors["raw"], linewidth=0.8, alpha=0.35, label="Raw reward")
ax1.plot(train_steps_arr, smoothed,
         color=colors["smooth"], linewidth=2.0, label="Smoothed (EMA=0.9)")

if eval_steps:
    ax1.scatter(eval_steps, eval_rewards,
                color=colors["eval"], s=20, zorder=5,
                label="Eval reward", alpha=0.8)

ax1.set_xlabel("Training Timesteps")
ax1.set_ylabel("Episode Reward")
ax1.legend(facecolor="#1e2230", edgecolor=colors["grid"],
           labelcolor=colors["text"], fontsize=9)

# 标注最终收敛值
final_reward = float(np.mean(train_rewards_arr[-20:]))
ax1.axhline(final_reward, color=colors["smooth"], linewidth=0.8,
            linestyle=":", alpha=0.5)
ax1.text(train_steps_arr[-1], final_reward + 0.2,
         f"Final avg: {final_reward:.2f}",
         color=colors["smooth"], fontsize=8, ha="right")

# ── 子图2：Explained Variance 曲线 ─────────────────────────────
ax2 = axes[1]
style_ax(ax2, "Explained Variance (Critic Quality) — PPO Baseline (seed=0)")

ev_steps_arr = np.array(ev_steps)
ev_arr = np.array(ev_values)

ax2.plot(ev_steps_arr, ev_arr,
         color=colors["ev"], linewidth=1.8, label="Explained Variance")
ax2.axhline(1.0, color=colors["grid"], linewidth=0.8, linestyle="--", alpha=0.5)
ax2.axhline(0.0, color=colors["grid"], linewidth=0.8, linestyle="--", alpha=0.5)
ax2.fill_between(ev_steps_arr, 0, ev_arr,
                 where=(ev_arr > 0), color=colors["ev"], alpha=0.08)

ax2.set_xlabel("Training Timesteps")
ax2.set_ylabel("Explained Variance")
ax2.set_ylim(-0.1, 1.05)
ax2.legend(facecolor="#1e2230", edgecolor=colors["grid"],
           labelcolor=colors["text"], fontsize=9)

# ── 4. 整体布局 ────────────────────────────────────────────────
plt.suptitle("PPO Baseline — seed=0 Training Summary",
             color=colors["text"], fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2.0)

out_path = "results/figures/training_curve_seed0.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"图已保存：{out_path}")
plt.show()