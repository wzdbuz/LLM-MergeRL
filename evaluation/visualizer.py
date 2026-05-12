import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False


def _find_latest_evaluations_npz(log_dir: str) -> str | None:
    if not os.path.exists(log_dir):
        return None
    # 优先找最近一次训练的子目录
    for sub in sorted(os.listdir(log_dir), reverse=True):
        candidate = os.path.join(log_dir, sub, "evaluations.npz")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(log_dir, "evaluations.npz")
    return candidate if os.path.exists(candidate) else None


def plot_learning_curves(log_dirs: dict, save_path: str = "results/figures"):
    os.makedirs(save_path, exist_ok=True)
    colors = {"baseline": "#4A7AB5", "llm_state": "#E8833A", "llm_reward": "#2E8B57"}
    labels = {"baseline": "Baseline", "llm_state": "LLM-state", "llm_reward": "LLM-reward"}

    plt.figure(figsize=(10, 5))
    found_any = False

    for name, log_dir in log_dirs.items():
        npz_path = _find_latest_evaluations_npz(log_dir)
        if npz_path is None:
            print(f"[plot] 找不到 {name} 的 evaluations.npz，跳过：{log_dir}")
            continue

        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]

        means = results.mean(axis=1)
        stds = results.std(axis=1)
        plt.plot(timesteps, means, color=colors.get(name, None), label=labels.get(name, name), linewidth=2)
        plt.fill_between(timesteps, means - stds, means + stds, color=colors.get(name, None), alpha=0.15)
        found_any = True

    if not found_any:
        print("[plot] 未找到任何评估数据，无法生成学习曲线。")
        plt.close()
        return

    plt.xlabel("训练步数", fontsize=12)
    plt.ylabel("平均回合奖励", fontsize=12)
    plt.title("学习曲线对比", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(save_path, "learning_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] 学习曲线已保存：{out}")


def plot_crash_rate_curves(log_dirs: dict, save_path: str = "results/figures"):
    """
    绘制评估过程中的碰撞率曲线（需要 evaluations.npz 里包含 crash_rates）。
    """
    os.makedirs(save_path, exist_ok=True)
    colors = {"baseline": "#4A7AB5", "llm_state": "#E8833A", "llm_reward": "#2E8B57"}
    labels = {"baseline": "Baseline", "llm_state": "LLM-state", "llm_reward": "LLM-reward"}

    plt.figure(figsize=(10, 5))
    found_any = False

    for name, log_dir in log_dirs.items():
        npz_path = _find_latest_evaluations_npz(log_dir)
        if npz_path is None:
            continue
        data = np.load(npz_path)
        if "crash_rates" not in data:
            continue
        timesteps = data["timesteps"]
        crash_rates = data["crash_rates"]
        plt.plot(timesteps, crash_rates, color=colors.get(name, None), label=labels.get(name, name), linewidth=2)
        found_any = True

    if not found_any:
        print("[plot] 未找到 crash_rates，无法生成碰撞率曲线。")
        plt.close()
        return

    plt.xlabel("训练步数", fontsize=12)
    plt.ylabel("碰撞率", fontsize=12)
    plt.title("碰撞率曲线对比", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(save_path, "crash_rate_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] 碰撞率曲线已保存：{out}")


def plot_bar_comparison(results: dict, save_path: str = "results/figures"):
    os.makedirs(save_path, exist_ok=True)

    metrics = ["mean_reward", "crash_rate", "mean_length", "mean_speed"]
    titles = ["平均回合奖励", "碰撞率", "平均回合步长", "平均速度"]
    ylabels = ["reward", "rate", "steps", "speed"]

    methods = ["baseline", "llm_state", "llm_reward"]
    xlabels = ["Baseline", "LLM-state", "LLM-reward"]
    colors = ["#4A7AB5", "#E8833A", "#2E8B57"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        values = [float(results.get(m, {}).get(metric, 0.0)) for m in methods]
        bars = ax.bar(xlabels, values, color=colors, width=0.55, edgecolor="white")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) if max(values) != 0 else 1.0) * 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "crash_rate":
            ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)

    plt.suptitle("核心指标对比", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = os.path.join(save_path, "bar_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] 指标柱状图已保存：{out}")
