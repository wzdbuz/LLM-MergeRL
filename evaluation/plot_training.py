import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tb_scalar(log_dir: str, tag: str):
    """从 tfevents 文件里读取指定指标"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_baseline(log_dir: str = "results/logs/baseline", save_dir: str = "results/figures"):
    os.makedirs(save_dir, exist_ok=True)

    # 找到实际的 tfevents 子目录（SB3 会建一个 PPO_x 子文件夹）
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir)
               if os.path.isdir(os.path.join(log_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"在 {log_dir} 下找不到子目录，请确认路径")
    log_path = max(subdirs, key=os.path.getmtime)  # 取最新的一次训练
    print(f"读取日志：{log_path}")

    ea = EventAccumulator(log_path)
    ea.Reload()
    print("可用指标：", ea.Tags()["scalars"])  # 先打印所有可用 tag，方便确认

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PPO Baseline Training Curve — merge-v0", fontsize=14, fontweight="bold")

    def plot_one(ax, tag, title, ylabel, color, smooth=True):
        try:
            steps, values = load_tb_scalar(log_path, tag)
        except KeyError:
            ax.set_title(f"{title}\n(数据缺失: {tag})")
            return

        ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.8, label="原始")

        # 简单滑动平均平滑曲线
        if smooth and len(values) > 10:
            k = max(1, len(values) // 20)
            smoothed = [sum(values[max(0,i-k):i+1]) / len(values[max(0,i-k):i+1])
                        for i in range(len(values))]
            ax.plot(steps, smoothed, color=color, linewidth=2, label="平滑")
            ax.legend(fontsize=9)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timesteps", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k"))
        ax.grid(True, alpha=0.3)

    plot_one(axes[0][0],
             tag="rollout/ep_rew_mean",
             title="平均奖励",
             ylabel="Episode Reward",
             color="#378ADD")

    plot_one(axes[0][1],
             tag="rollout/ep_len_mean",
             title="平均步长",
             ylabel="Episode Length",
             color="#1D9E75")

    plot_one(axes[1][0],
             tag="train/value_loss",
             title="Value Loss",
             ylabel="Loss",
             color="#D85A30")

    plot_one(axes[1][1],
             tag="train/explained_variance",
             title="Explained Variance（Critic质量）",
             ylabel="Explained Variance",
             color="#7F77DD")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "baseline_training_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图已保存：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_baseline()