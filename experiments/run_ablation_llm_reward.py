"""
消融实验：逐一去掉LLM-reward的4个语义先验维度，分析各维度贡献
在项目根目录下运行：python experiments/run_ablation_llm_reward.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import train
from experiments._seed_experiment_utils import (
    RunSummary,
    latest_subdir,
    summarize_last10_evals,
    aggregate_method,
    write_method_report,
)

# 消融实验配置：方法名 → ablation参数
# None表示完整版（已有结果可跳过），其余为去掉对应维度
ABLATION_CONFIGS = {
    "llm_reward_no_risk":     "risk",      # 去掉风险等级
    "llm_reward_no_gap":      "gap",       # 去掉间距充裕度
}

TRAIN_SEEDS = [0, 1, 2]


if __name__ == "__main__":
    all_reports = {}

    for method_name, ablation in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"消融实验：{method_name}（ablation={ablation}）")
        print(f"{'='*60}")

        per_run: list[RunSummary] = []

        for seed in TRAIN_SEEDS:
            experiment_name = f"{method_name}_seed{seed}"
            log_root = os.path.join("results", "logs", experiment_name)
            latest = latest_subdir(log_root)
            eval_csv_path = os.path.join(latest, "eval_results.csv") if latest else None

            if not (eval_csv_path and os.path.exists(eval_csv_path)):
                print(f"\n=== Train {method_name} seed={seed} ===")
                # 传入ablation参数到trainer
                train(
                    mode="llm_reward",
                    experiment_name=experiment_name,
                    train_seed=seed,
                    ablation=ablation,
                )
                latest = latest_subdir(log_root)
                if latest is None:
                    raise FileNotFoundError(f"No log dir under: {log_root}")
                eval_csv_path = os.path.join(latest, "eval_results.csv")
                if not os.path.exists(eval_csv_path):
                    raise FileNotFoundError(f"Missing eval_results.csv: {eval_csv_path}")
            else:
                print(f"\n=== Skip (found eval_results.csv) {method_name} seed={seed} ===")

            r_med, v_med, crash_mean, n_rows = summarize_last10_evals(eval_csv_path)
            per_run.append(
                RunSummary(
                    method=method_name,
                    train_seed=seed,
                    reward_median_last10=r_med,
                    speed_median_last10=v_med,
                    crash_rate_mean_last10=crash_mean,
                    used_eval_rows=n_rows,
                    csv_path=eval_csv_path,
                )
            )
            print(
                f"[summary:{method_name} seed={seed}] "
                f"reward_median={r_med:.3f} speed_median={v_med:.3f} "
                f"crash_rate={crash_mean:.3f} rows={n_rows}"
            )

        report = aggregate_method(method_name, per_run)
        all_reports[method_name] = report

        out_csv = os.path.join("results", "analysis", f"{method_name}_report.csv")
        write_method_report(out_csv, report, per_run)
        print(f"Wrote report: {out_csv}")

    # 汇总打印消融实验结果
    print(f"\n{'='*70}")
    print("消融实验汇总结果")
    print(f"{'='*70}")
    print(f"{'方法':<28} {'奖励均值':>10} {'奖励标准差':>10} {'速度均值':>10} {'碰撞率均值':>10}")
    print("-" * 70)
    for method_name, report in all_reports.items():
        print(
            f"{method_name:<28} "
            f"{report['reward_median_mean']:>10.3f} "
            f"{report['reward_median_std']:>10.3f} "
            f"{report['speed_median_mean']:>10.3f} "
            f"{report['crash_rate_mean_mean']:>10.3f}"
        )