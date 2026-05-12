import os
import sys

# Ensure project root is on sys.path when running as a script:
#   python experiments/run_seeds_ppo_baseline.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import train
from experiments._seed_experiment_utils import (
    RunSummary,
    latest_subdir,
    summarize_last10_evals,
    aggregate_method,
    write_method_report,
)


if __name__ == "__main__":
    method = "ppo_baseline"
    train_seeds = [0, 1, 2]

    per_run: list[RunSummary] = []

    for seed in train_seeds:
        experiment_name = f"{method}_seed{seed}"

        # 断点续跑：如果该 seed 已经有 eval_results.csv，就跳过训练直接做汇总
        log_root = os.path.join("results", "logs", experiment_name)
        latest = latest_subdir(log_root)
        eval_csv_path = os.path.join(latest, "eval_results.csv") if latest else None

        if not (eval_csv_path and os.path.exists(eval_csv_path)):
            print(f"\n=== Train {method} seed={seed} ({experiment_name}) ===")
            train(mode="baseline", experiment_name=experiment_name, train_seed=seed)
            latest = latest_subdir(log_root)
            if latest is None:
                raise FileNotFoundError(f"No log dir under: {log_root}")
            eval_csv_path = os.path.join(latest, "eval_results.csv")
            if not os.path.exists(eval_csv_path):
                raise FileNotFoundError(f"Missing eval_results.csv: {eval_csv_path}")
        else:
            print(f"\n=== Skip training (found eval_results.csv) {method} seed={seed} ===")

        r_med, v_med, crash_mean, n_rows = summarize_last10_evals(eval_csv_path)
        per_run.append(
            RunSummary(
                method=method,
                train_seed=seed,
                reward_median_last10=r_med,
                speed_median_last10=v_med,
                crash_rate_mean_last10=crash_mean,
                used_eval_rows=n_rows,
                csv_path=eval_csv_path,
            )
        )
        print(
            f"[summary:{method} seed={seed}] reward_median={r_med:.3f} speed_median={v_med:.3f} "
            f"crash_rate_mean={crash_mean:.3f} rows={n_rows}"
        )

    report = aggregate_method(method, per_run)
    out_csv = os.path.join("results", "analysis", f"{method}_report.csv")
    write_method_report(out_csv, report, per_run)
    print(f"\nWrote report: {out_csv}")
