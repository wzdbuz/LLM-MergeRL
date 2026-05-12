import os
import csv
from dataclasses import dataclass
from statistics import median
from typing import Optional

import numpy as np


@dataclass
class RunSummary:
    method: str
    train_seed: int
    reward_median_last10: float
    speed_median_last10: float
    crash_rate_mean_last10: float
    used_eval_rows: int
    csv_path: str


def latest_subdir(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def load_eval_csv(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def summarize_last10_evals(eval_csv_path: str) -> tuple[float, float, float, int]:
    rows = load_eval_csv(eval_csv_path)
    if not rows:
        raise ValueError(f"Empty eval csv: {eval_csv_path}")

    eval_indices = sorted({int(r["eval_index"]) for r in rows})
    last10 = set(eval_indices[-10:])
    filtered = [r for r in rows if int(r["eval_index"]) in last10]

    rewards = [float(r["reward"]) for r in filtered]
    speeds = [float(r["mean_speed"]) for r in filtered]
    crashes = [int(r["collision"]) for r in filtered]

    return float(median(rewards)), float(median(speeds)), float(np.mean(crashes)), len(filtered)


def aggregate_method(method: str, run_summaries: list[RunSummary]) -> dict:
    reward_vals = [s.reward_median_last10 for s in run_summaries]
    speed_vals = [s.speed_median_last10 for s in run_summaries]
    crash_vals = [s.crash_rate_mean_last10 for s in run_summaries]

    return {
        "method": method,
        "n_runs": len(run_summaries),
        "reward_median_mean": float(np.mean(reward_vals)),
        "reward_median_std": float(np.std(reward_vals)),
        "speed_median_mean": float(np.mean(speed_vals)),
        "speed_median_std": float(np.std(speed_vals)),
        "crash_rate_mean_mean": float(np.mean(crash_vals)),
        "crash_rate_mean_std": float(np.std(crash_vals)),
    }


def write_method_report(out_csv: str, method_report: dict, per_run: list[RunSummary]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 主报告（1 行）
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_runs",
                "reward_median_mean",
                "reward_median_std",
                "speed_median_mean",
                "speed_median_std",
                "crash_rate_mean_mean",
                "crash_rate_mean_std",
            ],
        )
        w.writeheader()
        w.writerow(method_report)

    # 每次 run 的明细
    detail_csv = out_csv.replace(".csv", "_runs.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "train_seed",
                "reward_median_last10",
                "speed_median_last10",
                "crash_rate_mean_last10",
                "used_eval_rows",
                "csv_path",
            ],
        )
        w.writeheader()
        for s in per_run:
            w.writerow(
                {
                    "method": s.method,
                    "train_seed": s.train_seed,
                    "reward_median_last10": s.reward_median_last10,
                    "speed_median_last10": s.speed_median_last10,
                    "crash_rate_mean_last10": s.crash_rate_mean_last10,
                    "used_eval_rows": s.used_eval_rows,
                    "csv_path": s.csv_path,
                }
            )

