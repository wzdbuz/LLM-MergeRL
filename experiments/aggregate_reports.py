import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_single_row(csv_path: str) -> dict:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) != 1:
            raise ValueError(f"Expected 1 row in {csv_path}, got {len(rows)}")
        return rows[0]


if __name__ == "__main__":
    # 聚合各方法的报告（每个方法脚本会生成 results/analysis/<method>_report.csv）
    #methods = ["ppo_baseline", "ppo_llm_state", "ppo_llm_reward", "dqn_baseline"]
    methods = ["ppo_baseline", "ppo_llm_state", "ppo_llm_reward"]
    rows: list[dict] = []

    for m in methods:
        p = os.path.join("results", "analysis", f"{m}_report.csv")
        if not os.path.exists(p):
            print(f"[skip] missing: {p}")
            continue
        rows.append(_read_single_row(p))

    out_csv = os.path.join("results", "analysis", "final_report.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if not rows:
            raise FileNotFoundError("No per-method reports found under results/analysis/")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")
