import csv
import os
 
for seed in [0, 1, 2]:
    path = f'results/logs/ppo_llm_state_seed{seed}'
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    latest = max(subdirs)
    csv_path = os.path.join(path, latest, 'eval_results.csv')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"seed{seed}: {len(rows)}行, 目录={latest}")
    print(f"  最后一行: reward={rows[-1]['reward']}, speed={rows[-1]['mean_speed']}, collision={rows[-1]['collision']}")
    print(f"  第一行:   reward={rows[0]['reward']}, speed={rows[0]['mean_speed']}")
    print()