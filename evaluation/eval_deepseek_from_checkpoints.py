import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import evaluate


if __name__ == "__main__":
    # 用 DeepSeek API 做“正式评估”（不会影响训练）
    # 默认评估种子池与每 seed episode 数量已在 evaluator.py 里设置为：12 seeds / 6 seeds used / 1 episode/seed
    for mode in ["llm_state", "llm_reward"]:
        print(f"\n===== DeepSeek Eval: {mode} =====")
        evaluate(
            mode=mode,
            model_path=f"results/checkpoints/{mode}/best_model",
            use_fake_llm=False,  # 关键：强制走 DeepSeek
        )

