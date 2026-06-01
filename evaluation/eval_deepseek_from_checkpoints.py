import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import evaluate


if __name__ == "__main__":
    # 用 DeepSeek API 做“正式评估”（不会影响训练）
    for mode in ["llm_state", "llm_reward"]:
        print(f"\n===== DeepSeek Eval: {mode} =====")
        evaluate(
            mode=mode,
            model_path=f"results/checkpoints/{mode}/best_model",
            use_fake_llm=False,  # 关键：强制走 DeepSeek
        )

