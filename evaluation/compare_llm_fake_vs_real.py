import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import evaluate


if __name__ == "__main__":
    print("===== FakeLLM eval =====")
    fake_state = evaluate(
        mode="llm_state",
        model_path="results/checkpoints/ppo_llm_state_seed0/best_model",
        n_seeds=6,
        n_episodes_per_seed=1,
        use_fake_llm=True,
    )
    fake_reward = evaluate(
        mode="llm_reward",
        model_path="results/checkpoints/ppo_llm_reward_seed0/best_model",
        n_seeds=6,
        n_episodes_per_seed=1,
        use_fake_llm=True,
    )

    print("\n===== Real LLM eval =====")
    real_state = evaluate(
        mode="llm_state",
        model_path="results/checkpoints/ppo_llm_state_seed0/best_model",
        n_seeds=6,
        n_episodes_per_seed=1,
        use_fake_llm=False,
    )
    real_reward = evaluate(
        mode="llm_reward",
        model_path="results/checkpoints/ppo_llm_reward_seed0/best_model",
        n_seeds=6,
        n_episodes_per_seed=1,
        use_fake_llm=False,
    )

    print("\n===== FakeLLM vs RealLLM =====")
    print(f"{'':20} {'FakeLLM':>12} {'RealLLM':>12}")
    print("-" * 44)
    for key in ["mean_reward", "crash_rate", "mean_speed"]:
        print(f"LLM-state  {key:<16}{fake_state.get(key, 0.0):>12.4f}{real_state.get(key, 0.0):>12.4f}")
        print(f"LLM-reward {key:<16}{fake_reward.get(key, 0.0):>12.4f}{real_reward.get(key, 0.0):>12.4f}")