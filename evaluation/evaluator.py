import numpy as np
from stable_baselines3 import PPO

from config.config import get_config
from env.highway_wrapper import make_env
from evaluation.metrics import compute_metrics


def _default_eval_seeds() -> list[int]:
    # 建议：固定一个 seed 序列，单次评估时从中取不重复的子集，
    # 避免“同一个 seed 重复跑很多次”导致评估方差被低估。
    return [42, 137, 256, 891, 1024, 2048, 3141, 5678, 7777, 9999, 12345, 65536]


def evaluate(
    mode: str = "baseline",
    model_path: str | None = None,
    n_episodes_per_seed: int = 1,
    n_seeds: int = 6,
    seeds: list[int] | None = None,
    return_raw: bool = False,
    use_fake_llm: bool = True,
):
    """
    加载模型并评估。

    与“固定一个 seed 跑 50 次”的做法相比，这里默认做 3 次独立评估，
    每次使用不同 seed（但 seed 序列本身是固定的，保证可复现）。
    """
    env_config, _, _ = get_config(mode)

    if model_path is None:
        model_path = f"results/checkpoints/{mode}/best_model"

    model = PPO.load(model_path)

    seed_list = seeds if seeds is not None else _default_eval_seeds()[:n_seeds]
    if len(seed_list) != n_seeds:
        raise ValueError(f"n_seeds={n_seeds} 但 seeds 仅有 {len(seed_list)} 个: {seed_list}")

    rewards: list[float] = []
    lengths: list[int] = []
    crashes: list[int] = []
    mean_speeds: list[float] = []

    for seed in seed_list:
        env = make_env(env_config, seed=seed, mode=mode, use_fake_llm=use_fake_llm)

        for _ in range(n_episodes_per_seed):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_reward = 0.0
            ep_len = 0
            crashed = False
            speed_sum = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                if info.get("crashed", False):
                    crashed = True

                # 速度指标：用 highway-env 内部 ego 车速（更直观）
                try:
                    speed_sum += float(env.unwrapped.vehicle.speed)
                except Exception:
                    pass

            rewards.append(ep_reward)
            lengths.append(ep_len)
            crashes.append(int(crashed))
            mean_speeds.append(speed_sum / max(ep_len, 1))

    if return_raw:
        return rewards

    results = compute_metrics(rewards=rewards, lengths=lengths, crashes=crashes, mean_speeds=mean_speeds)

    print(f"\n===== {mode} ({('RealLLM' if not use_fake_llm else 'FakeLLM')}) 评估结果 =====")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<16}: {v:.4f}")
        else:
            print(f"  {k:<16}: {v}")

    return results
