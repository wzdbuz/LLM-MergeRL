import numpy as np
from stable_baselines3 import PPO
from config.config import ENV_CONFIG, TRAIN_CONFIG
from env.highway_wrapper import make_env


def evaluate(mode: str = "baseline", model_path: str = None, n_episodes: int = 50):
    """
    加载模型，跑 n_episodes 个 episode，输出：
    - 平均奖励
    - 碰撞率（论文核心指标）
    - 平均存活步数
    """
    if model_path is None:
        model_path = f"results/checkpoints/{mode}/best_model"

    model = PPO.load(model_path)
    env = make_env(ENV_CONFIG, seed=999, mode=mode)  # 加上 mode 参数

    rewards, lengths, crashes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward, ep_len, crashed = 0, 0, False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if info.get("crashed", False):
                crashed = True

        rewards.append(ep_reward)
        lengths.append(ep_len)
        crashes.append(int(crashed))

    results = {
        "mean_reward": np.mean(rewards),
        "std_reward":  np.std(rewards),
        "mean_length": np.mean(lengths),
        "crash_rate":  float(np.mean(crashes)),
    }

    print(f"\n===== {mode} 评估结果 =====")
    for k, v in results.items():
        print(f"  {k:<16}: {v:.4f}")

    return results