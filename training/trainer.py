import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from config.config import ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG
from env.highway_wrapper import make_env


def train(mode: str = "baseline", experiment_name: str = "baseline"):
    print("=" * 50)
    print(f"训练 PPO — {ENV_CONFIG['id']} — 模式: {mode}")
    print("=" * 50)

    os.makedirs(f"results/checkpoints/{experiment_name}", exist_ok=True)
    os.makedirs(f"results/logs/{experiment_name}", exist_ok=True)

    train_env = SubprocVecEnv([
        (lambda i: lambda: Monitor(
            make_env(ENV_CONFIG, seed=TRAIN_CONFIG["seed"] + i, mode=mode)
        ))(i)
        for i in range(TRAIN_CONFIG["n_envs"])
    ])

    eval_env = SubprocVecEnv([
        lambda: Monitor(make_env(ENV_CONFIG, seed=TRAIN_CONFIG["seed"] + 99, mode=mode))
    ])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=TRAIN_CONFIG["seed"],
        tensorboard_log=f"results/logs/{experiment_name}",
        **PPO_CONFIG,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"results/checkpoints/{experiment_name}",
        log_path=f"results/logs/{experiment_name}",
        eval_freq=TRAIN_CONFIG["eval_freq"],
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(f"results/checkpoints/{experiment_name}/final_model")
    print(f"\n✅ 训练完成，模型保存在 results/checkpoints/{experiment_name}")
    return model