import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from config.config import ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG
from env.highway_wrapper import make_env


def train_baseline():
    print("=" * 50)
    print(f"训练 PPO Baseline — {ENV_CONFIG['id']}")  # 自动读取配置
    print("=" * 50)

    os.makedirs(TRAIN_CONFIG["save_path"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["log_path"], exist_ok=True)

    n_envs = TRAIN_CONFIG["n_envs"]

    # 训练环境：12 个并行进程
    train_env = SubprocVecEnv([
        (lambda i: lambda: Monitor(make_env(ENV_CONFIG, seed=TRAIN_CONFIG["seed"] + i)))(i)
        for i in range(n_envs)
    ])

    # 评估环境：单个就够
    eval_env = SubprocVecEnv([
    lambda: Monitor(make_env(ENV_CONFIG, seed=TRAIN_CONFIG["seed"] + 99))
    ])

    # 创建 PPO 模型，MlpPolicy 会根据 obs_space 自动构建网络结构
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=TRAIN_CONFIG["seed"],
        tensorboard_log=TRAIN_CONFIG["log_path"],
        **PPO_CONFIG,
    )

    # EvalCallback：每隔 eval_freq 步自动评估，并保存最优模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=TRAIN_CONFIG["save_path"],
        log_path=TRAIN_CONFIG["log_path"],
        eval_freq=TRAIN_CONFIG["eval_freq"],
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        deterministic=True,   # 评估时用确定性策略（不采样，直接取 argmax）
        render=False,
    )

    # 开始训练
    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )

    # 保存最终模型（最优模型已经由 EvalCallback 自动保存了）
    model.save(f"{TRAIN_CONFIG['save_path']}/final_model")
    print(f"\n训练完成，模型保存在 {TRAIN_CONFIG['save_path']}")
    return model