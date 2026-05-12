import os
from datetime import datetime
from utils.seed_manager import set_seed
from rl_agent.PPO import make_ppo_model
from training.callback import MultiSeedEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure  # 新增

from config.config import get_config
from env.highway_wrapper import make_env


def train(mode: str = "baseline", experiment_name: str = "baseline", train_seed: int | None = None):
    ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG = get_config(mode)
    experiment_name = experiment_name or mode
    if train_seed is not None:
        TRAIN_CONFIG["seed"] = int(train_seed)
    set_seed(TRAIN_CONFIG["seed"])
    # 强制：训练阶段一律使用 FakeLLM（避免训练过程中产生海量 DeepSeek 调用费用）
    use_fake_llm = True

    print("=" * 50)
    print(f"训练 PPO — {ENV_CONFIG['id']} — 模式: {mode}")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join("results", "logs", experiment_name, timestamp)
    save_path = f"results/checkpoints/{experiment_name}"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path,  exist_ok=True)

    train_env = SubprocVecEnv([
        (lambda i: lambda: Monitor(
            make_env(ENV_CONFIG, seed=TRAIN_CONFIG["seed"] + i, mode=mode, use_fake_llm=use_fake_llm)
        ))(i)
        for i in range(TRAIN_CONFIG["n_envs"])
    ])

    model = make_ppo_model(
        env=train_env,
        ppo_config=PPO_CONFIG,
        log_path=log_path,
        seed=TRAIN_CONFIG["seed"],
    )

    # 禁用 TensorBoard，只用 stdout 和 csv，彻底解决权限问题
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)

    eval_callback = MultiSeedEvalCallback(
        env_config=ENV_CONFIG,
        mode=mode,
        use_fake_llm=use_fake_llm,
        eval_freq=TRAIN_CONFIG["eval_freq"],
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        n_eval_seeds=TRAIN_CONFIG.get("n_eval_seeds", 3),
        eval_seed_pool=TRAIN_CONFIG.get("eval_seed_pool", list(range(1, 13))),
        best_model_save_path=save_path,
        log_path=log_path,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )

    train_env.close()
    print("训练环境已关闭")

    model.save(f"{save_path}/final_model")
    print(f"\n训练完成，模型保存在 {save_path}")
    return model

