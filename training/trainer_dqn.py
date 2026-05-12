from stable_baselines3.common.monitor import Monitor
from rl_agent.DQN import make_dqn_model
from training.callback import MultiSeedEvalCallback
from env.highway_wrapper import make_env
from config.config import get_config
from utils.seed_manager import set_seed
from datetime import datetime
import os

def train_dqn(experiment_name, train_seed):
    ENV_CONFIG, DQN_CONFIG, TRAIN_CONFIG = get_config("baseline_dqn")
    TRAIN_CONFIG["seed"] = int(train_seed)
    set_seed(train_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join("results", "logs", experiment_name, timestamp)
    save_path = os.path.join("results", "checkpoints", experiment_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # DQN只用单环境
    train_env = Monitor(make_env(ENV_CONFIG, seed=train_seed, mode="baseline"))

    model = make_dqn_model(
        env=train_env,
        dqn_config=DQN_CONFIG,
        log_path=log_path,
        seed=train_seed,
    )

    eval_callback = MultiSeedEvalCallback(
        env_config=ENV_CONFIG,
        mode="baseline",
        use_fake_llm=True,
        eval_freq=TRAIN_CONFIG["eval_freq"],
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        n_eval_seeds=TRAIN_CONFIG["n_eval_seeds"],
        eval_seed_pool=TRAIN_CONFIG["eval_seed_pool"],
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
    model.save(os.path.join(save_path, "final_model"))
    print(f"训练完成，模型保存在 {save_path}")