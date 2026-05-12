from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def make_ppo_model(env, ppo_config: dict, log_path: str, seed: int) -> PPO:
    # SB3 支持通过 device 指定训练设备（"cuda"/"cpu"/"auto"）；建议在 yaml 里配置 device: "cuda"
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        # 不传 tensorboard_log，避免创建 TensorBoard writer
        **ppo_config,
    )
    return model

