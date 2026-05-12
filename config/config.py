import os
import yaml


def load_config(yaml_path: str) -> dict:
    """读取 yaml 配置文件"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config(mode: str = "baseline") -> tuple:
    """
    根据实验模式加载对应的 yaml 配置，
    返回 (ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG)
    """
    yaml_map = {
        "baseline":   "config/experiment_configs/baseline.yaml",
        "llm_state":  "config/experiment_configs/llm_state.yaml",
        "llm_reward": "config/experiment_configs/llm_reward.yaml",
        "baseline_dqn": "config/experiment_configs/baseline_dqn.yaml",
    }

    yaml_path = yaml_map.get(mode)
    if yaml_path is None:
        raise ValueError(f"未知的实验模式: {mode}，可选: {list(yaml_map.keys())}")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    cfg = load_config(yaml_path)

    ENV_CONFIG   = cfg["env"]
    # SAC 用 sac 字段，PPO 用 ppo 字段
    PPO_CONFIG = cfg.get("ppo") or cfg.get("sac") or {}
    TRAIN_CONFIG = cfg["train"]

    # 自动补充保存路径（根据 mode 生成，不写死在 yaml 里）
    TRAIN_CONFIG["save_path"] = f"results/checkpoints/{mode}"
    TRAIN_CONFIG["log_path"]  = f"results/logs/{mode}"

    return ENV_CONFIG, PPO_CONFIG, TRAIN_CONFIG

"""# 环境配置
ENV_CONFIG = {
    "id": "merge-v0",
    "config": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 30,
        "collision_reward": -5,
        "reward_speed_range": [20, 30],
        "high_speed_reward": 0.4,
        "merging_speed_reward": 0.5,
        "lane_change_reward": 0,
    }
}

# PPO 超参（SB3 默认值已经很好，先不动）
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 512,   # 每次更新前收集多少步数据
    "batch_size": 512,
    "n_epochs": 10,       # 每批数据用几遍
    "gamma": 0.99,        # 折扣因子
    "gae_lambda": 0.95,
    "clip_range": 0.2,    # PPO 核心超参，限制策略更新幅度
    "ent_coef": 0.01,     # 熵正则，鼓励探索
    "verbose": 1,
}

# 训练流程配置
TRAIN_CONFIG = {
    "total_timesteps": 1_000_000,
    "seed": 42,
    "n_eval_episodes": 20,       # 每次评估跑 20 个 episode
    "eval_freq": 50_000,
    "save_path": "results/checkpoints/baseline",
    "log_path": "results/logs/baseline",
    "n_envs": 12,
}

# 实验模式（后续 LLM 实验会用到）
class Config:
    mode = "baseline"
    use_llm_state = False
    use_llm_reward = False
    lambda_risk = 0.5"""