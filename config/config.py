# 环境配置
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
    lambda_risk = 0.5