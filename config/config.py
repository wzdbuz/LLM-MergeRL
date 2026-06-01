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
