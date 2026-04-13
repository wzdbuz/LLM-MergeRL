import numpy as np
from fusion.fusion_base import FusionBase


class RewardShaping(FusionBase):
    """
    LLM-reward：用语义先验对原始奖励做塑形
    观测不变，奖励 = 原始奖励 + λ × LLM奖励加成
    """

    def __init__(self, use_fake_llm: bool = True, lambda_weight: float = 0.5):
        super().__init__(use_fake_llm)
        self.lambda_weight = lambda_weight  # 控制 LLM 奖励的影响强度

    def process(self, obs: np.ndarray, reward: float) -> tuple:
        flat_obs = obs.flatten()
        prior = self.get_prior(obs.reshape(5, 5))
        bonus = prior.to_reward_bonus()                              # LLM 奖励加成
        new_reward = reward + self.lambda_weight * bonus             # 加权叠加
        return flat_obs, new_reward                                  # 观测不变