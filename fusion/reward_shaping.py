import numpy as np
from fusion.fusion_base import FusionBase


class RewardShaping(FusionBase):
    """
    LLM-reward：用语义先验对原始奖励做塑形
    观测不变，奖励 = 原始奖励 + λ × LLM奖励加成
    ablation: None=完整版，'risk'/'urgency'/'gap'/'speed'=去掉对应维度
    """

    def __init__(self, use_fake_llm: bool = True, lambda_weight: float = 0.5,
                 ablation: str = None):
        super().__init__(use_fake_llm)
        self.lambda_weight = lambda_weight
        self.ablation = ablation

    def process(self, obs: np.ndarray, reward: float) -> tuple:
        flat_obs = obs.flatten()
        n_vehicles = obs.size // 5
        prior = self.get_prior(obs.reshape(n_vehicles, 5))

        # 消融实验：将对应维度置0
        if self.ablation == "risk":
            prior.risk_level = 0.0
        elif self.ablation == "urgency":
            prior.merge_urgency = 0.0
        elif self.ablation == "gap":
            prior.gap_adequacy = 0.0
        elif self.ablation == "speed":
            prior.speed_advice = 0.0

        bonus = prior.to_reward_bonus()
        new_reward = reward + self.lambda_weight * bonus
        return flat_obs, new_reward