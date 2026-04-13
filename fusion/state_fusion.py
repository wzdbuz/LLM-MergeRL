import numpy as np
from fusion.fusion_base import FusionBase


class StateFusion(FusionBase):
    """
    LLM-state：将语义先验拼接到原始观测向量后面
    原始 obs: (25,) → 融合后 obs: (29,)
    多出来的 4 维就是 [risk_level, merge_urgency, gap_adequacy, speed_advice]
    """

    def process(self, obs: np.ndarray, reward: float) -> tuple:
        flat_obs = obs.flatten()                        # (5,5) → (25,)
        prior = self.get_prior(obs.reshape(5, 5))       # 获取语义先验
        prior_vec = prior.to_vector()                   # 转成 (4,) 向量
        new_obs = np.concatenate([flat_obs, prior_vec]) # 拼接 → (29,)
        return new_obs, reward                          # 奖励不变