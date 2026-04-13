from abc import ABC, abstractmethod
import numpy as np
from llm_module.semantic_prior import SemanticPrior


class FusionBase(ABC):
    """两种融合方式的抽象基类，统一接口"""

    def __init__(self, use_fake_llm: bool = True):
        if use_fake_llm:
            from llm_module.fake_llm import FakeLLM
            self.llm = FakeLLM()
        else:
            from llm_module.llm_api import LLMClient
            self.llm = LLMClient()

    def get_prior(self, obs: np.ndarray) -> SemanticPrior:
        return self.llm.get_prior(obs)

    @abstractmethod
    def process(self, obs: np.ndarray, reward: float) -> tuple:
        """
        输入原始观测和奖励，输出处理后的观测和奖励
        返回: (new_obs, new_reward)
        """
        pass