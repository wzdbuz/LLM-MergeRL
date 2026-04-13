# 语义先验的数据结构
from dataclasses import dataclass
import numpy as np

@dataclass
class SemanticPrior:
    """
    LLM 输出的语义先验，所有字段都归一化到 [0, 1]
    供 LLM-state 和 LLM-reward 两种融合方式使用
    """
    risk_level: float        # 当前场景危险程度，0=安全，1=极危险
    merge_urgency: float     # 汇入紧迫程度，0=可以等，1=必须立刻汇入
    gap_adequacy: float      # 主路间距是否充足，0=间距不够，1=间距很充足
    speed_advice: float      # 速度建议，0=应该减速，1=应该加速

    def to_vector(self) -> np.ndarray:
        """转成 numpy 向量，直接拼接到状态里（LLM-state 用）"""
        return np.array([
            self.risk_level,
            self.merge_urgency,
            self.gap_adequacy,
            self.speed_advice,
        ], dtype=np.float32)

    def to_reward_bonus(self) -> float:
        """
        转成奖励加成（LLM-reward 用）
        安全、间距充足时给正奖励；危险时给负奖励
        """
        bonus = 0.0
        bonus += (1.0 - self.risk_level) * 0.3     # 越安全奖励越高
        bonus += self.gap_adequacy * 0.2            # 间距充足时鼓励汇入
        bonus -= self.risk_level * 0.5              # 危险时惩罚
        return float(np.clip(bonus, -1.0, 1.0))