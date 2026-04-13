# 调试用，不花钱
import numpy as np
from llm_module.semantic_prior import SemanticPrior


class FakeLLM:
    """
    调试用的伪 LLM，根据观测做简单规则计算，模拟 LLM 输出格式。
    训练时用这个，评估时换成真实 API。
    """

    def get_prior(self, obs: np.ndarray) -> SemanticPrior:
        ego = obs[0]
        others = obs[1:]

        ego_speed = float(ego[3])
        visible = [v for v in others if v[0] > 0.5]

        # 简单规则：前方有车且距离近 → 危险
        risk = 0.1
        gap = 0.8
        for v in visible:
            rel_x = float(v[1])
            rel_y = float(v[2])
            if 0 < rel_x < 0.3 and abs(rel_y) < 0.2:
                risk = min(risk + 0.4, 1.0)
                gap = max(gap - 0.4, 0.0)

        return SemanticPrior(
            risk_level=round(risk, 2),
            merge_urgency=round(1.0 - gap, 2),
            gap_adequacy=round(gap, 2),
            speed_advice=round(1.0 - risk, 2),
        )