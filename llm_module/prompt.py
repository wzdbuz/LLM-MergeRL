# 构造发给LLM的prompt
from typing import List
import numpy as np
from state_encoder.encoder import encode_obs

def build_merge_prompt(obs: np.ndarray) -> str:
    """
    将观测矩阵转成发给 LLM 的 Prompt。
    obs shape: (5, 5)
    """
    scene = encode_obs(obs)  # 调用 encoder，不再重复写解析逻辑

    ego_speed = scene["ego"]["speed"]

    vehicle_desc = ""
    for v in scene["vehicles"]:
        vehicle_desc += (
            f"  - 车辆{v['id']}：{v['position']}{v['lane']}，"
            f"相对距离x={v['rel_x']:.2f}，速度={v['speed']:.2f}\n"
        )

    if not vehicle_desc:
        vehicle_desc = "  - 周围无可见车辆\n"

    prompt = f"""你是一个自动驾驶决策助手，当前场景是高速公路匝道汇入。

当前状态：
- 自车速度：{ego_speed:.2f}（归一化值，1.0为最高速）
- 周围车辆：
{vehicle_desc}
请分析当前驾驶风险，输出以下JSON格式，所有值在0到1之间：

{{
  "risk_level": <0到1，0=完全安全，1=极度危险>,
  "merge_urgency": <0到1，0=可以继续等待，1=必须立刻汇入>,
  "gap_adequacy": <0到1，0=主路间距不足，1=间距非常充裕>,
  "speed_advice": <0到1，0=应该减速，1=应该加速>,
  "reasoning": "<简短说明，不超过20字>"
}}

只输出JSON，不要其他内容。"""

    return prompt
