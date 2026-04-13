# 构造发给LLM的prompt
from typing import List
import numpy as np


def build_merge_prompt(obs: np.ndarray) -> str:
    """
    将 highway-env 的 Kinematics 观测转成自然语言描述
    obs shape: (vehicles_count, 5)，每行是 [presence, x, y, vx, vy]
    """
    ego = obs[0]   # 第0行是自车
    others = obs[1:]

    # 自车状态
    ego_speed = float(ego[3])  # vx

    # 找到存在的其他车辆
    visible = [(i, v) for i, v in enumerate(others) if v[0] > 0.5]

    vehicle_desc = ""
    for i, v in visible:
        rel_x = float(v[1])   # 相对x（正=前方）
        rel_y = float(v[2])   # 相对y（正=右侧/主路方向）
        speed = float(v[3])
        position = "前方" if rel_x > 0 else "后方"
        lane = "主路" if abs(rel_y) > 0.3 else "同道"
        vehicle_desc += f"  - 车辆{i+1}：{position}{lane}，"
        vehicle_desc += f"相对距离x={rel_x:.2f}，速度={speed:.2f}\n"

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