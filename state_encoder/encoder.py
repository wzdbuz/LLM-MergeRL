import numpy as np


def encode_obs(obs: np.ndarray) -> dict:
    """
    将原始观测矩阵解析为结构化字典，供 prompt.py 调用。
    obs shape: (5, 5)，每行为 [presence, x, y, vx, vy]
    """
    ego = obs[0]
    others = obs[1:]

    ego_info = {
        "speed": float(ego[3]),   # vx：纵向速度
    }

    vehicles = []
    for i, v in enumerate(others):
        if v[0] > 0.5:            # presence > 0.5 才算有效车辆
            rel_x = float(v[1])
            rel_y = float(v[2])
            vehicles.append({
                "id":       i + 1,
                "position": "前方" if rel_x > 0 else "后方",
                "lane":     "同道" if abs(rel_y) <= 0.3 else "主路",
                "rel_x":    rel_x,
                "speed":    float(v[3]),
            })

    return {
        "ego":     ego_info,
        "vehicles": vehicles,
    }