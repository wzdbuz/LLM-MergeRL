import sys
import os
import numpy as np
sys.path.insert(0, ".")

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from config.config import get_config

# ── 配置 ──────────────────────────────────────────────────────────────────────
SEED       = 100
MODE       = "baseline"
MODEL_PATH = "D:/毕业设计/LLM-MergeRL/results/before_result/checkpoints/ppo_baseline_seed0/best_model.zip"

# ── 确认模型文件信息 ──────────────────────────────────────────────────────────
import time
if os.path.exists(MODEL_PATH):
    mtime = os.path.getmtime(MODEL_PATH)
    print(f"模型文件: {MODEL_PATH}")
    print(f"修改时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
    print(f"文件大小: {os.path.getsize(MODEL_PATH)} bytes")
else:
    print(f"警告：模型文件不存在: {MODEL_PATH}")

# ── 环境创建（开启渲染）──────────────────────────────────────────────────────
env_config, _, _ = get_config(MODE)
print(f"\n环境配置: duration={env_config['config'].get('duration')}, "
      f"high_speed_reward={env_config['config'].get('high_speed_reward')}")

env = gym.make(env_config["id"], render_mode="human")
env.unwrapped.configure(env_config["config"])
obs, _ = env.reset(seed=SEED)

model = PPO.load(MODEL_PATH)
print(f"模型设备: {model.device}")
obs, _ = env.reset(seed=SEED)

terminated = truncated = False
step = 0
speed_sum = 0.0

print(f"\n=== 开始仿真 seed={SEED} ===")
print(f"{'步数':>4}  {'速度(m/s)':>10}  {'动作':>6}  {'奖励':>8}")
print("-" * 40)

ACTION_NAMES = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    try:
        vehicle = env.unwrapped.vehicle
        if hasattr(vehicle, 'speed'):
            current_speed = float(vehicle.speed)
        elif hasattr(vehicle, 'velocity'):
            vel = vehicle.velocity
            current_speed = float(np.sqrt(vel.get('vx', 0)**2 + vel.get('vy', 0)**2))
        else:
            current_speed = 0.0
        speed_sum += current_speed
    except Exception as e:
        current_speed = 0.0

    action_name = ACTION_NAMES.get(int(action), str(action))
    print(f"{step:>4}  {current_speed:>10.4f}  {action_name:>6}  {reward:>8.4f}  pos={env.unwrapped.vehicle.position}")

print("-" * 40)
print(f"总步数: {step}")
print(f"平均速度: {speed_sum/step:.4f} m/s")
print(f"碰撞: {info.get('crashed', False)}")
print(f"terminated: {terminated}  （碰撞或驶出边界）")
print(f"truncated:  {truncated}   （超时）")
print(f"{step:>4}  {current_speed:>10.4f}  {action_name:>6}  {reward:>8.4f}  pos={env.unwrapped.vehicle.position}")
print(f"完整info: {info}")

env.close()