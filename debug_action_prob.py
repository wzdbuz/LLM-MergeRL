import sys
import os
import numpy as np
import torch
sys.path.insert(0, ".")

import highway_env  # noqa: F401
from stable_baselines3 import PPO
from config.config import get_config
from env.highway_wrapper import make_env  # ← 用make_env

SEED       = 137
MODE       = "llm_state"
MODEL_PATH = "results/checkpoints/ppo_llm_state_seed0/best_model.zip"

env_config, _, _ = get_config(MODE)

# ← 用make_env创建环境，会自动加LLMStateWrapper
env = make_env(env_config, seed=SEED, mode=MODE, use_fake_llm=True)

model = PPO.load(MODEL_PATH)
obs, _ = env.reset(seed=SEED)

terminated = truncated = False
step = 0

ACTION_NAMES = {0: "LEFT", 1: "IDLE", 2: "RIGHT", 3: "FASTER", 4: "SLOWER"}

print(f"\n=== 开始仿真 seed={SEED} ===")
print(f"{'步':>3}  {'速度':>7}  {'动作':>6}  {'LEFT':>6}  {'IDLE':>6}  {'RIGHT':>6}  {'FASTER':>7}  {'SLOWER':>7}  {'奖励':>8}")
print("-" * 80)

while not (terminated or truncated):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    speed = float(env.unwrapped.vehicle.speed)
    action_name = ACTION_NAMES.get(int(action), str(action))

    print(f"{step:>3}  {speed:>7.3f}  {action_name:>6}  "
          f"{probs[0]:>6.3f}  {probs[1]:>6.3f}  {probs[2]:>6.3f}  "
          f"{probs[3]:>7.3f}  {probs[4]:>7.3f}  {reward:>8.4f}")

print("-" * 80)
print(f"crashed={info.get('crashed')}, terminated={terminated}, truncated={truncated}")
env.close()