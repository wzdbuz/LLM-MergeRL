import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from llm_module.fake_llm import FakeLLM
from llm_module.prompt import build_merge_prompt

from fusion import StateFusion, RewardShaping

# 构造一个假的观测：自车前方有一辆很近的车
obs = np.zeros((5, 5), dtype=np.float32)
obs[0] = [1, 0, 0, 0.7, 0]      # 自车，速度0.7
obs[1] = [1, 0.15, 0.0, 0.5, 0]  # 前方很近的车

obs_2d = obs  # shape (5, 5)

# 测试 FakeLLM
fake = FakeLLM()
prior = fake.get_prior(obs)
print("FakeLLM 输出：", prior)
print("状态向量：", prior.to_vector())
print("奖励加成：", prior.to_reward_bonus())

# 测试 prompt 生成
print("\n生成的 Prompt：")
print(build_merge_prompt(obs))

# 测试 LLM-state
sf = StateFusion(use_fake_llm=True)
new_obs, new_reward = sf.process(obs_2d, reward=3.0)
print(f"\nLLM-state 融合后观测维度: {new_obs.shape}")  # 期望 (29,)
print(f"新增的4维语义先验: {new_obs[-4:]}")

# 测试 LLM-reward
rs = RewardShaping(use_fake_llm=True, lambda_weight=0.5)
new_obs2, new_reward2 = rs.process(obs_2d, reward=3.0)
print(f"\nLLM-reward 原始奖励: 3.0")
print(f"LLM-reward 塑形后奖励: {new_reward2:.4f}")
print(f"变化量: {new_reward2 - 3.0:.4f}")