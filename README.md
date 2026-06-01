# 基于大语言模型语义先验引导的强化学习自动驾驶决策系统（LLM-MergeRL）

## 项目简介

本项目提出一种融合**大语言模型（LLM）**与**近端策略优化（PPO）**的自动驾驶决策框架，在 highway-env 的匝道汇入场景（merge-v0）中，探究将 LLM 语义推理结果通过**状态融合**与**奖励塑形**两种方式引入 PPO 训练过程的可行性与实际效果。

传统强化学习方法在自动驾驶任务中通常存在以下问题：

* 奖励信号稀疏，学习效率低
* 缺乏对交通场景的高层语义理解能力
* 在复杂交互场景中决策保守，泛化能力较弱

针对上述问题，本项目利用大语言模型对当前场景进行语义分析，生成包含**风险等级、并线紧迫度、间距充裕度、速度建议**四个维度的语义先验向量，并通过以下两种融合方式引导策略优化：

* **LLM-state**：将语义先验拼接至观测向量，扩充状态表示
* **LLM-reward**：将语义先验转化为即时辅助奖励信号，进行奖励塑形

---

## 运行环境

### 硬件环境

| 硬件 | 配置 |
|------|------|
| 操作系统 | Windows 11 |
| 处理器 | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3050 |
| 内存 | 16GB DDR4 |

### 软件环境

| 软件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.5.1（CUDA 12.1）|
| Stable-Baselines3 | 2.7.1 |
| highway-env | 1.10.2 |
| gymnasium | 0.29.x |
| numpy | 1.26.x |
| matplotlib | 3.x |

---

## 安装步骤

```bash
git clone https://github.com/wzdbuz/LLM-MergeRL.git
cd LLM-MergeRL

pip install -r requirements.txt
```

### 配置 DeepSeek API Key

训练阶段使用 FakeLLM，无需 API Key。正式评估阶段需要配置 DeepSeek API Key：

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your_api_key_here"
```

### 修改 highway-env 源码

本项目对 highway-env 的 merge_env.py 做了以下修改，需手动同步：

```
文件路径：
{Python环境}/site-packages/highway_env/envs/merge_env.py
```

主要修改内容：
1. `_rewards` 函数：换道判断改为真正换道成功才给奖励
2. `_is_terminated` 函数：终点线从 x>370 改为 x>1000
3. `_make_road` 函数：道路末段从 150m 延长至 1200m

修改完成后删除缓存文件：
```bash
del {Python环境}/site-packages/highway_env/envs/merge_env.cpython-310.pyc
```

---

## 项目结构

```
LLM-MergeRL/
├── config/
│   └── experiment_configs/
│       ├── baseline.yaml        # PPO Baseline 配置
│       ├── llm_state.yaml       # LLM-state 配置
│       ├── llm_reward.yaml      # LLM-reward 配置
│       └── baseline_dqn.yaml    # DQN Baseline 配置
├── env/
│   └── highway_wrapper.py       # 环境包装器（LLMStateWrapper / LLMRewardWrapper）
├── experiments/
│   ├── run_seeds_ppo_baseline.py
│   ├── run_seeds_ppo_llm_state.py
│   ├── run_seeds_ppo_llm_reward.py
│   ├── run_seeds_dqn_baseline.py
│   └── run_ablation_llm_reward.py  # 消融实验
├── fusion/
│   ├── reward_shaping.py        # LLM-reward 奖励塑形
│   └── state_fusion.py          # LLM-state 状态融合
├── llm_module/
│   ├── fake_llm.py              # 规则先验代理（训练阶段使用）
│   ├── real_llm.py              # DeepSeek API（评估阶段使用）
│   ├── prompt_builder.py        # Prompt 模板
│   └── semantic_prior.py        # 语义先验数据结构
├── training/
│   ├── trainer.py               # 训练主流程
│   └── callback.py              # 周期性评估回调
├── evaluation/
│   ├── compare_methods.py       # 训练曲线对比图
│   └── compare_llm_fake_vs_real.py  # FakeLLM vs 真实LLM 对比
└── results/
    ├── checkpoints/             # 模型保存
    ├── logs/                    # 训练日志
    ├── figures/                 # 生成图表
    └── analysis/                # 汇总报告
```

---

## 使用方法

### 训练

```bash
# PPO Baseline（3次独立训练）
python experiments/run_seeds_ppo_baseline.py

# LLM-state
python experiments/run_seeds_ppo_llm_state.py

# LLM-reward
python experiments/run_seeds_ppo_llm_reward.py

# DQN Baseline
python experiments/run_seeds_dqn_baseline.py
```

### 消融实验

```bash
python experiments/run_ablation_llm_reward.py
```

### 评估与可视化

```bash
# 训练曲线对比图 + EV曲线
python evaluation/compare_methods.py

# FakeLLM vs 真实LLM 对比评估
python evaluation/compare_llm_fake_vs_real.py

# 轨迹可视化
python evaluation/plot_trajectory.py
```

### 调试工具

```bash
# 单步速度调试
python debug_speed.py

# 动作概率分布调试
python debug_action_prob.py

# 碰撞原因调试
python debug_crash.py

# 道路网络结构调试
python debug_road.py
```

---

## 实验设计

本项目对比以下四种方法：

| 方法 | 算法 | 融合方式 |
|------|------|------|
| PPO-Baseline | PPO | 无 |
| DQN-Baseline | DQN | 无 |
| LLM-state | PPO | 观测拼接语义先验 |
| LLM-reward | PPO | 奖励塑形 |

每种方法独立重复训练3次，采用固定24个评估种子，每隔6144步评估一次，取最后10次评估共240个回合奖励的中位数作为最终性能得分。

---

## 主要实验结果

| 方法 | 平均回合奖励±标准差 | 平均速度±标准差 | 碰撞率 |
|------|------|------|------|
| PPO-Baseline | 28.37±27.11 | 28.31±1.71 | 0.21±0.10 |
| DQN-Baseline | 26.01±25.33 | 24.29±1.07 | 0.26±2.53 |
| LLM-state | 38.70±24.70 | 29.74±0.21 | 0.22±0.24 |
| LLM-reward | 32.95±27.83 | 29.05±0.68 | 0.19±3.64 |

---

## 项目创新点

* 提出 LLM 语义先验与 PPO 强化学习的轻量化融合框架，不修改底层训练算法
* 系统对比状态融合与奖励塑形两种融合方式，验证奖励塑形在综合性能上的优势
* 设计 FakeLLM 规则代理机制，解决训练阶段实时调用 LLM 的成本与延迟问题
* 通过消融实验量化各语义先验维度的独立贡献，验证奖励加成函数设计的合理性

---

## 作者信息

* 姓名：周若瑜
* 本科专业：计算机科学与技术
* 项目类型：本科毕业设计
* 指导教师：费蓉老师

---

## 说明

本项目仅用于学术研究与毕业设计，不涉及商业用途。