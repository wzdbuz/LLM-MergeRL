# 基于大语言模型语义先验引导的强化学习自动驾驶决策系统（LLM-MergeRL）

## 项目简介

本项目旨在研究一种融合 **大语言模型（LLM）** 与 **强化学习（Reinforcement Learning, RL）** 的自动驾驶决策方法，通过引入语义先验信息，提升智能体在复杂交通环境中的决策能力。

传统强化学习方法在自动驾驶任务中通常存在以下问题：

* 奖励信号稀疏，学习效率低
* 对复杂场景理解能力不足
* 泛化能力较弱

针对上述问题，本项目提出：
利用大语言模型对场景进行语义理解，并生成**高层决策先验**，引导强化学习策略优化。

---

## 方法概述

本项目整体框架如下：

1. **语义先验生成（LLM模块）**

   * 输入：环境场景描述（如交通状况、天气、行人行为等）
   * 输出：语义先验信息（风险提示、行为建议等）

2. **状态编码（State Encoder）**

   * 将环境观测编码为神经网络可处理的特征表示

3. **强化学习策略（PPO）**

   * 基于 Proximal Policy Optimization 算法进行策略学习
   * 融合语义先验信息进行决策优化

4. **训练模块（Training Pipeline）**

   * 包括数据采样（rollout）、策略更新、日志记录等

---

## 项目结构

```id="6m3bne"
LLM-MergeRL/
│
├── LLM_module/            # 语义先验生成模块
│   └── semantic_prior.py
│
├── rl_agent/              # 强化学习算法（PPO）
│   ├── PPO.py
│   ├── network.py
│   └── __init__.py
│
├── state_encoder/         # 状态编码模块
│   ├── encoder.py
│   └── __init__.py
│
├── training/              # 训练流程
│   ├── trainer.py
│   ├── rollout.py
│   └── callback.py
│
├── utils/                 # 工具函数
│   ├── logger.py
│   ├── checkpoint.py
│   └── seed_manager.py
│
├── main.py                # 程序入口
└── requirements.txt       # 依赖文件
```

---

## 环境配置

建议使用 Python 3.8 及以上版本。

```bash
git clone https://github.com/wzdbuz/LLM-MergeRL.git
cd LLM-MergeRL

pip install -r requirements.txt
```

---

## 使用方法

运行训练程序：

```bash
python main.py
```

---

## 实验设计

本项目主要对比以下几种方法：

* 基线方法：传统强化学习（PPO）
* 改进方法：引入LLM语义先验的强化学习

评估指标包括：

* 平均奖励（Average Reward）
* 收敛速度（Convergence Speed）
* 成功率（Success Rate）
* 样本效率（Sample Efficiency）

---

## 项目创新点

* 引入大语言模型进行场景语义理解
* 将语义先验融入强化学习决策过程
* 提升复杂交通场景下的决策鲁棒性

---

## 后续工作

* 引入奖励塑形（Reward Shaping）进一步优化学习效率
* 提升语义先验的稳定性与一致性
* 扩展至多智能体协同决策场景
* 引入更真实的自动驾驶仿真环境

---

## 作者信息

* 姓名：周若瑜
* 本科专业：计算机科学与技术
* 项目类型：本科毕业设计

---

## 说明

本项目仅用于学术研究与毕业设计，不涉及商业用途。

---
