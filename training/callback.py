import os
import csv
import time
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.highway_wrapper import make_env


def make_eval_callback(eval_env, save_path: str, log_path: str, eval_freq: int, n_eval_episodes: int):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )


class MetricsCallback(BaseCallback):
    """记录训练时每个 episode 的奖励与步长（可选）。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))
        return True


class MultiSeedEvalCallback(BaseCallback):
    """
    训练过程周期性评估（更适合论文复现实验口径）：
    - 每 eval_freq 步评估一次
    - 固定 12 个评估 seed：奇数次评估用前 6 个，偶数次评估用后 6 个（循环交替）
    - 每个 seed 跑 n_eval_episodes 个 episode（论文方案一般设为 1）
    - 记录 reward / 平均速度 / 碰撞，并保存到 CSV
    - 同时保存 evaluations.npz（包含 reward 序列 + crash/speed 曲线），便于画图
    """

    def __init__(
        self,
        *,
        env_config: dict,
        mode: str,
        use_fake_llm: bool,
        eval_freq: int,
        n_eval_episodes: int,
        n_eval_seeds: int,
        eval_seed_pool: list[int],
        best_model_save_path: str,
        log_path: str,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.env_config = env_config
        self.mode = mode
        self.use_fake_llm = use_fake_llm
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.n_eval_seeds = int(n_eval_seeds)
        self.eval_seed_pool = list(eval_seed_pool)
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic

        os.makedirs(self.best_model_save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        if len(self.eval_seed_pool) < self.n_eval_seeds:
            raise ValueError("eval_seed_pool length must be >= n_eval_seeds")

        self.best_mean_reward = -np.inf
        self._eval_idx = 0  # 从 0 开始计数；展示时用 eval_index = _eval_idx + 1

        self._timesteps: list[int] = []
        self._results: list[np.ndarray] = []
        self._ep_lengths: list[np.ndarray] = []
        self._crash_rates: list[float] = []
        self._mean_speeds: list[float] = []

        self._csv_path = os.path.join(self.log_path, "eval_results.csv")
        self._csv_inited = False

    def _ensure_csv(self) -> None:
        if self._csv_inited:
            return
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["eval_step", "eval_index", "seed", "episode_index", "reward", "mean_speed", "collision"])
        self._csv_inited = True

    def _next_eval_seeds(self) -> list[int]:
        # 论文方案：12 个 seed，奇数次评估用前 6 个，偶数次评估用后 6 个
        if len(self.eval_seed_pool) == 12 and self.n_eval_seeds == 6:
            return list(self.eval_seed_pool[:6] if (self._eval_idx % 2 == 0) else self.eval_seed_pool[6:])

        # fallback：轮转取 seed（保持可复现）
        start = (self._eval_idx * self.n_eval_seeds) % len(self.eval_seed_pool)
        return [self.eval_seed_pool[(start + i) % len(self.eval_seed_pool)] for i in range(self.n_eval_seeds)]

    # ↓ 修改1：加入 seed 参数
    def _rollout_one_episode(self, env: Monitor, seed: int) -> tuple[float, int, float, int]:
        obs, _ = env.reset(seed=seed)  # ← 每次 reset 传入 seed，确保场景可复现且各不相同
        terminated = truncated = False
        ep_reward = 0.0
        ep_len = 0
        crashed = 0
        speed_sum = 0.0

        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            if info.get("crashed", False):
                crashed = 1
            try:
                speed_sum += float(env.unwrapped.vehicle.speed)
            except Exception:
                pass

        mean_speed = speed_sum / max(ep_len, 1)
        return ep_reward, ep_len, mean_speed, crashed

    def _save_evaluations_npz(self) -> None:
        out_path = os.path.join(self.log_path, "evaluations.npz")
        np.savez(
            out_path,
            timesteps=np.asarray(self._timesteps, dtype=np.int64),
            results=np.asarray(self._results, dtype=np.float32),
            ep_lengths=np.asarray(self._ep_lengths, dtype=np.int64),
            crash_rates=np.asarray(self._crash_rates, dtype=np.float32),
            mean_speeds=np.asarray(self._mean_speeds, dtype=np.float32),
        )

    def _do_eval(self) -> None:
        self._ensure_csv()

        seeds = self._next_eval_seeds()
        all_rewards: list[float] = []
        all_lengths: list[int] = []
        all_mean_speeds: list[float] = []
        all_crashes: list[int] = []

        for seed in seeds:
            # ↓ 修改2：make_env 补全参数（原来是 make_env(...)）
            eval_env = Monitor(make_env(self.env_config, seed=seed, mode=self.mode, use_fake_llm=self.use_fake_llm))

            if self._eval_idx == 0 and seed == seeds[0]:
                print(f"[debug] eval env config: {eval_env.unwrapped.config}")

            for ep_i in range(self.n_eval_episodes):
                # ↓ 修改3：删除重复的一行，只保留传 seed 的调用
                ep_reward, ep_len, mean_speed, crashed = self._rollout_one_episode(eval_env, seed=seed)
                all_rewards.append(float(ep_reward))
                all_lengths.append(int(ep_len))
                all_mean_speeds.append(float(mean_speed))
                all_crashes.append(int(crashed))

                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            int(self.num_timesteps),
                            int(self._eval_idx + 1),
                            int(seed),
                            int(ep_i),
                            float(ep_reward),
                            float(mean_speed),
                            int(crashed),
                        ]
                    )

        mean_reward = float(np.mean(all_rewards)) if all_rewards else -np.inf
        crash_rate = float(np.mean(all_crashes)) if all_crashes else 0.0
        mean_speed = float(np.mean(all_mean_speeds)) if all_mean_speeds else 0.0

        # 记录到 logger（stdout/csv 等）
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/crash_rate", crash_rate)
        self.logger.record("eval/mean_speed", mean_speed)
        self.logger.record("eval/n_eval_episodes_total", len(all_rewards))
        self.logger.record("eval/n_eval_seeds", self.n_eval_seeds)
        self.logger.dump(self.num_timesteps)

        # 保存 best_model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        # 保存 evaluations.npz（reward 序列 + crash/speed 曲线）
        self._timesteps.append(int(self.num_timesteps))
        self._results.append(np.asarray(all_rewards, dtype=np.float32))
        self._ep_lengths.append(np.asarray(all_lengths, dtype=np.int64))
        self._crash_rates.append(crash_rate)
        self._mean_speeds.append(mean_speed)
        self._save_evaluations_npz()

        self._eval_idx += 1

        if self.verbose > 0:
            print(
                f"[eval] t={self.num_timesteps} mean_reward={mean_reward:.3f} crash_rate={crash_rate:.3f} "
                f"mean_speed={mean_speed:.3f} seeds={seeds}"
            )

    def _on_step(self) -> bool:
        # ↓ 修改4：加 self.num_timesteps > 0 避免训练开始第0步就触发评估
        if self.eval_freq > 0 and self.num_timesteps > 0 and (self.num_timesteps % self.eval_freq == 0):
            self._do_eval()
        return True