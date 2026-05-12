from stable_baselines3 import DQN

def make_dqn_model(env, dqn_config, log_path, seed):
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=dqn_config.get("learning_rate", 1e-4),
        buffer_size=dqn_config.get("buffer_size", 100000),
        learning_starts=dqn_config.get("learning_starts", 1000),
        batch_size=dqn_config.get("batch_size", 64),
        gamma=dqn_config.get("gamma", 0.99),
        exploration_fraction=dqn_config.get("exploration_fraction", 0.1),
        exploration_final_eps=dqn_config.get("exploration_final_eps", 0.05),
        device=dqn_config.get("device", "cpu"),
        verbose=dqn_config.get("verbose", 1),
        seed=seed,
    )
    return model