from config import config

if config.mode == "baseline":
    config.use_llm_state = False
    config.use_llm_reward = False

elif config.mode == "llm_state":
    config.use_llm_state = True

elif config.mode == "llm_reward":
    config.use_llm_reward = True