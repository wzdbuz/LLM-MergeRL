import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import train
from evaluation.evaluator import evaluate

if __name__ == "__main__":
    train(mode="llm_reward", experiment_name="llm_reward")
    evaluate(mode="llm_reward", model_path="results/checkpoints/llm_reward/best_model")