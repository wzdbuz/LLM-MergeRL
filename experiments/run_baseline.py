import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import train_baseline
from evaluation.evaluator import evaluate

if __name__ == "__main__":
    train_baseline()
    evaluate()