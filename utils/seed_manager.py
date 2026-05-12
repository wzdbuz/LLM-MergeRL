import random
import numpy as np
import torch


def set_seed(seed: int):
    """统一设置所有随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"随机种子已设置为 {seed}")