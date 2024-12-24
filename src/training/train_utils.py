# src/training/train_utils.py
import torch


def set_seed(seed: int = 42):
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 필요 시 cudnn 설정 등 추가
