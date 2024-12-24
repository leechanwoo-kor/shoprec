"""
test_inference.py
-----------------
batch_infer 로직 체크 등.
"""

import torch
from src.inference.batch_infer import batch_inference
from src.models.two_tower import TwoTowerModel


def test_batch_inference():
    model = TwoTowerModel(num_users=10, num_items=20, embed_dim=8)
    topk_indices = batch_inference(model, num_users=10, num_items=20, k=5)
    assert topk_indices.shape == (10, 5), "Should return (num_users, K) shape"
