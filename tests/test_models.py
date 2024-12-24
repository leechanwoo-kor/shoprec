"""
test_models.py
--------------
TwoTowerModel 등이 제대로 forward/backward 되는지 간단 점검.
"""

import torch
from src.models.two_tower import TwoTowerModel


def test_two_tower_forward():
    model = TwoTowerModel(num_users=100, num_items=200, embed_dim=16)
    user_ids = torch.tensor([0, 1, 2])
    item_ids = torch.tensor([10, 20, 30])

    logits = model(user_ids, item_ids)
    assert logits.shape == (3,), "Output shape should be (batch, )"
