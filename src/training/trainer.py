# src/training/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ..models.two_tower import TwoTowerModel  # 상대경로 import 예시
from ..evaluation.metrics import compute_auc


def train_two_tower_model(
    train_dataset, num_users, num_items, embed_dim=32, epochs=5, batch_size=256, lr=1e-3
):
    model = TwoTowerModel(num_users, num_items, embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            user_ids = batch["user"]
            item_ids = batch["item"]
            labels = batch["label"].float()

            optimizer.zero_grad()
            logits = model(user_ids, item_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss = {total_loss / len(train_loader):.4f}")

    return model
