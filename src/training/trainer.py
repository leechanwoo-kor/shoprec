# src/training/trainer.py
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


class RecDataset(Dataset):
    def __init__(self, df, user_encoder, item_encoder):
        self.user_ids = user_encoder.transform(df["user_id"].values)
        self.item_ids = item_encoder.transform(df["product_id"].values)
        self.labels = df["label"].values.astype(float)
        # 필요 시 기타 피처(예: price, quantity)도 추가 가능

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class TwoTowerModel(torch.nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embed_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim)

    def forward(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        # 내적을 로짓(logit)으로 사용
        logit = (u_emb * i_emb).sum(dim=-1)
        return logit


def train_model(train_df, valid_df, epochs=20, patience=3):
    # 1) user, item 인코딩
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_encoder.fit(pd.concat([train_df["user_id"], valid_df["user_id"]]))
    item_encoder.fit(pd.concat([train_df["product_id"], valid_df["product_id"]]))

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    train_dataset = RecDataset(train_df, user_encoder, item_encoder)
    valid_dataset = RecDataset(valid_df, user_encoder, item_encoder)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    model = TwoTowerModel(num_users, num_items, embed_dim=32)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")  # 현재까지의 최소 Validation Loss
    no_improvement = 0            # 개선되지 않은 epoch 누적 수

    for epoch in range(epochs):
        # === (1) TRAIN PHASE ===
        model.train()
        total_loss = 0
        for batch in train_loader:
            user_ids = batch["user"]
            item_ids = batch["item"]
            labels = batch["label"]

            optimizer.zero_grad()
            logits = model(user_ids, item_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # === (2) VALIDATION PHASE ===
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                user_ids = batch["user"]
                item_ids = batch["item"]
                labels = batch["label"]

                logits = model(user_ids, item_ids)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)

        print(
            f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # === (3) Early Stopping Check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), "src/models/best_model.pt")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return model, user_encoder, item_encoder


if __name__ == "__main__":
    # (예) load train_df, valid_df
    df_all = pd.read_parquet("data/processed/event_log.parquet")
    # 분할 로직 예시 (임의 80:20 split)
    msk = np.random.rand(len(df_all)) < 0.8
    train_df = df_all[msk]
    valid_df = df_all[~msk]

    model, user_enc, item_enc = train_model(train_df, valid_df)

    # 모델 파라미터 저장
    torch.save(model.state_dict(), "src/models/two_tower.pt")

    # 인코더(라벨인코더) 저장
    joblib.dump(user_enc, "src/models/user_encoder.pkl")
    joblib.dump(item_enc, "src/models/item_encoder.pkl")
