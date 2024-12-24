# src/inference/batch_infer.py
import torch
import numpy as np


def batch_inference(model, num_users, num_items, k=10):
    """
    모든 user, item 임베딩을 추론 → dot product로 상위 K개 item 선정
    """
    model.eval()
    with torch.no_grad():
        user_ids = torch.arange(num_users)
        item_ids = torch.arange(num_items)
        user_emb = model.user_embedding(user_ids)  # (num_users, embed_dim)
        item_emb = model.item_embedding(item_ids)  # (num_items, embed_dim)

    # 예시: (num_users, num_items) dot product를 직접 계산 (소규모일 때)
    # 대규모 시에는 Approx. Nearest Neighbor(FAISS, Annoy 등) 사용
    scores = user_emb @ item_emb.T  # (num_users, num_items)

    # 각 유저별 Top-K
    topk_indices = torch.topk(scores, k, dim=1).indices  # (num_users, k)
    return topk_indices


if __name__ == "__main__":
    # 예시 실행
    pass
