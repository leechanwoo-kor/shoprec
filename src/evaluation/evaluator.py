# src/evaluation/evaluator.py
import numpy as np
from .metrics import recall_at_k, ndcg_at_k


def evaluate_model(model, test_dataset):
    """
    예시:
    1) test_dataset에서 batch 로딩
    2) 모델 스코어링
    3) recall@K, ndcg@K 등 산출
    """
    # 간단 예시
    total_recall = 0
    num_samples = 0

    for batch in test_dataset:  # 실제론 DataLoader etc.
        user_id = batch["user"]
        item_id = batch["item"]
        label = batch["label"]
        # ... 모델로 추론, pred_items 구하기
        # recall = recall_at_k([...], pred_items, k=10)
        # total_recall += recall
        # num_samples += 1

    avg_recall = total_recall / num_samples
    return {"recall@10": avg_recall}
