# src/evaluation/metrics.py
import numpy as np

def recall_at_k(true_items, pred_items, k=10):
    """
    true_items: 실제로 구매(또는 클릭)된 아이템 리스트
    pred_items: 모델이 예측한 상위 K 아이템 리스트
    """
    # 교집합 / 실제 아이템 수
    # 실제론 유저별로 여러 개의 true item이 있을 수 있으니 확장 필요
    hits = len(set(true_items).intersection(set(pred_items[:k])))
    return hits / len(true_items) if len(true_items) > 0 else 0.0

def ndcg_at_k(...):
    """
    NDCG 계산 로직
    """
    pass
