import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

###########################
# NDCG 계산 유틸
###########################
def dcg_at_k(rel_list, k=10):
    """
    rel_list: 길이 k의 리스트, 각 아이템의 정답 여부 (1/0)
    DCG = sum_{i=1 to k} (rel_i / log2(i+1))
    """
    dcg = 0.0
    for i, rel in enumerate(rel_list[:k], start=1):
        dcg += rel / np.log2(i+1)
    return dcg

def ndcg_at_k(recommended_items, ground_truth_items, k=10):
    """
    recommended_items: 순위대로 정렬된 아이템 리스트 (길이 >= k)
    ground_truth_items: 정답 아이템 set (label=1)
    """
    # 실제 추천결과 중, 정답이면 1 / 아니면 0
    rel_list = [1 if item in ground_truth_items else 0 for item in recommended_items[:k]]
    
    dcg = dcg_at_k(rel_list, k)
    # ideal rel_list: 정답인 아이템이 모두 상위에 있다고 가정
    # ground_truth_items의 최대 수는 k개를 넘을 수 있음
    ideal_rel_list = sorted(rel_list, reverse=True)
    idcg = dcg_at_k(ideal_rel_list, k)
    
    return dcg / idcg if idcg > 0 else 0.0

###########################
# Recall@K, NDCG@K 계산
###########################
def evaluate_recall_ndcg(model, test_df, user_encoder, item_encoder, K=10):
    """
    model: 학습된 TwoTower 모델 (eval 모드)
    test_df: (user_id, product_id, label) 형태의 DataFrame (test split)
    user_encoder, item_encoder: 학습 시 사용했던 LabelEncoder
    K: top-K
    """

    model.eval()

    # 1) 유저별 실제 정답 아이템(라벨=1) 모으기
    #   user_id를 기준으로 groupby → label=1인 product_id들을 set으로 저장
    user2trueitems = defaultdict(set)
    for row in test_df.itertuples(index=False):
        u_id = getattr(row, "user_id")
        i_id = getattr(row, "product_id")
        label = getattr(row, "label")
        if label == 1:
            user2trueitems[u_id].add(i_id)

    # 2) 유저/아이템 임베딩 계산
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    with torch.no_grad():
        # user_idx: 0 .. num_users-1
        # product_idx: 0 .. num_items-1
        user_idx_tensor = torch.arange(num_users)
        product_idx_tensor = torch.arange(num_items)

        user_emb = model.user_embedding(user_idx_tensor)  # (num_users, embed_dim)
        item_emb = model.item_embedding(product_idx_tensor)  # (num_items, embed_dim)

    # 3) 전수 dot-product → (num_users x num_items) 점수 행렬
    #    (대규모 시 ANN 등 고려)
    scores = user_emb @ item_emb.T

    # 4) Recall, NDCG 계산
    sum_recall = 0.0
    sum_ndcg = 0.0
    total_users_with_truth = 0
    
    # user_idx -> 실제 user_id (역인코딩)
    for u_idx in range(num_users):
        user_id = user_encoder.inverse_transform([u_idx])[0]
        # 실제 정답 아이템
        ground_truth = user2trueitems.get(user_id, set())

        if len(ground_truth) == 0:
            # 이 유저는 test에서 label=1 아이템이 없음 → 스킵
            continue

        total_users_with_truth += 1

        # 해당 유저의 점수 (1D tensor, shape=(num_items,))
        user_scores = scores[u_idx]
        # Top-K 아이템 인덱스
        topk_idx = torch.topk(user_scores, K).indices.tolist()
        # 인덱스→아이디 역인코딩
        topk_product_ids = item_encoder.inverse_transform(topk_idx)
        
        # --- Recall@K ---
        # ground_truth와 교집합
        hits = len(set(topk_product_ids).intersection(ground_truth))
        recall_k = hits / len(ground_truth)  # 정답 아이템 중 몇 개 맞췄나
        sum_recall += recall_k

        # --- NDCG@K ---
        ndcg_k = ndcg_at_k(topk_product_ids, ground_truth, K)
        sum_ndcg += ndcg_k

    # 5) 평균값 계산
    mean_recall = sum_recall / total_users_with_truth if total_users_with_truth > 0 else 0.0
    mean_ndcg = sum_ndcg / total_users_with_truth if total_users_with_truth > 0 else 0.0

    return mean_recall, mean_ndcg

###########################
# 예시 실행부
###########################
if __name__ == "__main__":
    import joblib
    from src.training.trainer import TwoTowerModel

    # 1) test_df 로드 (예: test split만 추출해 parquet or csv 준비)
    test_df = pd.read_parquet("../data/processed/test_events.parquet")

    # 2) 인코더 로딩
    user_encoder = joblib.load("../models/user_encoder.pkl")
    item_encoder = joblib.load("../models/item_encoder.pkl")
    
    # 3) 모델 객체 생성 & 가중치 로드
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    model = TwoTowerModel(num_users, num_items, embed_dim=32)

    model_path = "../models/two_tower.pt"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # 4) Recall@10, NDCG@10 측정
    recall10, ndcg10 = evaluate_recall_ndcg(model, test_df, user_encoder, item_encoder, K=10)
    print(f"Recall@10: {recall10:.4f}, NDCG@10: {ndcg10:.4f}")

    # 필요 시 K=5,20 등 다양한 K 값으로 측정해보면 좋습니다.
