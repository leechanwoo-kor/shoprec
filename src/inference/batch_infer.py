# src/inference/batch_infer.py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from pathlib import Path

# (1) 모델 클래스 import
#     trainer.py에 정의된 TwoTowerModel을 import해야 합니다.
#     경로 구조에 따라 아래 import문을 수정하세요.
from src.training.trainer import TwoTowerModel

def batch_inference(model, user_encoder, item_encoder, K=10):
    """
    :param model: 학습된 TwoTower 모델 (이미 state_dict가 로드된 상태)
    :param user_encoder: user_id -> user_idx 변환(LabelEncoder)
    :param item_encoder: item_id -> item_idx 변환(LabelEncoder)
    :param K: 추천할 아이템 수
    :return: user_id -> list of top-K item_id 딕셔너리
    """
    model.eval()

    # 1) 유저/아이템 개수
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    with torch.no_grad():
        # 2) 전체 유저 임베딩
        user_indices = torch.arange(num_users)
        user_emb = model.user_embedding(user_indices)  # (num_users, embed_dim)

        # 3) 전체 아이템 임베딩
        item_indices = torch.arange(num_items)
        item_emb = model.item_embedding(item_indices)  # (num_items, embed_dim)

    # 4) dot product (전수 계산)
    scores = user_emb @ item_emb.T  # shape: (num_users, num_items)

    # 5) 각 유저별 Top-K
    topk_results = {}
    for u_idx in tqdm(range(num_users)):
        user_scores = scores[u_idx]  # (num_items,)
        topk_idx = torch.topk(user_scores, K).indices.tolist()  # 상위 K개 item_idx
        # item_idx -> item_id (역인코딩)
        topk_item_ids = item_encoder.inverse_transform(topk_idx)
        # user_idx -> user_id (역인코딩)
        user_id = user_encoder.inverse_transform([u_idx])[0]
        topk_results[user_id] = topk_item_ids
    
    return topk_results

if __name__ == "__main__":

    # ------------------------------------------------------
    # (2) model, encoders 로드
    # ------------------------------------------------------
    
    # (2-1) 인코더 로딩
    #   trainer.py 실행 직후, user_encoder / item_encoder를 joblib 등으로 따로 저장했을 것으로 가정
    #   예: joblib.dump(user_enc, "../models/user_encoder.pkl")
    #       joblib.dump(item_enc, "../models/item_encoder.pkl")
    user_encoder_path = Path("src/models/user_encoder.pkl")
    item_encoder_path = Path("src/models/item_encoder.pkl")
    user_encoder = joblib.load(user_encoder_path)
    item_encoder = joblib.load(item_encoder_path)

    # (2-2) 모델 객체 생성
    #   num_users, num_items는 인코더의 classes_ 길이와 동일하게 맞춰야 함
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    model = TwoTowerModel(num_users, num_items, embed_dim=32)

    # (2-3) 학습된 가중치 로드
    model_path = Path("src/models/two_tower.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------
    # (3) batch_inference 수행
    # ------------------------------------------------------
    user2items = batch_inference(model, user_encoder, item_encoder, K=10)
    
    # ------------------------------------------------------
    # (4) 결과 저장 (예: CSV)
    # ------------------------------------------------------
    # user2items는 { user_id: [item_id1, item_id2, ... item_idK], ... }
    data_for_csv = []
    for u_id, item_list in user2items.items():
        for rank, i_id in enumerate(item_list, start=1):
            data_for_csv.append([u_id, i_id, rank])
    
    df_result = pd.DataFrame(data_for_csv, columns=["user_id", "item_id", "rank"])
    df_result.to_csv("data/processed/batch_topk.csv", index=False)
    print("Batch inference done. Saved to data/processed/batch_topk.csv")
