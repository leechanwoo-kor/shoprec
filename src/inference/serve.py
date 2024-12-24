# src/inference/serve.py
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# 전역 모델 로드 (학습 완료된 파라미터)
model = None
# TODO: model = torch.load("path/to/model.pt")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json  # {"user_id": 123, "top_k": 10}
    user_id = data["user_id"]
    top_k = data.get("top_k", 10)

    # 모델 추론
    with torch.no_grad():
        user_emb = model.user_embedding(torch.tensor([user_id]))
        # ... item_emb 전처리, dot product, Top-K 추출

    # 예시 응답
    return jsonify({"user_id": user_id, "recommended_items": [100, 50, 200]})  # 예시


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
