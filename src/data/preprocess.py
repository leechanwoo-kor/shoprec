# src/data/preprocess.py
import pandas as pd
import json
import argparse
import numpy as np

def parse_json_column(json_str):
    """
    JSON 문자열 -> Python dict
    파싱 실패 시 빈 dict 반환.
    """
    try:
        return json.loads(json_str)
    except:
        return {}

def main(input_path, output_path):
    # 1. 원본 데이터 로드
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw data loaded. shape = {df.shape}")

    # 2. 날짜/시간 컬럼 변환
    df["Event Date"] = pd.to_datetime(df["Event Date"], errors="coerce")
    df["Event Datetime"] = pd.to_datetime(df["Event Datetime"], errors="coerce")

    # 3. JSON 파싱 (Semantic Event Properties)
    df["parsed_json"] = df["Semantic Event Properties"].apply(parse_json_column)

    # 4. products 필드 추출
    #    products = [{"name":"...", "price":..., "productID":"...", "quantity":...}, ...]
    df["products"] = df["parsed_json"].apply(lambda x: x.get("products", []))

    # 5. explode로 product 배열을 개별 행으로 펼치기
    df_exploded = df.explode("products", ignore_index=True)

    # 6. product 정보 컬럼화
    #    products가 dict가 아닌 NaN인 행도 있을 수 있어, 안전 처리
    df_exploded["product_id"] = df_exploded["products"].apply(
        lambda x: x.get("productID", None) if isinstance(x, dict) else None
    )
    df_exploded["product_name"] = df_exploded["products"].apply(
        lambda x: x.get("name", None) if isinstance(x, dict) else None
    )
    df_exploded["product_price"] = df_exploded["products"].apply(
        lambda x: x.get("price", np.nan) if isinstance(x, dict) else np.nan
    )
    df_exploded["product_quantity"] = df_exploded["products"].apply(
        lambda x: x.get("quantity", np.nan) if isinstance(x, dict) else np.nan
    )

    # 7. 구매 여부(label) 생성 예시
    #    예: "Order Complete"가 들어간 Event Category면 label=1, 그 외=0
    #    (실제 Event Category 종류에 맞춰 변경)
    df_exploded["label"] = df_exploded["Event Category"].apply(
        lambda x: 1 if "Order Complete" in str(x) else 0
    )

    # 8. 불필요한 컬럼 정리 (parsed_json, products 등)
    df_exploded.drop(columns=["parsed_json", "products", "Semantic Event Properties"], inplace=True)

    # 9. 최종 컬럼 순서 정리(예: user_id, item_id, timestamp, price, quantity, label 등)
    #    실제로 필요한 컬럼만 남깁니다.
    df_exploded.rename(columns={
        "Hashed User ID": "user_id"
    }, inplace=True)

    # 필요에 따라 timestamp는 Event Datetime / Event Date / 둘 중 하나로 선택
    # 여기서는 Event Datetime 사용
    final_cols = [
        "user_id",
        "product_id",
        "Event Category",
        "Event Datetime",
        "product_name",
        "product_price",
        "product_quantity",
        "label"
    ]
    df_final = df_exploded[final_cols].copy()

    # 10. 간단 결측치 처리 예시(숫자 결측=0)
    df_final["product_price"] = df_final["product_price"].fillna(0)
    df_final["product_quantity"] = df_final["product_quantity"].fillna(0)

    # 11. 저장
    #     CSV or Parquet 포맷. 예: CSV로 저장
    df_final.to_parquet(output_path, index=False)
    print(f"[INFO] Preprocessed dataset saved to {output_path} - shape={df_final.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/raw/event_log.csv")
    parser.add_argument("--output_path", type=str, default="../data/processed/event_log.parquet")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
