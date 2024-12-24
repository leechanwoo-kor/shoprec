# src/data/preprocess.py
import pandas as pd
from data_loader import load_raw_data
import os


def preprocess_main(input_path: str, output_path: str):
    """
    1) Raw 데이터 로딩
    2) 결측치/이상치 처리
    3) 카테고리 인코딩, Feature Engineering
    4) 결과 저장
    """
    df = load_raw_data(input_path)

    # 예시: 결측치 처리
    df.fillna({"column_x": 0}, inplace=True)
    # 예시: 범주형 인코딩, 스케일링 등

    # 저장
    df.to_parquet(output_path)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    # CLI 인자 파싱 후 사용 가능
    input_path = "../data/raw/user_item_log.csv"
    output_path = "../data/processed/user_item_log.parquet"
    preprocess_main(input_path, output_path)
