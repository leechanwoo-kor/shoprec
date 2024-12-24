# src/data/data_loader.py
import pandas as pd
import os


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    raw 디렉토리의 파일을 로딩하여 pandas DataFrame으로 반환.
    """
    df = pd.read_csv(file_path)
    return df


def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    processed 디렉토리의 전처리 완료 파일 로딩.
    """
    df = pd.read_parquet(file_path)
    return df
