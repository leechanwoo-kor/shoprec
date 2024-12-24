"""
test_data.py
------------
전처리/로더 동작이 정상인지 테스트(예: pytest).
"""

import pytest
from src.data.preprocess import preprocess_main
import os


def test_preprocess(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.parquet"

    # 임시 데이터 생성
    with open(input_path, "w") as f:
        f.write("user_id,item_id,action\n1,101,click\n2,202,purchase\n")

    preprocess_main(str(input_path), str(output_path))

    assert os.path.exists(output_path), "Preprocessed output file should exist"
