#!/usr/bin/env bash

# 전처리 실행 스크립트 예시
echo "Running preprocess..."
python -m src.data.preprocess \
    --input_path data/raw/user_item_log.csv \
    --output_path data/processed/user_item_log.parquet
echo "Preprocess done."
