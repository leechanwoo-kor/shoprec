#!/usr/bin/env bash

# 전처리 실행 스크립트 예시
echo "Running preprocess..."
python -m src.data.preprocess \
    --input_path data/raw/event_log.csv \
    --output_path data/processed/event_log.parquet
echo "Preprocess done."
