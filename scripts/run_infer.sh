#!/usr/bin/env bash

echo "Running batch inference..."
python -m src.inference.batch_infer \
    --model_path models/two_tower.pt \
    --top_k 10
echo "Batch inference done."
