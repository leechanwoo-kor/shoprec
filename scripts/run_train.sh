#!/usr/bin/env bash

echo "Running training..."
python -m src.training.trainer \
    --config config/params.yaml
echo "Training done."
