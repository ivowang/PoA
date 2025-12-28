#!/bin/bash
# Evaluation script for LoRA model
#
# IMPORTANT: Make sure the server is running before executing this script!
# Start server with:
#   python src/distributed_deployment_utils/start_server.py --config_path <config_path>

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default config for os_interaction with qwen25_32b_instruct
CONFIG_PATH="${1:-configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml}"
LORA_WEIGHTS_PATH="${2:-./lora_checkpoints/epoch_10}"
MODE="${3:-evaluate}"  # merge, evaluate, or both
NUM_SAMPLES="${4:-50}"

echo "=========================================="
echo "LoRA Model Evaluation"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "LoRA weights: $LORA_WEIGHTS_PATH"
echo "Mode: $MODE"
echo "Samples: $NUM_SAMPLES"
echo ""
if [ "$MODE" != "merge" ]; then
    echo "⚠️  Make sure the server is running!"
    echo "   Test with: curl -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8000/api/ping"
fi
echo "=========================================="
echo ""

python src/evaluate_lora.py \
    --mode "$MODE" \
    --config_path "$CONFIG_PATH" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --num_samples "$NUM_SAMPLES"
