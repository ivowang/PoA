#!/bin/bash
# Training script for LoRA fine-tuning using successful trajectories only
#
# This script automatically starts and stops the server for the specified environment.
#
# Usage:
#   ./scripts/train_lora_rl.sh [ENV] [OUTPUT_DIR] [NUM_EPOCHS] [NUM_WORKERS] [TRAJ_PER_WORKER]
#
#   ENV: Environment type - "os_interaction" or "db_bench" (default: "os_interaction")
#   OUTPUT_DIR: Directory to save checkpoints (default: "./lora_checkpoints")
#   NUM_EPOCHS: Number of training epochs (default: 100)
#   NUM_WORKERS: Number of parallel workers (default: 8)
#   TRAJ_PER_WORKER: Trajectories per worker (default: 2)

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Parse environment parameter
ENV="${1:-os_interaction}"
OUTPUT_DIR="${2:-./lora_checkpoints}"
NUM_EPOCHS="${3:-100}"
NUM_PARALLEL_WORKERS="${4:-8}"
TRAJECTORIES_PER_WORKER="${5:-2}"

# Validate environment
if [ "$ENV" != "os_interaction" ] && [ "$ENV" != "db_bench" ]; then
    echo "Error: Environment must be 'os_interaction' or 'db_bench'"
    echo "Got: $ENV"
    exit 1
fi

# Set config path based on environment
if [ "$ENV" == "os_interaction" ]; then
    CONFIG_PATH="configs/assignments/experiments/qwen25_7b_instruct/instance/os_interaction/instance/standard.yaml"
elif [ "$ENV" == "db_bench" ]; then
    CONFIG_PATH="configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml"
fi

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    echo ""
    echo "=========================================="
    echo "Cleaning up: Stopping server..."
    echo "=========================================="
    # Kill the server process if it's still running
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    # Also use shutdown script to ensure all related processes are terminated
    python src/distributed_deployment_utils/shutdown_server.py \
        --process_name "start_server.py" \
        --auto_confirm > /dev/null 2>&1
    echo "Server stopped."
    exit $exit_code
}

# Set trap to cleanup on script exit or interruption
trap cleanup EXIT INT TERM

# Start server in background
echo "=========================================="
echo "Starting server for environment: $ENV"
echo "=========================================="
python src/distributed_deployment_utils/start_server.py --config_path "$CONFIG_PATH" &
SERVER_PID=$!

# Wait for server to start (give it some time)
echo "Waiting for server to start..."
sleep 10

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start!"
    exit 1
fi

echo "Server started successfully (PID: $SERVER_PID)"
echo ""

echo "=========================================="
echo "LoRA Training (Parallel, Success Trajectories Only)"
echo "=========================================="
echo "Environment: $ENV"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Parallel workers: $NUM_PARALLEL_WORKERS"
echo "Trajectories per worker: $TRAJECTORIES_PER_WORKER"
echo "Total successful trajectories per epoch: $((NUM_PARALLEL_WORKERS * TRAJECTORIES_PER_WORKER))"
echo "(Each worker collects $TRAJECTORIES_PER_WORKER successful trajectories in parallel)"
echo "(Failed trajectories are discarded immediately to save memory)"
echo "(All trajectories will be printed to terminal for monitoring)"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train_lora_rl.py \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --num_parallel_workers "$NUM_PARALLEL_WORKERS" \
    --trajectories_per_worker "$TRAJECTORIES_PER_WORKER" \
    --learning_rate 2e-6 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --training_epochs_per_batch 4 \
    --lora_r 16 \
    --lora_alpha 32
