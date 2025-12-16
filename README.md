# LifelongAgentBench Complete Setup Guide

## Step 1: Clone Repository and Install Dependencies

```bash
cd LifelongAgentBench

pip install -r requirements.txt

pip install pre-commit==4.0.1
pre-commit install
pre-commit run --all-files
```

## Step 2: Prepare Docker Images

```bash
docker pull mysql:8.0

docker pull ubuntu

docker build -f scripts/dockerfile/os_interaction/default scripts/dockerfile/os_interaction --tag local-os/default
```

## Step 3: Generate Chat History Items

```bash
cd LifelongAgentBench
export PYTHONPATH=./

python src/factories/chat_history_item/offline/construct.py
```

This will create the following directory structure:
- `chat_history_items/standard/` - Contains db_bench.json, os_interaction.json, knowledge_graph.json
- `chat_history_items/previous_sample_utilization/` - Contains the same files

## Step 4: Download Dataset

```bash
pip install hf-cli

huggingface-cli download csyq/LifelongAgentBench \
  --repo-type dataset \
  --local-dir ./data \
  --local-dir-use-symlinks False
```

## Step 5: Convert Dataset Format

The downloaded dataset is in parquet format and needs to be converted to JSON format required by the project:

```bash
export PYTHONPATH=./

python convert_parquet_to_json.py
```

This will generate:
- `data/v0303/db_bench/processed/v0317_first500/entry_dict.json`

## Step 6: Configure Model Paths

**Important**: 
- If using models that require authentication (e.g., Llama), you need to login to HuggingFace first:
  ```bash
  huggingface-cli login
  # or
  python -c "from huggingface_hub import login; login()"
  ```
- The first run will download the model (may take several hours depending on network speed)

## Step 7: Start the Server

Start the server in one terminal (**keep it running**):

```bash
export PYTHONPATH=./

python ./src/distributed_deployment_utils/start_server.py \
  --config_path ./configs/assignments/experiments/llama_31_8b_instruct/instance/db_bench/instance/standard.yaml
```

**Verify the server is running correctly**:
```bash
# Test Task Server (port 8000)
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/api/ping

# Test Chat History Item Factory Server (port 8001)
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8001/api/ping
```

Both commands should return `{"response":"Hello, World!"}`

## Step 8: Run Experiments

Run experiments in **another terminal**:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python ./src/run_experiment.py \
  --config_path "configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml"
```

Or use other model configurations:
```bash
# Using Qwen2.5-7B-Instruct
python ./src/run_experiment.py \
  --config_path "configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml"
```

## Step 9: View Results

Experiment results will be saved in the `outputs/` directory, with each experiment creating a timestamped subdirectory.
