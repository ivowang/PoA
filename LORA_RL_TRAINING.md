# LoRA RL Training Guide

This guide explains how to use LoRA + Reinforcement Learning for fine-tuning on os_interaction tasks.

## Overview

Instead of using experience replay, this implementation uses:
- **LoRA (Low-Rank Adaptation)** for efficient parameter-efficient fine-tuning
- **PPO (Proximal Policy Optimization)** for reinforcement learning
- **Reward signals** based on task evaluation outcomes (correct/incorrect)

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

The key additional dependencies are:
- `peft`: For LoRA adapters
- `accelerate`: For distributed training support

## Training

### Prerequisites: Start the Server

**⚠️ IMPORTANT:** You must start the server **before** running training, as the task uses client-server architecture.

**Terminal 1 - Start Server:**
```bash
export PYTHONPATH=.
python src/distributed_deployment_utils/start_server.py \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml
```

Keep this terminal running! The server must stay active during training.

**Terminal 2 - Run Training:**
```bash
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/train_lora_rl.py \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --output_dir ./lora_checkpoints \
    --num_epochs 10 \
    --samples_per_epoch 50 \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --ppo_epochs 4 \
    --lora_r 16 \
    --lora_alpha 32
```

### Using the Convenience Script

```bash
# Basic usage with defaults
./scripts/train_lora_rl.sh

# Custom parameters
./scripts/train_lora_rl.sh \
    configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    ./my_lora_checkpoints \
    20 \
    100
```

### Parameters

- `--config_path`: Path to experiment config file (must be os_interaction task)
- `--output_dir`: Directory to save LoRA checkpoints (default: `./lora_checkpoints`)
- `--num_epochs`: Number of training epochs (default: 10)
- `--samples_per_epoch`: Number of samples to collect per epoch (default: 50)
- `--learning_rate`: Learning rate for optimizer (default: 1e-5)
- `--batch_size`: Batch size for PPO updates (default: 4)
- `--ppo_epochs`: Number of PPO update epochs per batch (default: 4)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling factor (default: 32)
- `--resume_from`: Path to LoRA checkpoint to resume training from

### Training Process

1. **Trajectory Collection**: For each sample, the agent interacts with the task environment
2. **Reward Calculation**: 
   - `+1.0` for correct evaluation outcomes
   - `-0.5` for incorrect outcomes
   - `0.0` for unknown/unset outcomes
3. **PPO Updates**: Multiple epochs of PPO updates on collected trajectories
4. **Checkpoint Saving**: LoRA weights saved after each epoch

### Output Structure

```
lora_checkpoints/
├── epoch_1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_stats.json
├── epoch_2/
│   └── ...
└── ...
```

Each epoch directory contains:
- `adapter_config.json`: LoRA configuration
- `adapter_model.bin`: LoRA adapter weights
- `training_stats.json`: Training statistics (rewards, accuracy, etc.)

## Evaluation

### Option 1: Evaluate with LoRA Adapter (Recommended, No Merge)

**Default mode** - Loads base model + LoRA adapter dynamically:

```bash
python src/evaluate_lora.py \
    --mode evaluate \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --lora_weights_path ./lora_checkpoints/epoch_10 \
    --num_samples 50
```

**Advantages:**
- ✅ LoRA weights are small (~100MB vs ~60GB for merged model)
- ✅ Can quickly switch between different checkpoints
- ✅ Saves disk space
- ✅ No need to merge before evaluation

**How it works:** The script loads the base model and applies LoRA weights on-the-fly during inference.

### Option 2: Merge LoRA Weights First (For Deployment)

If you want a standalone merged model (e.g., for deployment or sharing):

```bash
# Step 1: Merge LoRA weights into base model
python src/evaluate_lora.py \
    --mode merge \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --lora_weights_path ./lora_checkpoints/epoch_10 \
    --merged_model_path ./merged_model

# Step 2: Evaluate merged model (faster inference)
python src/evaluate_lora.py \
    --mode evaluate \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --lora_weights_path ./lora_checkpoints/epoch_10 \
    --use_merged_model \
    --merged_model_path ./merged_model \
    --num_samples 50
```

**When to use merge:**
- 🚀 Need faster inference (single model load)
- 📦 Deploying to production (simpler deployment)
- 🔄 Sharing model without requiring base model + adapter
- ⚠️ **Note:** Merged model is ~60GB (same size as base model)

### Using the Convenience Script

```bash
# Evaluate with LoRA
./scripts/evaluate_lora.sh \
    configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    ./lora_checkpoints/epoch_10 \
    evaluate \
    50

# Merge and evaluate
./scripts/evaluate_lora.sh \
    configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    ./lora_checkpoints/epoch_10 \
    both \
    50
```

## How It Works

### Reward Signal

The reward is computed based on the final evaluation outcome:
- **Correct**: `+1.0` reward
- **Incorrect**: `-0.5` reward  
- **Unknown/Unset**: `0.0` reward

### PPO Algorithm

1. Collect trajectories by running the current policy
2. Compute advantages from rewards
3. Update policy using clipped PPO objective:
   - Maximize: `min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)`
   - Add entropy bonus for exploration
4. Repeat for multiple epochs

### LoRA Configuration

Default LoRA config:
- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target modules**: `["q_proj", "v_proj", "k_proj", "o_proj"]`

This targets the attention layers, which are most effective for task-specific adaptation.

## Tips

1. **Start Small**: Begin with fewer samples per epoch to test the setup
2. **Monitor Training**: Check `training_stats.json` after each epoch
3. **Resume Training**: Use `--resume_from` to continue from a checkpoint
4. **GPU Memory**: LoRA is memory-efficient, but 32B models still need significant GPU memory
5. **Evaluation**: Use merged models for faster inference during evaluation

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `samples_per_epoch`
- Use gradient checkpointing (can be added to model loading)

### Low Performance
- Increase `num_epochs`
- Increase `samples_per_epoch`
- Adjust `learning_rate` (try 5e-6 or 2e-5)
- Increase LoRA rank `lora_r` (try 32 or 64)

### Training Instability
- Reduce `learning_rate`
- Increase `ppo_epochs`
- Adjust reward scaling

## Example Workflow

### Step-by-Step Guide

**1. Start Server (Terminal 1 - Keep Running)**
```bash
cd ~/PoA
export PYTHONPATH=.
python src/distributed_deployment_utils/start_server.py \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml
```

**Verify server is running:**
```bash
# Test Task Server (port 8000)
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8000/api/ping

# Test Chat History Item Factory Server (port 8001)
curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:8001/api/ping
```

**2. Train LoRA (Terminal 2)**
```bash
cd ~/PoA
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Using script
./scripts/train_lora_rl.sh

# Or directly
python src/train_lora_rl.py \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --output_dir ./lora_checkpoints \
    --num_epochs 10 \
    --samples_per_epoch 50
```

**3. Evaluate Best Checkpoint (Terminal 2, after training)**
```bash
# Server must still be running in Terminal 1!
python src/evaluate_lora.py \
    --mode evaluate \
    --config_path configs/assignments/experiments/qwen25_32b_instruct/instance/os_interaction/instance/standard.yaml \
    --lora_weights_path ./lora_checkpoints/epoch_10 \
    --num_samples 100
```

**Note:** The server in Terminal 1 must remain running throughout training and evaluation!

## Notes

- This implementation is specifically for **os_interaction** tasks only
- The base model should be **qwen25_32b_instruct** (as specified)
- **LoRA checkpoints are much smaller** than full model checkpoints (~100MB vs ~60GB)
- **You don't need to merge for evaluation** - the default mode loads LoRA adapter dynamically
- **Merge only if** you need faster inference or want a standalone model for deployment
- Merged models require full model storage (~60GB) but provide faster inference
