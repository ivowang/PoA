#!/usr/bin/env python3
"""
RL Training script for LoRA fine-tuning on os_interaction tasks.
Uses DPO (Direct Preference Optimization) algorithm.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

from src.language_models.instance.huggingface_lora_language_model import HuggingfaceLoRALanguageModel
from src.typings import ChatHistory, ChatHistoryItem, Role
from src.typings.session import SessionEvaluationOutcome
from src.typings.status import SampleStatus
from src.utils import ConfigLoader, SingletonLogger
from src.run_experiment import ConfigUtility, ConfigUtilityCaller
from src.tasks import Task, DatasetItem
from src.agents.instance.language_model_agent import LanguageModelAgent


class SuccessTrajectoryTrainer:
    """
    Trainer for fine-tuning LoRA adapters using only successful trajectories.
    
    This trainer maximizes the log probability of successful trajectories,
    which is equivalent to supervised fine-tuning on successful examples.
    
    Loss: L = -mean(log π_θ(y_success|x))
    where:
        - y_success: successful response
        - π_θ: policy model
        - x: prompt/context
    """
    
    def __init__(
        self,
        model: HuggingfaceLoRALanguageModel,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: Policy model to optimize (with LoRA)
            optimizer: Optimizer for the policy model
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
    
    def compute_loss(
        self,
        prompts: List[ChatHistory],
        successful_texts: List[str],
        system_prompt: str = "",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for successful trajectories.
        
        Args:
            prompts: List of chat histories (prompts)
            successful_texts: List of successful responses
            system_prompt: System prompt for the model
            
        Returns:
            Dict with loss and statistics
        """
        # Get log probs from policy model
        log_probs = self.model.get_log_probs(
            prompts, successful_texts, system_prompt=system_prompt
        )
        
        # Loss: maximize log probability = minimize negative log probability
        loss = -log_probs.mean()
        
        return {
            "loss": loss,
            "mean_log_prob": log_probs.mean(),
            "min_log_prob": log_probs.min(),
            "max_log_prob": log_probs.max(),
        }
    
    def update_with_gradient_accumulation(
        self,
        successful_trajectories: List[Dict[str, Any]],
    ):
        """
        Update model with gradient accumulation (does not call optimizer.step()).
        Caller should handle optimizer.step() and zero_grad().
        
        Args:
            successful_trajectories: List of dicts with keys:
                - prompt: ChatHistory (the context/prompt)
                - successful_text: str (successful response)
        """
        if len(successful_trajectories) == 0:
            return {"loss": 0.0, "mean_log_prob": 0.0}
        
        # Extract data
        prompts = [traj["prompt"] for traj in successful_trajectories]
        successful_texts = [traj["successful_text"] for traj in successful_trajectories]
        
        # Compute loss
        stats = self.compute_loss(prompts, successful_texts)
        loss = stats["loss"]
        
        # Check for numerical instability
        loss_value = loss.item()
        if abs(loss_value) > 1000.0 or not torch.isfinite(loss):
            import warnings
            warnings.warn(
                f"Large or non-finite loss detected: {loss_value:.4f}. "
                f"Mean log prob: {stats['mean_log_prob'].item():.4f}"
            )
            loss = torch.clamp(loss, min=-1000.0, max=1000.0)
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        return {
            "loss": loss.item(),
            "mean_log_prob": stats["mean_log_prob"].item(),
            "min_log_prob": stats["min_log_prob"].item(),
            "max_log_prob": stats["max_log_prob"].item(),
        }
    


def collect_trajectory_worker(args_tuple):
    """
    Worker function for parallel trajectory collection.
    Each worker loads its own model and task, collects trajectories independently.
    
    Args:
        args_tuple: Tuple containing:
            - worker_id: int
            - config_path: str
            - model_name: str
            - lora_config_dict: dict
            - lora_weights_path: Optional[str]
            - all_samples: List[str]
            - trajectories_per_worker: int
            - num_workers: int (total number of workers for stride calculation)
            - device_id: Optional[int] (GPU device ID)
    
    Returns:
        Dict with worker_id, trajectories, correct_count, failed_count
    """
    (
        worker_id,
        config_path,
        model_name,
        lora_config_dict,
        lora_weights_path,
        all_samples,
        trajectories_per_worker,
        num_workers,
        device_id,
    ) = args_tuple
    
    import os
    # Set CUDA device for this worker (before importing torch)
    # Note: CUDA_VISIBLE_DEVICES must be set before torch is imported
    # However, since torch is already imported in main process, we need a different approach
    # We'll use device_map to control which GPU to use
    worker_device_map = f"cuda:{device_id}" if device_id is not None else "auto"
    
    # Import after setting up device
    import torch
    from peft import LoraConfig
    
    # Load config in worker process
    from src.utils import ConfigLoader
    from src.run_experiment import ConfigUtility, ConfigUtilityCaller
    from src.agents.instance.language_model_agent import LanguageModelAgent
    from src.tasks import Task, DatasetItem
    from src.typings.session import SessionEvaluationOutcome
    
    raw_config = ConfigLoader().load_from(config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    
    # Create task for this worker
    task: Task[DatasetItem] = assignment_config.task.create()
    
    # Create LoRA config from dict
    lora_config = LoraConfig(**lora_config_dict)
    
    # Create model with LoRA (each worker loads its own copy)
    # Use device_map="auto" which will automatically distribute across available GPUs
    language_model = HuggingfaceLoRALanguageModel(
        model_name_or_path=model_name,
        role_dict={"user": "user", "agent": "assistant"},
        dtype=torch.bfloat16,
        device_map=worker_device_map,
        lora_config=lora_config,
        lora_weights_path=lora_weights_path,
        training_mode=False,  # Workers only do inference
    )
    
    # Create agent
    agent_config = assignment_config.agent
    agent = LanguageModelAgent(
        language_model=language_model,
        system_prompt="You are a helpful assistant.",
        inference_config_dict=agent_config.parameters.get("inference_config_dict", {}),
    )
    
    # Collect trajectories
    trajectories = []
    correct_count = 0
    failed_count = 0
    sample_idx = worker_id  # Start from different position for each worker
    total_attempts = 0
    
    while correct_count < trajectories_per_worker:
        current_sample = all_samples[sample_idx % len(all_samples)]
        # Stride by number of workers to avoid overlap between workers
        sample_idx += num_workers
        total_attempts += 1
        
        trajectory = collect_trajectory(task, agent, current_sample)
        
        # Print all trajectories (both success and failed) to terminal for monitoring
        if trajectory["generated_texts"]:
            outcome_str = trajectory["evaluation_outcome"].value if hasattr(trajectory["evaluation_outcome"], 'value') else str(trajectory["evaluation_outcome"])
            num_turns = len(trajectory["generated_texts"])
            reward = trajectory["reward"]
            
            # Print trajectory info to terminal (flush immediately for real-time display)
            status_icon = "✓" if trajectory["evaluation_outcome"] == SessionEvaluationOutcome.CORRECT else "✗"
            print(f"[Worker {worker_id}] {status_icon} Sample {current_sample}: {outcome_str} | "
                  f"Turns={num_turns} | Reward={reward:.2f} | "
                  f"Attempt {total_attempts}/{len(all_samples) * 10}", flush=True)
            
            # Print the last agent response (truncated if too long)
            last_response = trajectory["generated_texts"][-1]
            if len(last_response) > 200:
                last_response_preview = last_response[:200] + "..."
            else:
                last_response_preview = last_response
            print(f"  Response: {last_response_preview}", flush=True)
            print(flush=True)
            
            # Only keep successful trajectories, discard failed ones immediately to save memory
            if trajectory["evaluation_outcome"] == SessionEvaluationOutcome.CORRECT:
                # Only append successful trajectories
                trajectories.append(trajectory)
                correct_count += 1
            # Failed trajectories are discarded immediately - not stored in memory
        
        # Safety check
        if total_attempts > len(all_samples) * 10:
            break
    
    return {
        "worker_id": worker_id,
        "trajectories": trajectories,  # Only successful trajectories
        "correct_count": correct_count,
        "failed_count": 0,  # Not tracked anymore
        "total_attempts": total_attempts,
    }


def collect_trajectory(
    task: Task[DatasetItem],
    agent: LanguageModelAgent,
    sample_index: str,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """
    Collect a single trajectory by running agent on a task sample.
    
    Returns:
        Dict with:
            - chat_history: Full conversation
            - generated_texts: List of agent responses
            - log_probs: Log probabilities of generated texts
            - evaluation_outcome: Final evaluation result
            - reward: Computed reward
    """
    from src.typings.session import Session
    
    # Ensure task state is clean before reset
    # task.reset() requires current_sample_index to be None
    if hasattr(task, 'current_sample_index') and task.current_sample_index is not None:
        task.current_sample_index = None
        if hasattr(task, 'current_round'):
            task.current_round = 0
        if hasattr(task, '_Task__current_dataset_item'):
            task._Task__current_dataset_item = None
    
    # Initialize session
    session = Session(task_name=task.task_name, sample_index=sample_index)
    task.reset(session)
    
    generated_texts = []
    
    # Run interaction loop
    round_count = 0
    while session.sample_status == SampleStatus.RUNNING and round_count < max_rounds:
        # Agent inference
        agent.inference(session)
        
        # Get last agent response
        last_item = session.chat_history.get_item_deep_copy(-1)
        if last_item.role == Role.AGENT:
            generated_texts.append(last_item.content)
        
        # Task interaction
        task.interact(session)
        round_count += 1
    
    # Complete task if still running
    # complete() requires session.sample_status != RUNNING
    if session.sample_status == SampleStatus.RUNNING:
        # If still RUNNING after loop (e.g., reached max_rounds), set status first
        # This happens when our max_rounds is reached but task hasn't completed naturally
        session.sample_status = SampleStatus.TASK_LIMIT_REACHED
        if hasattr(task, '_get_default_task_output'):
            session.task_output = task._get_default_task_output()
        session.finish_reason = f"Training loop limit reached. The limit is {max_rounds}."
    
    # Now complete the task (status should not be RUNNING anymore)
    try:
        task.complete(session)
    except AssertionError as e:
        # If complete() still fails, set evaluation outcome to UNKNOWN
        # Note: logger is not available in worker processes, so we skip logging
        if session.evaluation_record.outcome == SessionEvaluationOutcome.UNSET:
            session.evaluation_record.outcome = SessionEvaluationOutcome.UNKNOWN
    
    # Always ensure task state is cleaned up for next sample
    # This is critical because task.reset() requires current_sample_index to be None
    if hasattr(task, 'current_sample_index') and task.current_sample_index == sample_index:
        task.current_sample_index = None
        if hasattr(task, 'current_round'):
            task.current_round = 0
        if hasattr(task, '_Task__current_dataset_item'):
            task._Task__current_dataset_item = None
    
    # Compute reward
    evaluation_outcome = session.evaluation_record.outcome
    reward = 1.0 if evaluation_outcome == SessionEvaluationOutcome.CORRECT else -0.0
    
    return {
        "chat_history": session.chat_history,
        "generated_texts": generated_texts,
        "evaluation_outcome": evaluation_outcome,
        "reward": reward,
        "sample_index": sample_index,
    }


def main():
    parser = argparse.ArgumentParser(description="RL Training with LoRA for os_interaction")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_checkpoints",
        help="Directory to save LoRA checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=None,
        help="Number of samples to collect per epoch (deprecated, use correct_samples_per_epoch instead)",
    )
    parser.add_argument(
        "--correct_samples_per_epoch",
        type=int,
        default=8,
        help="Number of CORRECT trajectories to collect per epoch (failed trajectories are discarded)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for DPO updates",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--training_epochs_per_batch",
        type=int,
        default=4,
        help="Number of training epochs per collected batch of successful trajectories",
    )
    parser.add_argument(
        "--num_parallel_workers",
        type=int,
        default=4,
        help="Number of parallel workers (agents) to collect trajectories simultaneously",
    )
    parser.add_argument(
        "--trajectories_per_worker",
        type=int,
        default=2,
        help="Number of successful trajectories each worker should collect",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to LoRA checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    raw_config = ConfigLoader().load_from(args.config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    logger = SingletonLogger.get_instance(logger_config)
    
    # Create task first
    task: Task[DatasetItem] = assignment_config.task.create()
    
    # Get task name for logging and validation
    task_name = task.task_name.value
    supported_tasks = ["os_interaction", "db_bench"]
    if task_name not in supported_tasks:
        logger.warning(
            f"Task '{task_name}' is not explicitly tested. "
            f"Supported tasks: {supported_tasks}. "
            f"Proceeding anyway..."
        )
    logger.info(f"Task type: {task_name}")
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    # Create model with LoRA
    # Get language model name from agent config
    agent_config = assignment_config.agent
    language_model_name = agent_config.parameters.get("language_model")
    if language_model_name is None:
        raise ValueError("Agent config must specify 'language_model' parameter")
    
    # Get language model config from language_model_dict
    language_model_config = assignment_config.language_model_dict[language_model_name]
    model_name = language_model_config.parameters["model_name_or_path"]
    
    # Create policy model with LoRA (will be trained)
    language_model = HuggingfaceLoRALanguageModel(
        model_name_or_path=model_name,
        role_dict={"user": "user", "agent": "assistant"},
        dtype=torch.bfloat16,
        device_map="auto",
        lora_config=lora_config,
        lora_weights_path=args.resume_from,
        training_mode=True,
    )
    
    # Create agent (reuse agent_config from above)
    agent = LanguageModelAgent(
        language_model=language_model,
        system_prompt="You are a helpful assistant.",
        inference_config_dict=agent_config.parameters.get("inference_config_dict", {}),
    )
    
    # Setup optimizer (only train LoRA parameters)
    trainable_params = [p for p in language_model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Create trainer for successful trajectories only
    trainer = SuccessTrajectoryTrainer(
        model=language_model,
        optimizer=optimizer,
        max_grad_norm=1.0,
    )
    
    # Get sample order (handle "default" case)
    if assignment_config.sample_order == "default":
        all_samples = task.get_sample_index_list()
    else:
        all_samples = list(assignment_config.sample_order)
    
    # Use correct_samples_per_epoch if specified, otherwise fall back to samples_per_epoch
    target_correct_samples = args.correct_samples_per_epoch
    if args.samples_per_epoch is not None and args.correct_samples_per_epoch == 8:  # Default value
        # Backward compatibility: if samples_per_epoch is set, use it as a limit
        target_correct_samples = args.samples_per_epoch
        logger.warning("Using deprecated --samples_per_epoch. Consider using --correct_samples_per_epoch instead.")
    
    logger.info(f"Starting training for {args.num_epochs} epochs (using only successful trajectories)")
    logger.info(f"Task: {task_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"Parallel workers: {args.num_parallel_workers}")
    logger.info(f"Trajectories per worker: {args.trajectories_per_worker}")
    logger.info(f"Total successful trajectories per epoch: {args.num_parallel_workers * args.trajectories_per_worker}")
    logger.info(f"Total available samples: {len(all_samples)}")
    
    # Prepare LoRA config dict for workers
    lora_config_dict = {
        "task_type": TaskType.CAUSAL_LM,
        "inference_mode": True,  # Workers only do inference
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    }
    
    # Determine GPU device IDs for workers
    # Note: Each worker will use device_map="auto" which automatically balances across GPUs
    # We still track device_id for potential future use, but currently all workers use "auto"
    num_gpus = torch.cuda.device_count()
    worker_devices = []
    for i in range(args.num_parallel_workers):
        if num_gpus > 0:
            # Distribute workers across available GPUs (for reference, actual mapping done by device_map="auto")
            device_id = i % num_gpus
        else:
            device_id = None  # CPU mode
        worker_devices.append(device_id)
    
    logger.info(f"Number of GPUs available: {num_gpus}")
    logger.info(f"Worker device assignment (for reference): {worker_devices}")
    logger.info(f"Note: Each worker uses device_map='auto' to balance GPU load")
    
    # Create temporary checkpoint path for sharing model weights between workers
    temp_checkpoint_dir = output_dir / "temp_worker_checkpoint"
    temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(args.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        
        # Save current model checkpoint for workers to load
        # Always save checkpoint so workers can load latest weights
        logger.info("Saving model checkpoint for workers...")
        language_model.save_lora_weights(str(temp_checkpoint_dir))
        worker_lora_path = str(temp_checkpoint_dir)
        
        # Prepare worker arguments
        worker_args = []
        for worker_id in range(args.num_parallel_workers):
            worker_args.append((
                worker_id,
                args.config_path,
                model_name,
                lora_config_dict,
                worker_lora_path,
                all_samples,
                args.trajectories_per_worker,
                args.num_parallel_workers,  # Pass num_workers for stride calculation
                worker_devices[worker_id],
            ))
        
        # Use multiprocessing Pool
        # Set start method before creating pool (only once, at the beginning)
        if epoch == 0:
            try:
                mp.set_start_method('spawn', force=True)  # Use spawn for CUDA compatibility
            except RuntimeError:
                pass  # Already set
        
        # Collect trajectories in parallel
        logger.info(f"Starting parallel trajectory collection with {args.num_parallel_workers} workers...")
        logger.info(f"Each worker will collect {args.trajectories_per_worker} successful trajectories")
        
        with Pool(processes=args.num_parallel_workers) as pool:
            worker_results = pool.map(collect_trajectory_worker, worker_args)
        
        logger.info(f"All {args.num_parallel_workers} workers completed trajectory collection")
        
        # Aggregate results from all workers
        all_trajectories = []
        total_correct = 0
        total_failed = 0
        total_attempts = 0
        
        for result in worker_results:
            all_trajectories.extend(result["trajectories"])
            total_correct += result["correct_count"]
            total_failed += result["failed_count"]
            total_attempts += result["total_attempts"]
            logger.info(
                f"Worker {result['worker_id']}: "
                f"Correct={result['correct_count']}, Failed={result['failed_count']}, "
                f"Total={len(result['trajectories'])}"
            )
        
        logger.info(f"Total successful trajectories collected: {len(all_trajectories)}")
        logger.info(f"Total CORRECT: {total_correct}")
        if total_attempts > 0:
            logger.info(f"Overall success rate: {total_correct/total_attempts*100:.1f}%")
        
        # Prepare successful trajectories for training
        # Only successful trajectories are kept (failed ones were discarded in workers)
        successful_trajectories = []
        
        for trajectory in all_trajectories:
            if not trajectory["generated_texts"]:
                continue
            
            # Only process successful trajectories
            if trajectory["evaluation_outcome"] != SessionEvaluationOutcome.CORRECT:
                continue
            
            # Create chat history up to (but not including) the last agent response
            last_chat_history = ChatHistory()
            chat_history_length = trajectory["chat_history"].get_value_length()
            for i in range(chat_history_length - 1):
                item_copy = trajectory["chat_history"].get_item_deep_copy(i)
                last_chat_history.inject(item_copy)
            
            successful_trajectories.append({
                "prompt": last_chat_history,
                "successful_text": trajectory["generated_texts"][-1],
                "sample_index": trajectory["sample_index"],
            })
        
        # Log summary
        logger.info(f"Epoch {epoch + 1} Collection Summary:")
        logger.info(f"  Successful trajectories prepared for training: {len(successful_trajectories)}")
        logger.info(f"  Total attempts: {total_attempts}")
        logger.info(f"  Success rate: {total_correct/total_attempts*100:.1f}%")
        
        if len(successful_trajectories) == 0:
            logger.warning("No successful trajectories collected, skipping epoch")
            continue
        
        # Set model to training mode (gradients needed)
        logger.info(f"Starting training updates - GPU utilization should be high here")
        logger.info(f"Total successful trajectories: {len(successful_trajectories)}, Batch size: {args.batch_size}, "
                   f"Gradient accumulation: {args.gradient_accumulation_steps}, "
                   f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
        language_model.model.train()
        
        # Training updates with gradient accumulation
        for training_epoch in range(args.training_epochs_per_batch):
            # Shuffle successful trajectories
            indices = torch.randperm(len(successful_trajectories))
            
            # Process with gradient accumulation
            optimizer.zero_grad()
            accumulated_stats = []
            batch_count = 0
            
            for i in range(0, len(successful_trajectories), args.batch_size):
                batch_indices = indices[i:i + args.batch_size]
                batch_trajectories = [successful_trajectories[idx] for idx in batch_indices]
                
                # Forward pass and accumulate gradients
                stats = trainer.update_with_gradient_accumulation(batch_trajectories)
                accumulated_stats.append(stats)
                batch_count += 1
                
                # Update weights every gradient_accumulation_steps batches
                if batch_count % args.gradient_accumulation_steps == 0 or (i + args.batch_size >= len(successful_trajectories)):
                    # Scale gradients by accumulation steps (for proper averaging)
                    for param in language_model.model.parameters():
                        if param.grad is not None:
                            param.grad.data.div_(args.gradient_accumulation_steps)
                    
                    # Clip gradients and step
                    torch.nn.utils.clip_grad_norm_(language_model.model.parameters(), trainer.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Log accumulated stats
                    if len(accumulated_stats) > 0:
                        avg_loss = sum(s['loss'] for s in accumulated_stats) / len(accumulated_stats)
                        avg_log_prob = sum(s['mean_log_prob'] for s in accumulated_stats) / len(accumulated_stats)
                        if (batch_count // args.gradient_accumulation_steps) % 5 == 0 or True:
                            logger.info(
                                f"Training Epoch {training_epoch + 1}, Update {batch_count // args.gradient_accumulation_steps}: "
                                f"Loss={avg_loss:.4f}, Mean Log Prob={avg_log_prob:.4f}, "
                                f"Effective batch size={len(accumulated_stats) * args.batch_size}"
                            )
                        accumulated_stats = []
        
        if epoch % 5 == 0:
            # Save checkpoint
            checkpoint_dir = output_dir / f"epoch_{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            language_model.save_lora_weights(str(checkpoint_dir))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Save training stats
            stats_file = checkpoint_dir / "training_stats.json"
            with open(stats_file, "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "num_successful_trajectories": len(successful_trajectories),
                    "total_correct": total_correct,
                    "total_failed": total_failed,
                    "total_attempts": total_attempts,
                    "success_rate": total_correct / total_attempts * 100 if total_attempts > 0 else 0.0,
                }, f, indent=2)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
