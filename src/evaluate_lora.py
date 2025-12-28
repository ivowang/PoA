#!/usr/bin/env python3
"""
Evaluation script that loads LoRA weights and merges them with base model for inference.
"""

import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.language_models.instance.huggingface_lora_language_model import HuggingfaceLoRALanguageModel
from src.utils import ConfigLoader, SingletonLogger
from src.run_experiment import ConfigUtility, ConfigUtilityCaller
from src.tasks import Task, DatasetItem
from src.agents.instance.language_model_agent import LanguageModelAgent
from src.typings.session import Session, SessionEvaluationOutcome
from src.typings.status import SampleStatus


def merge_lora_to_base_model(
    base_model_path: str,
    lora_weights_path: str,
    output_path: str,
):
    """
    Merge LoRA weights into base model and save the merged model.
    
    Args:
        base_model_path: Path to base model
        lora_weights_path: Path to LoRA adapter weights
        output_path: Path to save merged model
    """
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA weights from {lora_weights_path}")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed!")


def evaluate_with_lora(
    config_path: str,
    lora_weights_path: str,
    use_merged_model: bool = False,
    merged_model_path: Optional[str] = None,
    num_samples: Optional[int] = None,
):
    """
    Evaluate model with LoRA weights on os_interaction task.
    
    Args:
        config_path: Path to experiment config
        lora_weights_path: Path to LoRA checkpoint
        use_merged_model: If True, use merged model instead of loading LoRA separately
        merged_model_path: Path to merged model (if use_merged_model=True)
        num_samples: Number of samples to evaluate (None = all)
    """
    # Load config
    raw_config = ConfigLoader().load_from(config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    logger = SingletonLogger.get_instance(logger_config)
    
    # Create task first
    task: Task[DatasetItem] = assignment_config.task.create()
    
    # Ensure os_interaction task
    assert task.task_name.value == "os_interaction", \
        "This script only supports os_interaction tasks"
    
    # Get model name
    # Get language model name from agent config
    agent_config = assignment_config.agent
    language_model_name = agent_config.parameters.get("language_model")
    if language_model_name is None:
        raise ValueError("Agent config must specify 'language_model' parameter")
    
    # Get language model config from language_model_dict
    language_model_config = assignment_config.language_model_dict[language_model_name]
    model_name = language_model_config.parameters["model_name_or_path"]
    
    if use_merged_model and merged_model_path:
        # Use merged model (no LoRA)
        from src.language_models.instance.huggingface_language_model import HuggingfaceLanguageModel
        language_model = HuggingfaceLanguageModel(
            model_name_or_path=merged_model_path,
            role_dict={"user": "user", "agent": "assistant"},
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        # Use LoRA model
        language_model = HuggingfaceLoRALanguageModel(
            model_name_or_path=model_name,
            role_dict={"user": "user", "agent": "assistant"},
            dtype=torch.bfloat16,
            device_map="auto",
            lora_weights_path=lora_weights_path,
            training_mode=False,
        )
    
    # Create agent (reuse agent_config from above)
    agent = LanguageModelAgent(
        language_model=language_model,
        system_prompt="You are a helpful assistant.",
        inference_config_dict=agent_config.parameters.get("inference_config_dict", {}),
    )
    
    # Get sample order
    sample_order = assignment_config.sample_order
    if num_samples:
        sample_order = sample_order[:num_samples]
    
    logger.info(f"Evaluating on {len(sample_order)} samples")
    logger.info(f"Model: {model_name}")
    logger.info(f"LoRA weights: {lora_weights_path}")
    
    # Evaluate
    results = []
    correct_count = 0
    
    for sample_idx in sample_order:
        session = Session(task_name=task.task_name, sample_index=sample_idx)
        task.reset(session)
        
        # Run interaction
        round_count = 0
        max_rounds = 10
        while session.sample_status == SampleStatus.RUNNING and round_count < max_rounds:
            agent.inference(session)
            task.interact(session)
            round_count += 1
        
        if session.sample_status == SampleStatus.RUNNING:
            task.complete(session)
        
        # Record result
        outcome = session.evaluation_record.outcome
        if outcome == SessionEvaluationOutcome.CORRECT:
            correct_count += 1
        
        results.append({
            "sample_index": sample_idx,
            "outcome": outcome.value,
            "status": session.sample_status.value,
        })
        
        logger.info(f"Sample {sample_idx}: {outcome.value}")
    
    # Print summary
    accuracy = correct_count / len(results) if results else 0.0
    logger.info(f"\n=== Evaluation Summary ===")
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    
    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model or merge LoRA weights")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merge", "evaluate", "both"],
        default="evaluate",
        help="Mode: merge LoRA weights, evaluate, or both",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--merged_model_path",
        type=str,
        default=None,
        help="Path to save/load merged model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--use_merged_model",
        action="store_true",
        help="Use merged model for evaluation (requires merged_model_path)",
    )
    
    args = parser.parse_args()
    
    # Get base model path from config
    raw_config = ConfigLoader().load_from(args.config_path)
    assignment_config, _, _, _ = ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    
    # Get language model name from agent config
    agent_config = assignment_config.agent
    language_model_name = agent_config.parameters.get("language_model")
    if language_model_name is None:
        raise ValueError("Agent config must specify 'language_model' parameter")
    
    # Get language model config from language_model_dict
    language_model_config = assignment_config.language_model_dict[language_model_name]
    base_model_path = language_model_config.parameters["model_name_or_path"]
    
    if args.mode in ["merge", "both"]:
        if not args.merged_model_path:
            # Default output path
            args.merged_model_path = os.path.join(args.lora_weights_path, "merged_model")
        
        merge_lora_to_base_model(
            base_model_path=base_model_path,
            lora_weights_path=args.lora_weights_path,
            output_path=args.merged_model_path,
        )
    
    if args.mode in ["evaluate", "both"]:
        evaluate_with_lora(
            config_path=args.config_path,
            lora_weights_path=args.lora_weights_path,
            use_merged_model=args.use_merged_model,
            merged_model_path=args.merged_model_path,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
