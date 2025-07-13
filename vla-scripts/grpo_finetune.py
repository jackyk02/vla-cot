"""
Complete implementation of GRPO for OpenVLA using binary random rewards.
Includes full training pipeline with improved generation, reward calculation,
training loop optimizations, and comprehensive metrics.
Reference policy and KL divergence removed for simplicity.
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import wandb
import numpy as np
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig,
    AutoImageProcessor,
)

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.token2action import TokenActionConverter

# Action normalization constants
min_values = np.array([
    -0.7454732114076613,
    -0.6616071462631226,
    -0.9375,
    -0.1071428582072258,
    -0.20678570866584778,
    -0.1842857152223587,
    0.0
])
max_values = np.array([
    0.9375,
    0.8758928775787354,
    0.9321428537368774,
    0.1039285734295845,
    0.17678570747375488,
    0.14571428298950195,
    1.0
])
ranges = max_values - min_values

@dataclass
class GRPOVLAConfig:
    """Configuration for GRPO-VLA training."""
    
    # Model and Data Paths
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    # Training Parameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True

    # GRPO Specific Parameters
    num_generations: int = 8
    temperature: float = 0.9
    max_prompt_length: int = 512
    max_completion_length: int = 512

    # LoRA Parameters
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    # Logging Parameters
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None
    log_every: int = 5
    eval_every: int = 1000
    eval_samples: int = 100

def get_per_token_logps(
    model: torch.nn.Module,
    generated_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    pad_token_id : int
) -> torch.Tensor:
    """Compute per-token log probabilities for generated action sequences,
    but only for tokens that occur after the token 259.
    
    This code assumes that in each sequence the first occurrence of token 259 
    (ignoring the very first token, if it is special) marks the point after which 
    the log probabilities should be kept.
    """
    batch_size = generated_ids.size(0)
    attention_mask = (generated_ids != pad_token_id).long()
    
    # Compute model outputs under autocast
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=generated_ids,
            pixel_values=pixel_values.to(torch.bfloat16),
            attention_mask=attention_mask,
            return_dict=True
        )
    
    # ---
    # Standard practice: we predict token t+1 given token t.
    # So we use the logits for positions 0 ... (L-2) to predict tokens 1 ... (L-1)
    # (Note that generated_ids has shape (batch, L) and logits has shape (batch, L, vocab_size)
    #  so we slice logits to (batch, L-1, vocab_size) and generated_ids accordingly.)
    logits = output.logits[:, :-1]          # shape: (batch_size, seq_len-1, vocab_size)
    shifted_ids = generated_ids[:, 1:]        # shape: (batch_size, seq_len-1)
    
    # Compute log probabilities from logits.
    # log_probs has shape: (batch_size, seq_len-1, vocab_size)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather the log probability for each generated token
    per_token_logps = log_probs.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
    
    # ---
    # Find the first occurrence of token 259 in the shifted sequences.
    # (If you want to search the original generated_ids you'll need to adjust for the shift.)
    token_starts = []
    for b in range(batch_size):
        # Find indices in the shifted sequence where the token equals 259.
        # (We assume there is at least one occurrence; otherwise we default to the beginning.)
        idx = (shifted_ids[b] == 259).nonzero(as_tuple=False) + 1
        if idx.numel() > 0:
            # Use the first occurrence; note that the token following 259 is at the same index in per_token_logps.
            start = idx[0].item()
        else:
            start = 0
        token_starts.append(start)
    
    # For each sequence in the batch, we now want only the log probabilities
    # that occur at or after the found start index.
    # First, find the maximum remaining length over the batch for proper padding.
    max_length = max(per_token_logps[b, token_starts[b]:].size(0) for b in range(batch_size))
    result = torch.zeros((batch_size, max_length), device=per_token_logps.device, dtype=per_token_logps.dtype)
    
    for b in range(batch_size):
        start = token_starts[b]
        # Slice out log probs for tokens after (and including) the token following 259.
        # (Because the shifted sequence has the token that comes after each original token.)
        batch_logps = per_token_logps[b, start:]
        result[b, :batch_logps.size(0)] = batch_logps

    # (Optional) print inputs for debugging
    result = result[:, :7]
    return result

def calculate_rewards(
    action_gt: np.ndarray,
    action_sampled: np.ndarray,
    ranges: np.ndarray
) -> torch.Tensor:
    """Calculate normalized RMSE rewards with exponential scaling."""
    
    normalized_diff = (action_gt - action_sampled) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2, axis=1))
    rewards = np.exp(-nrmse)
    return torch.tensor(rewards, dtype=torch.float32)

def calculate_action_metrics(
    action_gt: np.ndarray,
    action_sampled: np.ndarray,
    ranges: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive action prediction metrics."""
    
    # Per-dimension errors
    errors = np.abs(action_gt - action_sampled)
    normalized_errors = errors / ranges
    
    # Mean Absolute Error (MAE) per dimension
    mae_per_dim = np.mean(errors, axis=0)
    
    # Root Mean Square Error (RMSE) per dimension
    rmse_per_dim = np.sqrt(np.mean(errors**2, axis=0))
    
    # Normalized errors
    nmae = np.mean(normalized_errors)
    nrmse = np.sqrt(np.mean(normalized_errors**2))
    
    # Success metrics (within threshold)
    success_5_percent = np.mean(np.all(normalized_errors < 0.05, axis=1))
    success_10_percent = np.mean(np.all(normalized_errors < 0.10, axis=1))
    success_20_percent = np.mean(np.all(normalized_errors < 0.20, axis=1))
    
    # Gripper accuracy (last dimension)
    gripper_correct = np.mean(
        (action_gt[:, -1] > 0.5) == (action_sampled[:, -1] > 0.5)
    )
    
    metrics = {
        "mae": np.mean(mae_per_dim),
        "rmse": np.mean(rmse_per_dim),
        "nmae": nmae,
        "nrmse": nrmse,
        "success_5_percent": success_5_percent,
        "success_10_percent": success_10_percent,
        "success_20_percent": success_20_percent,
        "gripper_accuracy": gripper_correct,
    }
    
    # Per-dimension metrics
    dim_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    for i, name in enumerate(dim_names):
        metrics[f"mae_{name}"] = mae_per_dim[i]
        metrics[f"rmse_{name}"] = rmse_per_dim[i]
    
    return metrics

def remove_padding(batch):
    # Find where the padding tokens (32000) start in input_ids
    non_padding_mask = batch['input_ids'][0] != 32000
    
    # Get the actual sequence length (before padding)
    actual_length = torch.sum(non_padding_mask)
    
    # Slice the input_ids and attention_mask
    batch['input_ids'] = batch['input_ids'][:, :actual_length]
    batch['attention_mask'] = batch['attention_mask'][:, :actual_length]
    
    return batch

def generate_with_padding(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    pad_token_id: int,
    temperature: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate tokens with proper padding handling and return action predictions.
    Returns:
        Tuple containing:
        - Padded generated sequences
        - Action predictions (last 7 tokens of each sequence)
    """
    
    batch_size = inputs["input_ids"].size(0)
    generated = []
    action_preds = []
    
    for i in range(batch_size):
        single_input = {k: v[i:i+1] for k, v in inputs.items()}
        single_input = remove_padding(single_input)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen_ids = model.generate(
                **single_input,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                pad_token_id=pad_token_id
            )
        generated.append(gen_ids)
        # Extract last 7 tokens for action prediction
        action_preds.append(gen_ids[:, -7:])
    
    max_len = max(ids.size(1) for ids in generated)
    padded = []
    
    for gen in generated:
        padding_needed = max_len - gen.size(1)
        if padding_needed > 0:
            padding = torch.full(
                (1, padding_needed),
                pad_token_id,
                device=gen.device,
                dtype=gen.dtype
            )
            gen = torch.cat([gen, padding], dim=1)
        padded.append(gen)
        
    return torch.cat(padded, dim=0), torch.cat(action_preds, dim=0)

def save_checkpoint(
    model: DDP,
    processor: Any,
    step: int,
    save_dir: Path,
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """Save model checkpoint with optional metrics."""
    
    checkpoint_dir = save_dir / f"checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.module.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # Save metrics if provided
    if metrics:
        import json
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

def evaluate(
    model: DDP,
    dataloader: DataLoader,
    action_tokenizer: ActionTokenizer,
    converter: TokenActionConverter,
    config: GRPOVLAConfig,
    device_id: int,
    num_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model performance on a subset of data."""
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            # Process inputs
            inputs = {
                k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items() if k != 'dataset_names'
            }
            
            # Get ground truth actions
            continuous_gt = converter.token_to_action(inputs["labels"].cpu().numpy())
            
            # Generate predictions with greedy decoding
            generated_ids, actions_pred = generate_with_padding(
                model.module,
                inputs,
                model.module.get_action_dim("bridge_orig"),
                model.module.config.pad_token_id,
                temperature=0.0  # Greedy decoding for evaluation
            )
            
            # Convert to continuous actions
            continuous_pred = converter.token_to_action(actions_pred.cpu().numpy())
            
            # Calculate metrics
            metrics = calculate_action_metrics(continuous_gt, continuous_pred, ranges)
            all_metrics.append(metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[f"eval/{key}"] = np.mean([m[key] for m in all_metrics])
    
    model.train()
    return avg_metrics

def train_step(
    model: DDP,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    action_tokenizer: ActionTokenizer,
    converter: TokenActionConverter,
    config: GRPOVLAConfig,
    device_id: int
) -> Dict[str, float]:
    """Execute single GRPO training step with comprehensive metrics."""
    
    # Process inputs
    inputs = {
        k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items() if k != 'dataset_names'
    }
    
    # Remove ground truth actions from input
    action_gt = inputs["labels"]
    continuous_gt = converter.token_to_action(inputs["labels"].cpu().numpy())

    # Storage for multiple generations
    all_policy_logps = []
    all_action_preds = []
    all_rewards = []
    all_continuous_preds = []
    
    # Generate multiple trajectories
    for _ in range(config.num_generations):
        generated_ids, actions_pred = generate_with_padding(
            model.module,
            inputs,
            model.module.get_action_dim("bridge_orig"),
            model.module.config.pad_token_id,
            config.temperature
        )
                
        policy_logps = get_per_token_logps(
            model.module,
            generated_ids,
            inputs["pixel_values"],
            model.module.config.pad_token_id
        )
        
        continuous_pred = converter.token_to_action(actions_pred.cpu().numpy())
        rewards = calculate_rewards(continuous_gt, continuous_pred, ranges)
        
        all_policy_logps.append(policy_logps)
        all_action_preds.append(actions_pred)
        all_rewards.append(rewards.to(device_id))
        all_continuous_preds.append(continuous_pred)
    
    # Stack results
    policy_logps = torch.stack(all_policy_logps, dim=1)
    action_preds = torch.stack(all_action_preds, dim=1)
    rewards = torch.stack(all_rewards, dim=1)
    
    # Calculate advantages
    advantages = (rewards - rewards.mean(dim=1, keepdim=True))
    advantages = advantages / (rewards.std(dim=1, keepdim=True) + 1e-4)
    advantages = advantages.unsqueeze(2)
    
    # Compute simplified GRPO loss (without KL divergence)
    importance_weights = torch.exp(policy_logps - policy_logps.detach())
    loss = -(importance_weights * advantages).mean()
    
    # Calculate additional metrics
    # Best generation metrics
    best_indices = rewards.argmax(dim=1)
    best_continuous_preds = np.stack([
        all_continuous_preds[i][j] 
        for j, i in enumerate(best_indices.cpu().numpy())
    ])
    best_metrics = calculate_action_metrics(continuous_gt, best_continuous_preds, ranges)
    
    # Average generation metrics
    avg_continuous_preds = np.mean(all_continuous_preds, axis=0)
    avg_metrics = calculate_action_metrics(continuous_gt, avg_continuous_preds, ranges)
    
    # Diversity metrics
    action_std = np.std(all_continuous_preds, axis=0).mean()
    
    # Backward pass
    normalized_loss = loss / config.grad_accumulation_steps
    normalized_loss.backward()
    
    metrics = {
        "loss": loss.item(),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "reward_max": rewards.max().item(),
        "reward_min": rewards.min().item(),
        "importance_weights_mean": importance_weights.mean().item(),
        "importance_weights_std": importance_weights.std().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
        "action_diversity": action_std,
        "best_reward": rewards.max(dim=1)[0].mean().item(),
    }
    
    # Add best generation metrics
    for key, value in best_metrics.items():
        metrics[f"best_{key}"] = value
        
    # Add average generation metrics
    for key, value in avg_metrics.items():
        metrics[f"avg_{key}"] = value
    
    return metrics

class MetricsTracker:
    """Track and aggregate training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
            self.metrics[key].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all tracked metrics."""
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.metrics.items()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()

@draccus.wrap()
def train_grpo_vla(cfg: GRPOVLAConfig) -> None:
    """Main training function for GRPO-VLA with comprehensive metrics."""
    
    print(f"Training OpenVLA Model `{cfg.vla_path}` with GRPO on `{cfg.dataset_name}`")
    
    # Setup distributed training
    assert torch.cuda.is_available(), "Training requires at least one GPU!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    
    # Configure experiment ID and directories
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+grpo-g{cfg.num_generations}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    
    # Setup quantization if needed
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantization requires LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    
    # Register OpenVLA components
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Device placement
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)
    
    # Setup LoRA if enabled
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    
    # Wrap models in DDP
    vla = DDP(
        vla,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=True
    )
    
    # Initialize action tokenizer and dataset components
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    converter = TokenActionConverter()
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path 
        else VicunaV15ChatPromptBuilder,
    )
    
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    # Save dataset statistics on main process
    if distributed_state.is_main_process:
        save_dataset_statistics(dataset.dataset_statistics, run_dir)
    
    # Setup data loading
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=0,  # TFDS handles parallelism
    )
    
    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Initialize optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(window_size=cfg.grad_accumulation_steps)
    
    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Execute training step
            metrics = train_step(
                vla,
                batch,
                optimizer,
                action_tokenizer,
                converter,
                cfg,
                device_id
            )
            
            # Update metrics tracker
            metrics_tracker.update(metrics)
            
            # Optimizer step if needed
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
                
                # Get averaged metrics
                avg_metrics = metrics_tracker.get_averages()
                
                # Log metrics
                if distributed_state.is_main_process and batch_idx % cfg.log_every == 0:
                    log_metrics = {f"train/{k}": v for k, v in avg_metrics.items()}
                    progress.set_postfix({
                        "loss": avg_metrics.get("loss", 0),
                        "reward": avg_metrics.get("reward_mean", 0),
                        "best_mae": avg_metrics.get("best_mae", 0)
                    })
                    wandb.log(log_metrics, step=batch_idx)
            
            # Evaluation
            if (batch_idx > 0 and 
                batch_idx % cfg.eval_every == 0 and 
                distributed_state.is_main_process):
                eval_metrics = evaluate(
                    vla,
                    dataloader,
                    action_tokenizer,
                    converter,
                    cfg,
                    device_id,
                    num_samples=cfg.eval_samples
                )
                wandb.log(eval_metrics, step=batch_idx)
                print(f"\nEvaluation at step {batch_idx}:")
                for key, value in eval_metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            # Save checkpoint
            if (batch_idx > 0 and 
                batch_idx % cfg.save_steps == 0 and 
                distributed_state.is_main_process):
                current_metrics = metrics_tracker.get_averages()
                save_checkpoint(
                    vla,
                    processor,
                    batch_idx,
                    adapter_dir if cfg.use_lora else run_dir,
                    metrics=current_metrics
                )
            
            # Check for max steps
            if batch_idx >= cfg.max_steps:
                break
    
    # Final evaluation
    if distributed_state.is_main_process:
        print("\nRunning final evaluation...")
        final_eval_metrics = evaluate(
            vla,
            dataloader,
            action_tokenizer,
            converter,
            cfg,
            device_id,
            num_samples=cfg.eval_samples * 2  # More samples for final eval
        )
        wandb.log(final_eval_metrics, step=cfg.max_steps)
        print("\nFinal evaluation results:")
        for key, value in final_eval_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save final checkpoint with metrics
        save_checkpoint(
            vla,
            processor,
            cfg.max_steps,
            adapter_dir if cfg.use_lora else run_dir,
            metrics=final_eval_metrics
        )
    
    print(f"Training completed! Model saved to {run_dir}")

if __name__ == "__main__":
    train_grpo_vla()