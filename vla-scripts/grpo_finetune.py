"""
Complete implementation of GRPO for OpenVLA using binary random rewards.
Includes full training pipeline with improved generation, reward calculation,
and training loop optimizations.
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
    -0.02872725307941437,
    -0.04170349963009357,
    -0.026093858778476715,
    -0.08092105075716972,
    -0.09288699507713317,
    -0.20718276381492615,
    0.0
])
max_values = np.array([
    0.028309678435325586,
    0.040855254605412394,
    0.040161586627364146,
    0.08192047759890528,
    0.07792850524187081,
    0.20382574498653397,
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
    num_generations: int = 2
    beta: float = 0.1
    temperature: float = 0.0
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

def get_per_token_logps(
    model: torch.nn.Module,
    generated_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities for generated action sequences,
    but only for tokens that occur after the token 259.
    
    This code assumes that in each sequence the first occurrence of token 259 
    (ignoring the very first token, if it is special) marks the point after which 
    the log probabilities should be kept.
    """
    batch_size = generated_ids.size(0)
    attention_mask = torch.ones_like(generated_ids)
    
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
    # (If you want to search the original generated_ids you’ll need to adjust for the shift.)
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
    save_dir: Path
) -> None:
    """Save model checkpoint."""
    
    checkpoint_dir = save_dir / f"checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.module.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

def train_step(
    model: DDP,
    ref_model: DDP,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    action_tokenizer: ActionTokenizer,
    converter: TokenActionConverter,
    config: GRPOVLAConfig,
    device_id: int
) -> Dict[str, float]:
    """Execute single GRPO training step."""
    
    # Process inputs
    inputs = {
        k: (v.to(device_id) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items() if k != 'dataset_names'
    }
    
    # Remove ground truth actions from input
    # print("in_ids: ", inputs["input_ids"])
    # print("labels: ", inputs["labels"])

    continuous_gt = converter.token_to_action(inputs["labels"].cpu().numpy())
    # print(continuous_gt)

    # Storage for multiple generations
    all_policy_logps = []
    all_ref_logps = []
    all_action_preds = []
    all_rewards = []
    
    # input_length = inputs["input_ids"].shape[1]
    
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
            inputs["pixel_values"]
        )
        
        with torch.no_grad():
            ref_logps = get_per_token_logps(
                ref_model.module,
                generated_ids,
                inputs["pixel_values"]
            )
        
        # actions_pred = generated_ids[:, input_length:]
        print(actions_pred)
        continuous_pred = converter.token_to_action(actions_pred.cpu().numpy())
        rewards = calculate_rewards(continuous_gt, continuous_pred, ranges)
        
        all_policy_logps.append(policy_logps)
        all_ref_logps.append(ref_logps)
        all_action_preds.append(actions_pred)
        all_rewards.append(rewards.to(device_id))
    
    # Stack results
    policy_logps = torch.stack(all_policy_logps, dim=1)
    ref_logps = torch.stack(all_ref_logps, dim=1)
    action_preds = torch.stack(all_action_preds, dim=1)
    rewards = torch.stack(all_rewards, dim=1)
    
    # Calculate KL divergence
    kl_div = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1
    
    # Calculate advantages
    advantages = (rewards - rewards.mean(dim=1, keepdim=True))
    advantages = advantages / (rewards.std(dim=1, keepdim=True) + 1e-8)
    advantages = advantages.unsqueeze(2)
    
    # Compute GRPO loss
    importance_weights = torch.exp(policy_logps - policy_logps.detach())
    policy_loss = -importance_weights * advantages
    total_loss = policy_loss + config.beta * kl_div
    loss = total_loss.mean()
    
    # Backward pass
    normalized_loss = loss / config.grad_accumulation_steps
    normalized_loss.backward()
    
    return {
        "loss": loss.item(),
        "reward": rewards.mean().item(),
        "kl": kl_div.mean().item(),
    }

@draccus.wrap()
def train_grpo_vla(cfg: GRPOVLAConfig) -> None:
    """Main training function for GRPO-VLA."""
    
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
        f"+grpo-g{cfg.num_generations}+b{cfg.beta}"
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
    
    # Create reference model
    ref_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    
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
    ref_vla = DDP(
        ref_vla,
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
    
    # Initialize optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Initialize metric tracking
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    recent_kls = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Execute training step
            metrics = train_step(
                vla,
                ref_vla,
                batch,
                optimizer,
                action_tokenizer,
                converter,
                cfg,
                device_id
            )
            
            # Store metrics
            recent_losses.append(metrics["loss"])
            recent_rewards.append(metrics["reward"])
            recent_kls.append(metrics["kl"])
            
            # Optimizer step if needed
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
                
                # Log metrics
                if distributed_state.is_main_process:
                    avg_metrics = {
                        "loss": sum(recent_losses) / len(recent_losses),
                        "reward": sum(recent_rewards) / len(recent_rewards),
                        "kl": sum(recent_kls) / len(recent_kls),
                    }
                    progress.set_postfix(avg_metrics)
                    # wandb.log(avg_metrics, step=batch_idx)
            
            # Save checkpoint
            if (batch_idx > 0 and 
                batch_idx % cfg.save_steps == 0 and 
                distributed_state.is_main_process):
                save_checkpoint(
                    vla,
                    processor,
                    batch_idx,
                    adapter_dir if cfg.use_lora else run_dir
                )
            
            # Check for max steps
            if batch_idx >= cfg.max_steps:
                break
    
    print(f"Training completed! Model saved to {run_dir}")

if __name__ == "__main__":
    train_grpo_vla()