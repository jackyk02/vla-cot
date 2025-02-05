"""
grpo_vla_binary.py

Implementation of GRPO for OpenVLA using binary random rewards for testing and development.
Incorporates initialization patterns from original finetune script.
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import numpy as np
import math

from prismatic.vla.token2action import TokenActionConverter
converter = TokenActionConverter()

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class GRPOVLAConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # GRPO Specific Parameters
    num_generations: int = 2  # Number of trajectories per input
    beta: float = 0.1  # KL penalty coefficient
    temperature: float = 0.0
    max_prompt_length: int = 512
    max_completion_length: int = 512

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on

min_values = np.array([-0.02872725307941437,
          -0.04170349963009357,
          -0.026093858778476715,
          -0.08092105075716972,
          -0.09288699507713317,
          -0.20718276381492615,
          0.0])
max_values = np.array([0.028309678435325586,
          0.040855254605412394,
          0.040161586627364146,
          0.08192047759890528,
          0.07792850524187081,
          0.20382574498653397,
          1.0])
ranges = max_values - min_values

def calculate_nrmse(action_gt, action_sampled):
    l0 = action_gt
    l1 = action_sampled

    # Normalize the difference by the range
    normalized_diff = (l0 - l1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))

    return nrmse

# Function for getting token-level log-probs
def get_per_token_logps(model, generated_ids, original_input_length, pixel_values):
    """Compute per-token log probabilities for generated action sequences."""
    # Prepare full input sequence (prompt + generated actions)
    batch_size, seq_len = generated_ids.shape
    attention_mask = torch.ones_like(generated_ids)
    
    # Forward pass through model with complete sequence
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=generated_ids,
            pixel_values=pixel_values.to(torch.bfloat16),
            attention_mask=attention_mask
        )
    # Extract relevant logits for action tokens
    logits = output.logits[:, original_input_length-1:-1, :]  # (B, action_len, vocab)
    action_tokens = generated_ids[:, original_input_length:]   # (B, action_len)
    print("act tok:" , action_tokens)
    
    # Compute log probabilities for actual generated tokens
    log_probs = torch.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=action_tokens.unsqueeze(-1)).squeeze(-1)
    
    return per_token_logps

@draccus.wrap()
def train_grpo_vla(cfg: GRPOVLAConfig) -> None:
    """Main training function for GRPO-VLA."""
    
    print(f"Training OpenVLA Model `{cfg.vla_path}` with GRPO on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Training assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
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

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # Create reference model for KL divergence
    ref_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig`
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

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    ref_vla = DDP(ref_vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Initialize Dataset
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # Save Dataset Statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=0,  # Important =>> TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    # if distributed_state.is_main_process:
    #     wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Create Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Initialize training metrics tracking
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_rewards = deque(maxlen=cfg.grad_accumulation_steps)
    recent_kls = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Make image directory
    os.makedirs("images", exist_ok=True)

    # Training Loop
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            
            # Remove the gt action from input_id
            batch["input_ids"] = batch["input_ids"][:, :-8] 
            action_gt = batch["labels"][:, 1:][:, -8:-1]
            continuous_gt = converter.token_to_action(action_gt.cpu().numpy())

            # Preprocess Batch Data
            def convert_tensor(tensor):
                device = vla.module.device
                return (tensor.to(device) if tensor.dtype in [torch.long, torch.int] 
                        else tensor.to(device, dtype=torch.bfloat16))
                    
            # Remove dataset_names and convert remaining tensors
            inputs = {k: convert_tensor(v) for k, v in batch.items() 
                    if k != 'dataset_names'}

            # Modified training loop section
            original_input_length = inputs["input_ids"].shape[1]
            all_policy_logps = []
            all_ref_logps = []
            all_action_preds = []
            reward_preds = []
            
            # Sample G outputs 
            for _ in range(cfg.num_generations):
                # Generate outputs for each sample in the batch individually
                generated_ids_list = []
                batch_size = inputs["input_ids"].size(0)
                for i in range(batch_size):
                    # Create a single-example input by slicing each tensor along the batch dimension
                    single_input = {k: v[i : i + 1] for k, v in inputs.items()}
                    gen_ids = vla.module.generate(
                        **single_input,
                        max_new_tokens=vla.module.get_action_dim("bridge_orig"),
                        do_sample=False,
                        temperature=cfg.temperature,
                    )
                    generated_ids_list.append(gen_ids)

                # Concatenate the individual outputs into a single tensor
                generated_ids = torch.cat(generated_ids_list, dim=0)
                
                # Get policy model log probabilities
                policy_logps = get_per_token_logps(
                    vla.module, 
                    generated_ids,
                    original_input_length,
                    inputs["pixel_values"]
                )
                
                # Get reference model log probabilities
                with torch.no_grad():
                    ref_logps = get_per_token_logps(
                        ref_vla.module,
                        generated_ids,
                        original_input_length,
                        inputs["pixel_values"]
                    )
                
                # Store results
                print("generated", generated_ids)
                all_policy_logps.append(policy_logps)
                all_ref_logps.append(ref_logps)
                all_action_preds.append(generated_ids[:, original_input_length:])
                
                # Calculate reward
                continuous_sampled = converter.token_to_action(
                    generated_ids[:, original_input_length:].cpu().numpy()
                )
                rewards_gen = np.array([
                    calculate_nrmse(gt, cs)
                    for gt, cs in zip(continuous_gt, continuous_sampled)
                ])  # shape: (B,)
                print("rewards_gen", rewards_gen)
                # Use exponential decay on the error to get a reward; higher reward is better
                reward_vals = np.exp(-rewards_gen)  # shape: (B,)
                reward_preds.append(torch.tensor(reward_vals, device=device_id, dtype=torch.float32))

            # print(all_policy_logps)
            # print(all_ref_logps)
            # print(all_action_preds)
            # print(reward_preds)

            policy_logps = torch.stack(all_policy_logps, dim=1)  # (B, G, L)
            ref_logps = torch.stack(all_ref_logps, dim=1)        # (B, G, L)
            action_preds = torch.stack(all_action_preds, dim=1)  # (B, G, L)
            rewards = torch.stack(reward_preds, dim=1)           # (B, G)

            # Compute per-token KL divergence: (B, G, L)
            per_token_kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1

            # Create a mask for the generated action tokens.
            # Here we assume that groundtruth action tokens are used to define the valid region.
            # Expand groundtruth mask from shape (B, L) to (B, G, L) to match generated tokens.
            action_gt = action_gt.to(action_preds.device)
            mask = (action_gt > action_tokenizer.action_token_begin_idx)  # (B, L)
            mask = mask.unsqueeze(1).expand(-1, cfg.num_generations, -1)    # (B, G, L)

            # Compute advantages per sample and per generation.
            # Instead of reshaping, compute along the generation dimension directly.
            advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-4)
            # Expand advantages to match (B, G, L)
            advantages = advantages.unsqueeze(2)  # (B, G, 1)

            # Compute the GRPO loss:
            per_token_loss = torch.exp(policy_logps - policy_logps.detach()) * advantages
            per_token_loss = -(per_token_loss - cfg.beta * per_token_kl)
            # Average over token dimension and then over batch and generations
            loss = ((per_token_loss * mask).sum(dim=2) / mask.sum(dim=2)).mean()

            # Compute action accuracy for logging.
            # Expand groundtruth actions (B, L) to (B, G, L)
            action_gt_exp = action_gt.unsqueeze(1).expand_as(action_preds)
            correct_preds = (action_preds == action_gt_exp) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Backpropagation step
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Store metrics for logging
            recent_losses.append(loss.item())
            recent_rewards.append(rewards.mean().item())
            recent_kls.append(per_token_kl.mean().item())

            # Optimizer step if needed
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

                # Log metrics
                if distributed_state.is_main_process:
                    metrics = {
                        "loss": sum(recent_losses) / len(recent_losses),
                        "reward": sum(recent_rewards) / len(recent_rewards),
                        "kl": sum(recent_kls) / len(recent_kls),
                        "action_accuracy": action_accuracy.item()
                    }
                    progress.set_postfix(metrics)
                    # wandb.log(metrics, step=batch_idx)

            # Save checkpoint
            if batch_idx > 0 and batch_idx % cfg.save_steps == 0 and distributed_state.is_main_process:
                save_checkpoint(vla, processor, batch_idx, adapter_dir if cfg.use_lora else run_dir)


def save_checkpoint(model, processor, step, save_dir):
    """Helper function to save model checkpoint."""
    checkpoint_dir = save_dir / f"checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.module.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

if __name__ == "__main__":
    train_grpo_vla()