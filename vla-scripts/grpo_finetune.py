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

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
import json_numpy as json

def get_batch_actions(instruction: str, image_path: str, batch_size: int = 4, temperature: float = 1.0):
    """
    Get batch predictions from the batch processing server.
    
    Args:
        instruction (str): The instruction for the robot
        image_path (str): Path to the input image
        batch_size (int, optional): Size of the batch. Defaults to 4.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
    
    Returns:
        numpy.ndarray: Array of predicted actions
    """
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Prepare the payload
    payload = {
        "instruction": instruction,
        "image_path": image_path,
        "batch_size": batch_size,
        "temperature": temperature
    }
    
    # Send request to server
    response = requests.post(
        "http://127.0.0.1:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        raise Exception(f"Error from server: {response.text}")
    
    response_data = json.loads(response.text)
    return np.array(response_data["output_ids"]), np.array(response_data["actions"])

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
    temperature: float = 0.7
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
            all_action_preds = []
            all_outputs = []

            batch_size = cfg.num_generations
            instruction = batch['lang'][0].lower()

            batch['img'][0].save(f"images/observation.jpg")
            image_path = "/root/vla-cot/images/observation.jpg"

            output_ids, actions = get_batch_actions(
                instruction=instruction,
                image_path=image_path,
                batch_size=1,
                temperature=0
            )
            print(output_ids)
            
            # for _ in range(cfg.num_generations):
            #     with torch.autocast("cuda", dtype=torch.bfloat16):
            #         output: CausalLMOutputWithPast = vla(
            #             input_ids=batch["input_ids"].to(device_id),
            #             attention_mask=batch["attention_mask"].to(device_id),
            #             pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            #             labels=batch["labels"],
            #         )
                
            #     action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            #     action_preds = action_logits.argmax(dim=2)
            #     all_action_preds.append(action_preds)
            #     all_outputs.append(output)

            # Stack predictions from multiple generations
            action_preds = torch.stack(all_action_preds, dim=1)  # [batch_size, num_generations, seq_len]
            batch_size = action_preds.size(0)
            
            # Reshape to [batch_size * num_generations, seq_len]
            action_preds = action_preds.view(-1, action_preds.size(-1))

            # Get model and reference model logprobs
            def get_per_token_logps(model, inputs):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                        labels=inputs["labels"]
                    )
                
                logits = output.logits[:, model.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get the relevant action predictions for this batch
                    # For reference model, we need to repeat the log_probs to match all generations
                log_probs = log_probs.repeat(cfg.num_generations, 1, 1)
                relevant_preds = action_preds  # Use all predictions
        
                # Gather log probs for the predicted actions
                per_token_logps = torch.gather(
                    log_probs,
                    dim=-1,
                    index=relevant_preds.unsqueeze(-1)
                ).squeeze(-1)
                
                return per_token_logps
                
            # Compute policy and reference logprobs
            per_token_logps = get_per_token_logps(vla, batch)
            ref_per_token_logps = get_per_token_logps(ref_vla, batch)

            # Compute KL divergence
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                          (ref_per_token_logps - per_token_logps) - 1

            # Create completion mask based on action_gt
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            mask = mask.repeat(cfg.num_generations, 1)  # Repeat for all generations

            # Generate random binary rewards for testing
            rewards = torch.randint(0, 2, (batch_size * cfg.num_generations,), device=device_id).float()

            # Compute advantages
            mean_grouped_rewards = rewards.view(-1, cfg.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, cfg.num_generations).std(dim=1)
            
            # Repeat for each generation
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(cfg.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(cfg.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

            # Compute GRPO loss
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - cfg.beta * per_token_kl)
            loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()
            
            # Compute accuracy for logging
            correct_preds = (action_preds == action_gt.repeat(cfg.num_generations, 1)) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Scale loss and backward pass
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            print(normalized_loss)

            # Store metrics
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