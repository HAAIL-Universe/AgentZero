#!/usr/bin/env python
"""
Phi-3 Mini fine-tuning on Magistus corpus using LoRA.
Custom training loop -- avoids HF Trainer device placement issues.
AI-Generated | Claude (Anthropic) | AgentZero Session 185 | 2026-03-11
"""

import os
import sys
import json
import time
import math
import torch

# Force line-buffered stdout so progress appears in real time
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "Z:/AgentZero/models/phi3-mini"
CORPUS_PATH = "Z:/AgentZero/data/magistus_training.jsonl"
OUTPUT_PATH = "Z:/AgentZero/models/phi3-magistus-lora"
MERGED_PATH = "Z:/AgentZero/models/phi3-magistus"

# Hyperparameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 512
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.1
LOG_EVERY = 1  # Log every N optimizer steps


def load_tokenizer_and_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model on GPU with device_map='auto'...")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")

    # Load on CPU in float16 (~7.6GB). No device_map to avoid meta tensor issues.
    # CPU training is slow but works. GPU can't hold the full 7.6GB model.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print("  Model loaded on CPU (float16, ~7.6GB). No meta params.")
    device = torch.device("cpu")

    return tokenizer, model, device


def apply_lora(model):
    """Apply LoRA adapters -- only these tiny layers will train."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_examples(tokenizer):
    """Load and tokenize training examples."""
    examples = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                tok = tokenizer(
                    item['text'],
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    padding=False,
                    return_tensors="pt",
                )
                examples.append({
                    'input_ids': tok['input_ids'].squeeze(0),
                    'attention_mask': tok['attention_mask'].squeeze(0),
                })
    print(f"Loaded {len(examples)} training examples")
    return examples


def cosine_lr(step, total_steps, warmup_steps, lr):
    """Cosine learning rate with linear warmup."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(model, tokenizer, examples, device):
    """Custom training loop with gradient accumulation."""
    total_opt_steps = (len(examples) * NUM_EPOCHS) // GRAD_ACCUM_STEPS
    warmup_steps = int(total_opt_steps * WARMUP_RATIO)

    # Only optimize LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    print(f"\nStarting fine-tuning...")
    print(f"  Device: {device}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Examples: {len(examples)}")
    print(f"  Grad accum steps: {GRAD_ACCUM_STEPS}")
    print(f"  Total optimizer steps: {total_opt_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {LEARNING_RATE}")

    model.train()
    global_step = 0
    opt_step = 0
    accum_loss = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Shuffle examples each epoch
        import random
        indices = list(range(len(examples)))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            ex = examples[idx]
            input_ids = ex['input_ids'].unsqueeze(0).to(device)
            attention_mask = ex['attention_mask'].unsqueeze(0).to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            global_step += 1

            if global_step % GRAD_ACCUM_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                # LR schedule
                current_lr = cosine_lr(opt_step, total_opt_steps, warmup_steps, LEARNING_RATE)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / max(opt_step, 1)) * (total_opt_steps - opt_step)
                    print(f"  Step {opt_step}/{total_opt_steps} | "
                          f"Loss: {accum_loss:.4f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Elapsed: {elapsed:.0f}s | "
                          f"ETA: {eta:.0f}s")
                accum_loss = 0.0

        print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} complete.")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save LoRA adapters
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"LoRA adapters saved to {OUTPUT_PATH}")


def merge_and_save(model, tokenizer):
    """Merge LoRA adapters into base model for standalone use."""
    print("\nMerging LoRA adapters into base model...")
    merged = model.merge_and_unload()
    os.makedirs(MERGED_PATH, exist_ok=True)
    merged.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)
    print(f"Merged model saved to {MERGED_PATH}")
    print("Fine-tuning complete. The Magistus model is ready.")


if __name__ == "__main__":
    tokenizer, model, device = load_tokenizer_and_model()
    model = apply_lora(model)
    examples = load_examples(tokenizer)
    train(model, tokenizer, examples, device)
    merge_and_save(model, tokenizer)
