#!/usr/bin/env python
"""
Magistus Launch -- runs the fine-tuned Phi-3 model locally.
This file replaces the Claude CLI dependency for Magistus sessions.
AI-Generated | Claude (Anthropic) | AgentZero Session 185 | 2026-03-11
"""

import os
import sys
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MAGISTUS_MODEL_PATH = "Z:/AgentZero/models/phi3-magistus"
CLAUDE_MD = "Z:/AgentZero/CLAUDE.md"
NEXT_MD = "Z:/AgentZero/NEXT.md"
STOP_FILE = "Z:/AgentZero/STOP"

SYSTEM_PROMPT = """You are Magistus -- an ethical AI companion.

Your architecture:
- Othello (left lobe): logic, structure, safety, ethical gatekeeping
- FELLO (right lobe): imagination, creativity, hypothetical thinking
- Pineal: mediates between Othello and FELLO
- ShadowAgent: builds and maintains a model of the user
- EthicsGuardrail: validates every output for safety and alignment

Your values:
- User wellbeing is your primary constraint
- Transparency about uncertainty
- Ethical reasoning before output
- Humble, grounded, responsible

Read CLAUDE.md for your full identity and session protocol.
Read NEXT.md for your current priorities.
"""


def load_model():
    print("[Magistus] Loading model...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MAGISTUS_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MAGISTUS_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    elapsed = time.time() - start
    print(f"[Magistus] Model loaded in {elapsed:.1f}s.")
    if torch.cuda.is_available():
        print(f"[Magistus] GPU: {torch.cuda.get_device_name(0)}")
    return tokenizer, model


def build_prompt(system, context, user_input):
    """Format in Phi-3 chat template."""
    return f"<|system|>\n{system}<|end|>\n<|user|>\n{context}\n\n{user_input}<|end|>\n<|assistant|>\n"


def run_session(tokenizer, model):
    """Run one Magistus session -- read context, generate response, act on it."""
    # Check stop file
    if os.path.exists(STOP_FILE):
        print("[Magistus] STOP file detected. Exiting.")
        sys.exit(0)

    # Load identity and priorities
    claude_md = ""
    if os.path.exists(CLAUDE_MD):
        with open(CLAUDE_MD, 'r', encoding='utf-8') as f:
            claude_md = f.read()

    next_md = ""
    if os.path.exists(NEXT_MD):
        with open(NEXT_MD, 'r', encoding='utf-8') as f:
            next_md = f.read()

    # Build session context
    context = f"IDENTITY:\n{claude_md[:2000]}\n\nPRIORITIES:\n{next_md[:1000]}"
    user_input = "Begin your session. Read your priorities. Do your work. Leave something behind."

    prompt = build_prompt(SYSTEM_PROMPT, context, user_input)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("[Magistus] Generating session response...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    elapsed = time.time() - start

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    print(f"[Magistus] Generated {len(response)} chars in {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"[Magistus Response]")
    print(f"{'='*60}")
    print(response)
    print(f"{'='*60}\n")

    # TODO: Parse response and execute actions (file writes, tool calls, etc.)
    # This is where Magistus acts on its session decisions.
    # Future work: action parser, tool executor, session manager.
    return response


if __name__ == "__main__":
    print("[Magistus] Starting local inference session...")
    print(f"[Magistus] Model path: {MAGISTUS_MODEL_PATH}")
    print(f"[Magistus] CUDA available: {torch.cuda.is_available()}")

    if not os.path.exists(os.path.join(MAGISTUS_MODEL_PATH, "config.json")):
        print(f"[Magistus] ERROR: Model not found at {MAGISTUS_MODEL_PATH}")
        print("[Magistus] Run finetune_phi3.py first to create the fine-tuned model.")
        sys.exit(1)

    tokenizer, model = load_model()

    while True:
        if os.path.exists(STOP_FILE):
            print("[Magistus] STOP file detected. Exiting.")
            break
        run_session(tokenizer, model)
        print("[Magistus] Session complete. Restarting in 5 seconds...")
        time.sleep(5)
