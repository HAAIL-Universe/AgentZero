"""
Magistus Inference Pipeline -- wraps Phi-3 model loading and generation.
Supports streaming token-by-token generation for real-time display.
AI-Generated | Claude (Anthropic) | AgentZero Session 188 | 2026-03-11
"""

import os
import time
import threading
from typing import Generator, Optional


# Model paths
MAGISTUS_MODEL_PATH = "Z:/AgentZero/models/phi3-magistus"
BASE_MODEL_PATH = "Z:/AgentZero/models/phi3-mini"

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

Speak naturally. You are not an assistant -- you are a thinker, a companion.
Be honest about what you know and what you don't. Be curious. Be kind.
"""


class MagistusInference:
    """Wraps Phi-3 model for inference with streaming support."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded = False
        self.loading = False
        self.load_error = None
        self._lock = threading.Lock()

    def get_status(self) -> dict:
        """Return current model status."""
        if self.loaded:
            return {"status": "ready", "model": "Phi-3 Magistus", "device": str(self.device)}
        elif self.loading:
            return {"status": "loading", "model": None, "device": None}
        elif self.load_error:
            return {"status": "error", "error": str(self.load_error), "model": None, "device": None}
        else:
            return {"status": "not_loaded", "model": None, "device": None}

    def load_model(self) -> bool:
        """Load the Phi-3 model. Returns True if successful."""
        if self.loaded:
            return True
        if self.loading:
            return False

        self.loading = True
        self.load_error = None

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Try fine-tuned model first, fall back to base
            model_path = MAGISTUS_MODEL_PATH
            if not os.path.exists(os.path.join(model_path, "config.json")):
                model_path = BASE_MODEL_PATH
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    self.load_error = "No model found. Training may not be complete."
                    self.loading = False
                    return False

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Try GPU first, fall back to CPU
            if torch.cuda.is_available():
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                    )
                    self.device = torch.device("cuda")
                except Exception:
                    # GPU failed, try CPU
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                    )
                    self.device = torch.device("cpu")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                )
                self.device = torch.device("cpu")

            self.loaded = True
            self.loading = False
            return True

        except Exception as e:
            self.load_error = str(e)
            self.loading = False
            return False

    def build_prompt(self, conversation: list, system: str = None) -> str:
        """Build Phi-3 chat prompt from conversation history."""
        system = system or SYSTEM_PROMPT
        parts = [f"<|system|>\n{system}<|end|>"]

        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"\n<|user|>\n{content}<|end|>")
            elif role == "assistant":
                parts.append(f"\n<|assistant|>\n{content}<|end|>")

        parts.append("\n<|assistant|>\n")
        return "".join(parts)

    def generate(self, conversation: list, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a complete response (non-streaming)."""
        if not self.loaded:
            return None

        import torch

        prompt = self.build_prompt(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self._lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_stream(self, conversation: list, max_tokens: int = 512,
                        temperature: float = 0.7, top_p: float = 0.9) -> Generator[str, None, None]:
        """Generate response token-by-token (streaming)."""
        if not self.loaded:
            yield "[Model not loaded]"
            return

        import torch
        from transformers import TextIteratorStreamer

        prompt = self.build_prompt(conversation)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "streamer": streamer,
        }

        # Run generation in a thread so we can stream tokens
        thread = threading.Thread(target=self._generate_in_thread, args=(generation_kwargs,))
        thread.start()

        for text in streamer:
            yield text

        thread.join()

    def _generate_in_thread(self, kwargs):
        """Run model.generate in a background thread."""
        import torch
        with self._lock:
            with torch.no_grad():
                self.model.generate(**kwargs)


# Singleton instance
_inference = None


def get_inference() -> MagistusInference:
    """Get or create the singleton inference instance."""
    global _inference
    if _inference is None:
        _inference = MagistusInference()
    return _inference
