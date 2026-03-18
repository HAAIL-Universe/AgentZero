---
topic: Silent 2K Token Truncation in Local Inference
status: implemented
priority: medium
estimated_complexity: small
researched_at: 2026-03-19T00:10:00Z
---

# Silent 2K Token Truncation in Local Inference

## Problem Statement

The local (non-vLLM) inference path in `agent_zero_inference.py` silently truncates input
to 2,048 tokens despite the model supporting 40,960 tokens (config.model_context_limit).
This means that in local inference mode, ~95% of the model's context window is unused,
and long conversations are silently truncated with no warning or error. The user receives
a response based on incomplete context, which can cause hallucinations, loss of
conversation coherence, and missed safety-critical information.

The vLLM streaming path (`_vllm_stream`) does NOT have this problem -- it passes
`max_tokens` as an output cap, not an input cap. The issue is confined to the local
HuggingFace tokenizer calls.

## Current State in Agent Zero

### agent_zero_inference.py -- Three truncation points

**Line 257 (generate method):**
```python
inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
```

**Line 399 (generate_stream method):**
```python
inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
```

Both lines use `max_length=2048` which is a hardcoded input truncation limit. The tokenizer
silently drops all tokens beyond position 2048 from the prompt. No warning is logged.

**Lines 243, 283, 382 (function signatures):**
```python
def generate(self, conversation: list, max_tokens: int = 2048, ...):
def generate_with_tools(self, ..., max_tokens: int = 2048, ...):
def generate_stream(self, conversation: list, max_tokens: int = 2048, ...):
```

`max_tokens` here controls max_new_tokens for generation output, which is correct. But
the same value (2048) is also hardcoded as the input truncation limit -- these should be
independent parameters.

### Config (config.py:212-214)
```python
model_context_limit: int = Field(
    default=40960, ge=4096, le=131072,
    description="Model context window size in tokens (Qwen3-235B max: 40960)"
)
```

The config knows the real limit (40960) but the inference module ignores it for local mode.

### When this triggers

Local inference is used when `VLLM_API_URL` is not set (development, testing, or fallback
mode). In production on RunPod, vLLM is used and this bug doesn't manifest. But any
developer or tester running Agent Zero locally will silently lose context.

## Industry Standard / Research Findings

### 1. Silent Truncation is a Known Production Anti-Pattern

The Ollama project (issue #14259) documented that "chat history and embedding truncation
happens silently with no user-visible indication." The community response was to add
explicit logging when truncation occurs. Agent-Zero (issue #1162) similarly flagged
"silent output truncation in LLM calls" as a critical bug because "the model still
produces confident answers" with incomplete context.

- Ollama silent truncation: https://github.com/ollama/ollama/issues/14259
- Agent-Zero silent truncation: https://github.com/agent0ai/agent-zero/issues/1162

### 2. Context Window Management Best Practices

Redis Engineering (2026) recommends: "Implement token counters and overflow alerts to
prevent models from losing vital data midstream." The recommended pattern is to log a
warning when input exceeds the context limit, and to use the full model context limit
minus the generation budget as the input cap.

- Redis context overflow guide: https://redis.io/blog/context-window-overflow/
- Atlan LLM context limitations: https://atlan.com/know/llm-context-window-limitations/

### 3. HuggingFace Tokenizer Truncation

The HuggingFace tokenizer `max_length` parameter should be set to the model's actual
context limit, not a hardcoded value. When `max_length` is left unset (None), the
tokenizer uses the model's configured maximum length automatically. For local inference,
the correct pattern is: `max_length = model_context_limit - max_new_tokens` to reserve
space for generation while using as much context as possible.

- HuggingFace padding/truncation docs: https://huggingface.co/docs/transformers/en/pad_truncation
- HuggingFace tokenizer docs: https://huggingface.co/docs/transformers/main_classes/tokenizer
- Community discussion: https://discuss.huggingface.co/t/tokenizer-truncation/18494

## Proposed Implementation

### Change 1: Use Config-Driven Context Limit

**File:** `agent_zero/agent_zero_inference.py`

Import config and compute the correct input limit:

```python
from config import config as _cfg
```

### Change 2: Fix generate() (line 257)

Replace:
```python
inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
```

With:
```python
input_limit = _cfg.model_context_limit - max_tokens  # reserve space for generation
inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_limit)
if inputs["input_ids"].shape[1] >= input_limit:
    _log.warning(
        "Input truncated from estimated %d tokens to %d (context_limit=%d, max_new_tokens=%d)",
        len(self.tokenizer.encode(prompt)),
        input_limit,
        _cfg.model_context_limit,
        max_tokens,
    )
```

### Change 3: Fix generate_stream() (line 399)

Same fix as Change 2 -- replace the hardcoded `max_length=2048`.

### Change 4: Add Logging Import

Ensure the logger is available at module level:

```python
from logging_config import get_logger
_log = get_logger("inference")
```

(If `logging_config` is not importable in the local inference context, fall back to
`import logging; _log = logging.getLogger("agent_zero.inference")`)

### Change 5: Add Config Fields for Inference Limits

**File:** `agent_zero/config.py` -- add to Agent ZeroConfig:

```python
inference_max_new_tokens: int = Field(
    default=2048, ge=256, le=8192,
    description="Default max new tokens for generation (output cap)"
)
```

Then reference `_cfg.inference_max_new_tokens` as the default value in generate() instead
of hardcoding 2048.

## Test Specifications

### test_agent_zero_inference_truncation.py

```python
def test_local_tokenizer_uses_config_context_limit():
    """Local inference path uses model_context_limit, not hardcoded 2048."""
    # Mock tokenizer to capture max_length argument
    # Verify max_length == config.model_context_limit - max_tokens

def test_local_tokenizer_reserves_space_for_generation():
    """Input limit = context_limit - max_new_tokens."""
    # With context_limit=40960, max_tokens=2048: input_limit should be 38912

def test_truncation_logs_warning():
    """When input exceeds limit, a warning is logged."""
    # Provide a prompt longer than input_limit
    # Capture log output, verify warning contains "truncated"

def test_vllm_path_unaffected():
    """vLLM streaming path doesn't use tokenizer truncation."""
    # Verify _vllm_stream passes max_tokens to API, not tokenizer

def test_generate_with_tools_uses_config_limit():
    """generate_with_tools also respects config context limit."""

def test_default_max_tokens_from_config():
    """Default max_tokens parameter comes from config, not hardcoded 2048."""
```

## Estimated Impact

- **Context utilization:** Local inference goes from using 5% of context window (2048/40960)
  to using ~95% (38912/40960 with default generation budget).

- **Conversation quality:** Long conversations no longer silently lose history in local mode.
  Developers and testers see the same behavior locally as in production.

- **Observability:** Truncation events are logged with token counts, making context pressure
  visible in logs rather than silently degrading response quality.

- **Backward compatibility:** The vLLM production path is unchanged. Only the local
  HuggingFace inference path is fixed. The default generation budget (2048 tokens)
  remains the same.

## Related Papers
- `research/papers/token_estimation_accuracy.md` -- covers context manager's 4 chars/token
  heuristic (complementary: context estimation feeds into inference truncation decisions)
