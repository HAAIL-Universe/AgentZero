---
topic: Domain-Neutral Prompt Normalization
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T23:45:00Z
---

# Domain-Neutral Prompt Normalization

## Problem Statement

Agent Zero has hardcoded domain-specific strings scattered across 7+ modules. These strings
couple the system to a single test scenario ("ask for a raise" / "manager is unpredictable")
and make the system fragile: any user input that doesn't match these exact phrases falls
through to generic handling. The hardcoded strings exist in:

1. **agent_zero_server.py:4540-4585** -- `_normalize_recap_goal()`, `_normalize_recap_constraints()`,
   `_extract_recap_context()`, `_friendly_path_name()` all contain if/elif chains matching
   specific career phrases
2. **reasoning_framework.py:33-34, 121-124, 635** -- `STRATEGIC_KEYWORDS` includes "ask for a
   raise", "want a raise", "career move"; `STRONG_STRATEGIC` duplicates these; `_classify_domain()`
   has hardcoded keyword dicts for 5 domains
3. **user_model.py:462-473** -- `_detect_cluster()` matches "ask for a raise", "manager",
   returns hardcoded cluster names
4. **behavioural_shadow.py:303** -- topic classification keywords
5. **context_manager.py:240** -- topic detection keywords
6. **growth_companion.py:139** -- topic keywords
7. **outcome_patterns.py:38** -- topic extraction keywords

The same 5-domain keyword dictionary (career, productivity, health, relationships, technology)
is duplicated across at least 5 files with slight variations in keyword lists. There is no
single source of truth. Adding a new domain requires editing 5+ files.

## Current State in Agent Zero

### Duplicated Topic Keyword Dictionaries

**reasoning_framework.py:634-640:**
```python
keywords = {
    "career": ["job", "manager", "promotion", "salary", "raise", "work", "career", "role", "quit", "resign", "full time"],
    "productivity": ["habit", "focus", "procrast", "routine", "stuck", "productive"],
    "health": ["sleep", "exercise", "diet", "energy", "health"],
    "relationships": ["partner", "family", "friend", "relationship"],
    "technology": ["code", "software", "program", "build"],
}
```

**context_manager.py:240:**
```python
"career": ["job", "career", "work", "salary", "raise", "promotion", "boss", "manager", "interview"],
```

**behavioural_shadow.py:303:**
```python
"career": ["career", "job", "work", "promotion", "interview", "salary"],
```

**outcome_patterns.py:38:**
```python
"career": ["career", "job", "work", "promotion", "interview", "salary", "manager", "raise", "role"],
```

Note the inconsistency: "boss" appears only in context_manager; "manager" appears in
reasoning_framework and outcome_patterns but not behavioural_shadow; "interview" is in
3 of 4 but not reasoning_framework. This means the same user input may be classified
differently depending on which module processes it.

### Hardcoded Recap Normalization (agent_zero_server.py:4537-4563)

```python
def _normalize_recap_goal(goal: str) -> str:
    lower = text.lower()
    if "ask for a raise" in lower:
        return "you were thinking about asking for a raise"
    if "focus on ai full time" in lower or "leave my current role" in lower:
        return "you were considering moving into AI full time"
    # ...

def _normalize_recap_constraints(constraints: str) -> str:
    if "manager is unpredictable" in lower and "rejection" in lower:
        return "you were worried about rejection, especially because your manager felt unpredictable"
    # ...
```

These functions only handle 2 specific scenarios. Any other goal or constraint passes
through with minimal normalization (just I'm -> you were replacement).

### Hardcoded Path Names (agent_zero_server.py:4575-4585)

```python
mapping = {
    "build case first": "building your case first",
    "low-risk probe": "testing the waters before making a direct ask",
    "direct move": "making a direct ask",
    # ...
}
```

Only 6 paths are mapped. New paths added by the LLM won't be normalized.

## Industry Standard / Research Findings

### 1. Configurable Topic Taxonomies

Modern chatbot systems externalize topic/domain configurations into data files rather than
embedding them in code. This is the standard pattern in both rule-based and ML-based
systems. FlowHunt and Aisera both implement configurable domain taxonomies that can be
updated without code changes.

- FlowHunt domain classification: https://www.flowhunt.io/faq/chatbot-ai-domain-classification/
- Aisera hybrid conversation design: https://docs.aisera.com/aisera-platform/crafting-the-conversation/conversation-design-icm-llm-and-hybrid

### 2. LLM-Based Classification for Open-Domain Systems

For behavioral coaching chatbots, LLM-based topic classification outperforms keyword
matching because it handles paraphrases, implicit intent, and cross-domain inputs. The
ProactiMate system (2025) uses a Chain of Models approach where the LLM itself classifies
user intent using few-shot prompting rather than keyword lists. MIcha (2025) uses GPT-4
with Motivational Interviewing principles for intent detection.

- ProactiMate: behavioral coaching chatbot with LLM classification (preprint 2025)
- MIcha: LLM-based chatbot for cognitive restructuring, https://arxiv.org/html/2501.15599v1
- Few-shot prompting guide: https://www.promptingguide.ai/techniques/fewshot

### 3. Data-Driven Keyword Dictionaries

When keyword matching is appropriate (for speed, determinism, or as a pre-filter), the
standard practice is to load keyword dictionaries from a configuration file (JSON/YAML)
rather than hardcoding them. This allows: (a) a single source of truth for all modules,
(b) updates without code changes, (c) per-deployment customization, (d) A/B testing of
different taxonomies. Topic tagging models typically use a predefined taxonomy loaded
from configuration.

- Topic tagging taxonomy approaches: https://medium.com/nlplanet/two-minutes-nlp-basic-taxonomy-of-topic-tagging-models-and-elementary-use-cases-763113d0eba
- NLP taxonomy classifier: https://huggingface.co/TimSchopf/nlp_taxonomy_classifier
- Keyword-based chatbot patterns: https://www.analyticsvidhya.com/blog/2023/02/how-to-build-a-chatbot-using-natural-language-processing/

### 4. Hybrid Approach: Keywords + LLM Fallback

The recommended pattern for Agent Zero (which already uses an LLM) is:
1. **Fast path:** keyword dictionary lookup (loaded from config) for known topics
2. **Slow path:** LLM-based classification when keywords don't match
3. **Learning loop:** log unmatched inputs, periodically review and add keywords

This matches Aisera's ICM+LLM hybrid pattern and is standard in production chatbot systems.

## Proposed Implementation

### Change 1: Create Topic Taxonomy Configuration File

**New file:** `agent_zero/data/topic_taxonomy.json`

```json
{
  "version": 1,
  "domains": {
    "career": {
      "keywords": ["job", "manager", "boss", "promotion", "salary", "raise",
                    "work", "career", "role", "quit", "resign", "full time",
                    "interview", "office", "coworker", "colleague"],
      "strong_signals": ["ask for a raise", "want a raise", "career move",
                         "quit my job", "leave my job"],
      "description": "Career decisions, workplace dynamics, job transitions"
    },
    "productivity": {
      "keywords": ["habit", "focus", "procrast", "routine", "stuck",
                    "productive", "discipline", "motivation", "procrastinate",
                    "distract", "organize", "schedule", "time management"],
      "strong_signals": ["struggling with productivity", "can't focus",
                         "keep procrastinating"],
      "description": "Habits, focus, routines, time management"
    },
    "health": {
      "keywords": ["sleep", "exercise", "diet", "energy", "health", "fitness",
                    "weight", "anxiety", "stress", "meditation", "mindful",
                    "burnout", "exhaustion"],
      "strong_signals": ["burning out", "can't sleep", "health is suffering"],
      "description": "Physical and mental health, wellness, stress"
    },
    "relationships": {
      "keywords": ["partner", "family", "friend", "relationship", "parent",
                    "child", "spouse", "dating", "marriage", "conflict",
                    "communication", "trust", "boundary"],
      "strong_signals": ["relationship is struggling", "fight with",
                         "breaking up"],
      "description": "Interpersonal relationships, family, social"
    },
    "personal_growth": {
      "keywords": ["goal", "purpose", "meaning", "values", "identity",
                    "confidence", "self-esteem", "fear", "courage", "growth"],
      "strong_signals": ["don't know who I am", "lost my purpose",
                         "afraid of failure"],
      "description": "Self-discovery, values, identity, confidence"
    },
    "financial": {
      "keywords": ["money", "budget", "debt", "savings", "invest",
                    "financial", "afford", "expense", "income"],
      "strong_signals": ["financially free", "drowning in debt"],
      "description": "Financial planning, money management"
    },
    "technology": {
      "keywords": ["code", "software", "program", "build", "tech", "AI",
                    "machine learning", "startup", "project"],
      "strong_signals": [],
      "description": "Technology, coding, technical projects"
    }
  },
  "recap_normalization": {
    "goal_patterns": {
      "I want to": "you wanted to",
      "I need to": "you needed to",
      "I'm trying to": "you were trying to",
      "I'm going to": "you were going to",
      "I am": "you were",
      "I'm": "you were"
    },
    "constraint_patterns": {
      "I'm worried": "you were worried",
      "I'm scared": "you were scared",
      "I'm afraid": "you were afraid",
      "I am worried": "you were worried"
    }
  },
  "path_friendly_names": {
    "build case first": "building your case first",
    "low-risk probe": "testing the waters before making a direct ask",
    "direct move": "making a direct ask",
    "accountability loop": "using an accountability loop",
    "reduce scope": "reducing the scope",
    "remove friction": "removing friction first"
  }
}
```

### Change 2: Create Topic Taxonomy Loader Module

**New file:** `agent_zero/topic_taxonomy.py` (~60 lines)

```python
"""Centralized topic taxonomy -- single source of truth for domain keywords.

All modules that need topic classification import from here instead of
maintaining their own hardcoded keyword dictionaries.
"""

import json
import os
from typing import Optional

_TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "data", "topic_taxonomy.json"
)

_taxonomy: dict = {}


def _load_taxonomy() -> dict:
    """Load taxonomy from JSON file. Cached after first load."""
    global _taxonomy
    if _taxonomy:
        return _taxonomy
    with open(_TAXONOMY_PATH, "r", encoding="utf-8") as f:
        _taxonomy = json.load(f)
    return _taxonomy


def get_domain_keywords() -> dict[str, list[str]]:
    """Return {domain: [keywords]} dict for all domains."""
    tax = _load_taxonomy()
    return {
        domain: info["keywords"]
        for domain, info in tax["domains"].items()
    }


def get_strong_signals() -> dict[str, list[str]]:
    """Return {domain: [strong_signal_phrases]} for strategic routing."""
    tax = _load_taxonomy()
    return {
        domain: info.get("strong_signals", [])
        for domain, info in tax["domains"].items()
    }


def classify_domain(text: str, shadow: Optional[dict] = None) -> str:
    """Classify text into a domain using keyword matching.

    Returns domain name or 'general' if no match.
    """
    lower = text.lower()
    keywords = get_domain_keywords()
    for domain, terms in keywords.items():
        if any(term in lower for term in terms):
            return domain
    # Fallback to shadow priorities
    if shadow:
        priorities = shadow.get("revealed_priorities", [])
        if priorities:
            return priorities[0].get("topic", "general")
    return "general"


def get_recap_goal_patterns() -> dict[str, str]:
    """Return {pattern: replacement} for goal recap normalization."""
    tax = _load_taxonomy()
    return tax.get("recap_normalization", {}).get("goal_patterns", {})


def get_recap_constraint_patterns() -> dict[str, str]:
    """Return {pattern: replacement} for constraint recap normalization."""
    tax = _load_taxonomy()
    return tax.get("recap_normalization", {}).get("constraint_patterns", {})


def get_path_friendly_names() -> dict[str, str]:
    """Return {path_name: friendly_description} mapping."""
    tax = _load_taxonomy()
    return tax.get("path_friendly_names", {})
```

### Change 3: Update All Consumer Modules

**File: reasoning_framework.py** (lines 634-648)

Replace the hardcoded `_classify_domain()` with:
```python
from topic_taxonomy import classify_domain

# Remove the local keywords dict and for loop
# Replace with:
def _classify_domain(lower: str, shadow: dict) -> str:
    return classify_domain(lower, shadow)
```

Also update `STRONG_STRATEGIC` (line 120-124) to load from taxonomy:
```python
from topic_taxonomy import get_strong_signals

# Build STRONG_STRATEGIC from all domains' strong signals
_all_strong = set()
for signals in get_strong_signals().values():
    _all_strong.update(signals)
STRONG_STRATEGIC = _all_strong
```

**File: context_manager.py** (line 240)

Replace the local topic keywords dict with:
```python
from topic_taxonomy import get_domain_keywords
# Use get_domain_keywords() instead of hardcoded dict
```

**File: behavioural_shadow.py** (line 303)

Replace the local topic keywords with:
```python
from topic_taxonomy import get_domain_keywords
```

**File: outcome_patterns.py** (line 38)

Replace the local topic keywords with:
```python
from topic_taxonomy import get_domain_keywords
```

**File: growth_companion.py** (line 139)

Replace the local topic keywords with:
```python
from topic_taxonomy import get_domain_keywords
```

**File: agent_zero_server.py** (lines 4537-4585)

Replace `_normalize_recap_goal()`, `_normalize_recap_constraints()`, and
`_friendly_path_name()` with data-driven versions:

```python
from topic_taxonomy import (
    get_recap_goal_patterns,
    get_recap_constraint_patterns,
    get_path_friendly_names,
)

def _normalize_recap_goal(goal: str) -> str:
    text = str(goal or "").strip().strip(".")
    if text.startswith("an improved next state"):
        return ""
    for pattern, replacement in get_recap_goal_patterns().items():
        text = text.replace(pattern, replacement)
    return text

def _normalize_recap_constraints(constraints: str) -> str:
    text = str(constraints or "").strip().strip(".")
    for pattern, replacement in get_recap_constraint_patterns().items():
        text = text.replace(pattern, replacement)
    return text

def _friendly_path_name(path: str) -> str:
    lower = str(path or "").strip().lower()
    mapping = get_path_friendly_names()
    return mapping.get(lower, str(path or "").strip())
```

**File: user_model.py** (lines 462-473)

Remove the hardcoded cluster detection for "ask for a raise" and "manager". Instead,
use the taxonomy's strong signals to detect clusters generically:

```python
from topic_taxonomy import get_strong_signals, classify_domain

def _detect_cluster(text: str) -> dict:
    lower = text.lower()
    domain = classify_domain(lower)
    signals = get_strong_signals()
    matched = [s for s in signals.get(domain, []) if s in lower]
    # Return generic cluster info based on detected domain
    # ...
```

### Change 4: Add Config Field for Taxonomy Path

**File: config.py** -- add to Agent ZeroConfig:

```python
topic_taxonomy_path: str = Field(
    default="data/topic_taxonomy.json",
    description="Path to topic taxonomy JSON (relative to agent_zero/)"
)
```

This allows per-deployment customization of the topic taxonomy.

## Test Specifications

### test_topic_taxonomy.py

```python
def test_taxonomy_loads_valid_json():
    """topic_taxonomy.json loads without error and has 'domains' key."""

def test_all_domains_have_keywords():
    """Every domain entry has a non-empty 'keywords' list."""

def test_get_domain_keywords_returns_all_domains():
    """get_domain_keywords() returns dict with all domains from JSON."""

def test_classify_domain_career():
    """'I want to ask for a raise' -> 'career'."""

def test_classify_domain_health():
    """'I can't sleep at night' -> 'health'."""

def test_classify_domain_general_fallback():
    """Unrecognized input returns 'general'."""

def test_classify_domain_shadow_fallback():
    """When no keywords match, falls back to shadow priorities."""

def test_get_strong_signals_returns_phrases():
    """Strong signals include multi-word phrases like 'ask for a raise'."""

def test_recap_goal_normalization():
    """'I want to get a promotion' -> 'you wanted to get a promotion'."""

def test_recap_constraint_normalization():
    """'I'm worried about rejection' -> 'you were worried about rejection'."""

def test_friendly_path_names():
    """'build case first' -> 'building your case first'."""

def test_friendly_path_unknown():
    """Unknown path name returned as-is."""

# --- Consistency Tests ---

def test_all_consumer_modules_use_taxonomy():
    """Verify no hardcoded keyword dicts remain in consumer modules.
    Grep for the old pattern and assert zero matches."""

def test_career_keywords_superset():
    """Taxonomy career keywords include all previously-used keywords
    from all modules (union of all 4 previous dicts)."""

# --- Integration Tests ---

def test_reasoning_framework_uses_taxonomy():
    """route_turn() uses taxonomy for domain classification."""

def test_context_manager_uses_taxonomy():
    """_classify_topics() uses taxonomy keywords."""
```

## Estimated Impact

- **Maintainability:** Adding a new domain (e.g., "education", "creativity") requires
  editing one JSON file instead of 5+ Python files. Reduces lines of hardcoded strings
  from ~80 across 7 files to 0 (all in JSON config).

- **Consistency:** All modules use the same keyword lists, eliminating the current
  inconsistency where the same input gets classified differently by different modules.

- **Extensibility:** Per-deployment customization is possible by swapping the JSON file.
  A/B testing of different taxonomies is trivial.

- **Backward Compatibility:** All existing keyword matches are preserved (the JSON
  contains the union of all current keyword lists). No behavioral changes for existing
  inputs.

## Related Papers
- `research/papers/adaptive_voice_personality.md` -- voice personality adapts per domain
  (will benefit from consistent domain classification)
- `research/papers/cognitive_runtime_configuration.md` -- configuration extraction pattern
  (this paper extends the same principle to topic classification)
