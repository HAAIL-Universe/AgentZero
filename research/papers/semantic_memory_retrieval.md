---
topic: Semantic Memory Retrieval Ranking
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Semantic Memory Retrieval Ranking

## Problem Statement

Agent Zero's memory retrieval system uses keyword overlap scoring to rank memories for context injection. This approach fails on synonym matching ("exercise" vs "workout"), conceptual similarity ("feeling overwhelmed" vs "too much on my plate"), and produces false positives on common terms. As the memory store grows, retrieval quality degrades because the linear scan with term matching has no term importance weighting, no document frequency normalization, and no semantic understanding.

## Current State in Agent Zero

### retrieval_policy.py (lines 92-94, 288-290, 324, 361)
Every retrieval function uses the same pattern:
```python
overlaps = [term for term in query_terms if term in content or term in note_type or term in scope]
if overlaps:
    score += min(6, len(overlaps) * 2)
```

This is a binary term-present check with linear capping. Problems:
- "exercise" in query won't match "workout" in a memory note
- Common terms like "goal" or "plan" match everything, diluting relevance
- No IDF weighting -- a term appearing in 1/100 notes scored same as one appearing in 90/100
- No term frequency saturation -- keyword stuffed notes rank equally
- No document length normalization -- short specific notes compete unfairly with long narrative notes

### memory_policy.py (lines 84-125)
`_score_note()` adds bonus points for route match, priority overlap, goal overlap, and continuity cues. These are good contextual signals but the core relevance score (keyword overlap) is weak.

### _query_terms() (retrieval_policy.py:523-525, memory_policy.py:128-130)
Extracts 4+ character words, removes stopwords. No stemming, no synonym expansion, no n-grams.

### Current retrieval flow:
1. Fetch recent notes (limit=12), sessions (limit=8), goals (limit=10), requests (limit=8)
2. Score each against query terms via keyword overlap
3. Add contextual bonuses (route, priority, continuity)
4. Sort by score descending, take top N per category
5. Render into prompt fragment

## Industry Standard / Research Findings

### 1. BM25 for Keyword Retrieval (Robertson & Zaragoza, 2009; Rank-BM25 library)
BM25 is the standard baseline for text retrieval, addressing three key weaknesses of term overlap:
- **IDF weighting**: Terms appearing in fewer documents get higher weight. If "meditation" appears in 3/200 notes but "plan" appears in 150/200, "meditation" contributes far more to relevance.
- **Term frequency saturation**: Score follows `tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl))` with k1=1.5, b=0.75. Prevents keyword-stuffed notes from dominating.
- **Length normalization**: Short, specific notes compete fairly with long narrative notes.

Reference: Robertson, S. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval. https://www.nowpublishers.com/article/Details/INR-019

### 2. Hybrid Retrieval with Reciprocal Rank Fusion (Cormack et al., 2009)
Production RAG systems combine sparse (BM25) and dense (embedding) retrieval via Reciprocal Rank Fusion (RRF):
```
RRF_score(d) = sum(1 / (k + rank_i(d))) for each retriever i
```
where k=60 (standard constant). This requires no hyperparameter tuning and consistently outperforms single-method retrieval.

Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009. https://dl.acm.org/doi/10.1145/1571941.1572114

### 3. LongMemEval Benchmark (Wu et al., ICLR 2025)
The LongMemEval benchmark evaluates five core long-term memory abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. Commercial assistants show a 30% accuracy drop on sustained interactions. The Hindsight system achieved 91.4% by running four parallel retrieval strategies (semantic, keyword/BM25, knowledge graph, temporal) with cross-encoder reranking.

Reference: Wu, X. et al. (2025). "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." ICLR 2025. https://arxiv.org/abs/2410.10813

### 4. Hybrid RAG Multi-Strategy (KDD 2025)
Recent research demonstrates weighted ensemble approaches blending BM25 (30% weight) with embedding scores (70% weight). The key insight: BM25 catches exact matches that embeddings miss, while embeddings catch semantic matches that BM25 misses.

Reference: Baban et al. (2025). "Optimizing Retrieval-Augmented Generation with Multi-Strategy Approaches." GenAI Personalization Workshop, KDD 2025. https://genai-personalization.github.io/assets/papers/GenAIRecP2025/11_Baban.pdf

### 5. Memory in the Age of AI Agents Survey (Liu et al., 2025)
Comprehensive survey identifying memory retrieval as a core capability gap in agent systems. Recommends multi-strategy retrieval combining keyword, semantic, temporal, and relational signals.

Reference: Liu, S. et al. (2025). "Memory in the Age of AI Agents: A Survey." https://arxiv.org/abs/2512.13564

### 6. RankRAG (NeurIPS 2024)
Unifies context ranking with retrieval-augmented generation by training the LLM to explicitly identify relevant contexts. Demonstrates that reranking after initial BM25 retrieval significantly improves downstream quality.

Reference: Yu, W. et al. (2024). "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf

## Proposed Implementation

### Phase 1: BM25 Scoring (replace keyword overlap)

**File: `agent_zero/bm25_scorer.py` (new)**

```python
class BM25Scorer:
    """Okapi BM25 scoring for memory retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf_cache: dict[str, float] = {}
        self.avg_dl: float = 0.0
        self.doc_count: int = 0

    def build_index(self, documents: list[dict], text_field: str = "content"):
        """Build IDF statistics from document collection."""
        # Count document frequency for each term
        # Compute average document length
        # Cache IDF values: log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_terms: list[str], document: dict, text_field: str = "content") -> float:
        """BM25 score for a single document against query terms."""
        # For each query term in document:
        #   tf = term frequency in document
        #   dl = document length (word count)
        #   idf = cached IDF value
        #   score += idf * (tf * (k1+1)) / (tf + k1*(1-b + b*dl/avgdl))
```

### Phase 2: Hybrid Scoring with RRF

**File: `agent_zero/memory_policy.py` (modify `_score_note`)**

Replace the keyword overlap block (lines 92-94) with BM25 scoring:
```python
# OLD:
overlaps = [term for term in query_terms if term in content or ...]
if overlaps:
    score += min(6, len(overlaps) * 2)

# NEW:
bm25_score = scorer.score(query_terms, note, text_field="content")
score += min(8, int(bm25_score * 4))  # Scale BM25 (typically 0-3) to int scoring range
```

**File: `agent_zero/retrieval_policy.py` (modify all _select_* functions)**

Same pattern: replace `overlaps = [term for term in query_terms if term in ...]` with BM25 scorer calls in:
- `_select_sessions()` (line 288)
- `_select_goals()` (line 324)
- `_select_requests()` (line 361)
- `_select_profile_cues()` (line 408)

### Phase 3: Index Management

The BM25 scorer needs to be built once per retrieval call from the fetched documents. Since we fetch at most 12+8+10+8 = 38 documents per turn, the index build is trivially fast (microseconds). No persistent index needed.

```python
# In build_retrieval_packet():
all_texts = [n.get("content","") for n in recent_notes] + \
            [s.get("summary","") for s in recent_sessions] + \
            [g.get("goal","") + " " + g.get("notes","") for g in goals]
scorer = BM25Scorer()
scorer.build_index([{"content": t} for t in all_texts])
```

### Phase 4: Query Expansion (optional, low priority)

Add simple stemming via suffix stripping (no external dependencies):
```python
def _stem(word: str) -> str:
    """Minimal suffix stemmer for retrieval."""
    for suffix in ("ing", "tion", "ment", "ness", "able", "ible", "ful", "less", "ous", "ive"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word
```

## Test Specifications

### test_bm25_scorer.py

```
test_bm25_empty_corpus -- empty doc list returns 0.0 scores
test_bm25_single_doc -- single doc scores > 0 for matching terms
test_bm25_idf_weighting -- rare term scores higher than common term
test_bm25_tf_saturation -- 10x repeated term doesn't score 10x higher
test_bm25_length_normalization -- short specific doc ranks above long generic doc
test_bm25_no_match -- query with no matching terms returns 0.0
test_bm25_multiple_query_terms -- multi-term query aggregates correctly
test_bm25_standard_params -- k1=1.5, b=0.75 are defaults
test_bm25_index_rebuild -- can rebuild index with different docs

test_hybrid_score_replaces_overlap -- BM25 score used instead of keyword overlap
test_hybrid_preserves_contextual_bonuses -- route/priority/goal bonuses still apply
test_retrieval_idf_discrimination -- "meditation" (rare) ranks above "plan" (common) for matching notes
test_retrieval_synonym_stemming -- "exercising" matches "exercise" note (if stemming enabled)
test_retrieval_backward_compat -- existing selection_reasons and retrieval_score fields preserved
test_retrieval_empty_query -- empty query returns empty selection (no crash)
test_retrieval_scoring_range -- BM25 scores scaled to existing 0-8 integer range
```

### Integration tests (modify existing test_retrieval_policy.py)
```
test_retrieval_packet_structure_unchanged -- output dict has same keys
test_retrieval_packet_counts_valid -- counts match selected items
test_retrieval_strategic_route_still_boosts -- strategic route bonuses preserved
test_retrieval_continuity_prompt_still_triggers -- continuity phrases still detected
```

## Estimated Impact

- **Precision improvement**: IDF weighting will suppress common terms ("goal", "plan", "feel") that currently match everything. Estimated 30-50% reduction in false positive memory retrievals.
- **Recall improvement**: Term frequency saturation and length normalization will surface short, specific notes that currently lose to long narrative notes. Estimated 20% improvement in retrieving the most relevant memory.
- **No latency impact**: BM25 on 38 documents is microseconds. No external dependencies required.
- **Backward compatible**: Output format unchanged. Contextual bonuses (route, priority, continuity) preserved.
- **Foundation for future semantic retrieval**: Once BM25 is in place, RRF fusion with embedding-based retrieval can be added incrementally.
