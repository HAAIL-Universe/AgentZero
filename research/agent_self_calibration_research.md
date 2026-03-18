# Should Agents See Their Own Performance Data?

## Research Survey (2024-2026)

### Research Question
In multi-agent AI systems, should agents be shown their own performance/calibration
data (e.g., "your proposals are adopted 62% of the time")? Does this improve or
distort their output?

---

## 1. Expert Persona Prompting: "You Are Good/Bad at X"

**Key Paper: "Playing Pretend: Expert Personas Don't Improve Factual Accuracy"**
(Basil, Shapiro, Shapiro, Mollick, Mollick, Meincke -- Dec 2025, SSRN)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5879722

**Findings:**
- Expert personas ("you are a physics expert") matched to task domain showed NO
  significant improvement in accuracy vs. a no-persona baseline
- Domain-mismatched expert personas sometimes DEGRADED performance
- **Low-knowledge/negative personas ("you are a toddler", "you are a layperson")
  reliably REDUCED accuracy**
- Exception: Gemini 2.0 Flash showed some benefit from in-domain expert personas

**Implication:** Telling a model "you're good at X" is largely neutral. Telling it
"you're bad at X" actively hurts. This is asymmetric -- negative framing is more
damaging than positive framing is helpful.

---

## 2. Confidence Calibration When Given Feedback

### 2a. Overconfidence is the Default

**"On Verbalized Confidence Scores for LLMs"** (Yang, ETH Zurich, 2024)
https://arxiv.org/pdf/2412.14737

- In zero-shot settings, LLMs predominantly assign confidence scores of 8+ out of 10
- This reflects RLHF training bias toward confident-sounding outputs
- Even well-performing models show minimal variation in confidence between correct
  and incorrect answers

**"Mind the Confidence Gap"** (2025)
https://arxiv.org/html/2502.11028v3

- Confidence remains surprisingly stable despite severe perception degradation
- Persistent overconfidence even at larger model scales

### 2b. The Accuracy-Calibration Tradeoff

**"Decoupling Reasoning and Confidence" (DCPO)** (2025)
https://arxiv.org/html/2603.09117

- **Critical finding:** Coupled optimization of correctness and calibration leads to
  an "accuracy-calibration tradeoff" -- improvements in calibration frequently come
  at the expense of reasoning accuracy
- Proposed solution: separate reward signals for reasoning vs. confidence tokens

### 2c. Rewarding Doubt (ICLR 2026)

**"Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence
Expression"** (Stangel, Bani-Harouni -- ICLR 2026)
https://arxiv.org/abs/2503.02623

- RL fine-tuning using logarithmic scoring rule that penalizes both over- and under-confidence
- Models trained this way show substantially improved calibration AND generalize to unseen tasks
- ECE drops to 0.0226, AUROC reaches 0.8592 on TriviaQA
- **This is evidence that calibration CAN be improved, but requires training, not just prompting**

### 2d. QA-Calibration (ICLR 2025)

https://assets.amazon.science/6d/70/c50b2eb141d3bcf1565e62b60211/qa-calibration-of-language-model-confidence-scores.pdf

- Calibration should be conditional on question-answer groups, not averaged globally
- Different question types require different calibration strategies

---

## 3. Meta-Cognitive Prompting with Effectiveness Scores

### 3a. Metacognitive Prompting Improves Understanding (NAACL 2024)

https://aclanthology.org/2024.naacl-long.106.pdf

- 5-stage metacognitive framework: understand -> preliminary judgment -> critical
  evaluation -> final decision -> confidence assessment
- Improves accuracy across diverse NLU tasks
- Key: the model evaluates its OWN reasoning process, not external performance data

### 3b. Think-Solve-Verify Framework (LREC 2024)

https://aclanthology.org/2024.lrec-main.1465/

- Improved performance from 67.3% to 72.8% on AQuA dataset
- Self-awareness prompts help, but the effect is about process reflection, not
  historical performance statistics

### 3c. AutoMeco: Intrinsic Meta-Cognition (EMNLP 2025)

**"Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens"**
(Ma, Yuan, Wang, Zhou -- EMNLP 2025)
https://aclanthology.org/2025.emnlp-main.171/

- LLMs DO have intrinsic meta-cognitive signals in hidden states
- These signals predict step-level errors in reasoning chains
- MIRA strategy propagates error signals through sequential Q-value adjustment
- **Key insight:** The meta-cognition is there but poorly accessible through prompting alone

### 3d. MIT Recursive Meta-Cognition (110% improvement claim)

- Recursive meta-cognition techniques showed up to 110% improvement over standard
  prompting on complex tasks (puzzles, coding)
- The improvement comes from structured self-reflection DURING the task, not from
  external performance statistics

---

## 4. Self-Fulfilling Prophecy: Does Negative Framing Make Models Worse?

### 4a. Yes -- Negative Personas Degrade Output

From the "Playing Pretend" study above:
- Low-knowledge personas ("toddler", "layperson") reliably reduced accuracy
- The model internalizes the framing and produces worse output

### 4b. Self-Bias Amplification

**"Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement"** (ACL 2024)
https://aclanthology.org/2024.acl-long.826/

- Self-feedback improves performance on SOME tasks while DEGRADING others
- Self-bias = tendency to favor own generation
- Self-refine pipelines AMPLIFY this bias while improving surface-level fluency
- Tested across 6 LLMs (GPT-4, GPT-3.5, Gemini, LLaMA2, Mixtral, DeepSeek)
- **External feedback with accurate assessment significantly reduces bias**

### 4c. Anti-Bayesian Drift

**Research from 2025 (arXiv 2505.19184)**
https://arxiv.org/pdf/2505.19184

- LLMs become MORE overconfident after encountering counter-arguments (anti-Bayesian)
- This undermines reliability in agentic settings requiring accurate self-assessment

### 4d. Verbalized Confidence vs. Actual Decisions (2026)

**"Are LLM Decisions Faithful to Verbal Confidence?"**
https://arxiv.org/html/2601.07767v1

- Models can often accurately verbalize their uncertainty
- But they FAIL to use this information to minimize loss
- Knowing your own accuracy and acting on it are different capabilities

---

## 5. Production Systems with Performance Metrics in Prompts

### Direct evidence is sparse. Key observations:

**AutoGenBench** (Microsoft, 2024)
https://microsoft.github.io/autogen/0.2/blog/2024/01/25/AutoGenBench/

- Provides benchmarking infrastructure for measuring agent performance
- No published evidence of feeding these metrics back INTO agent prompts

**Multi-Agent Feedback Survey (IJCAI 2025)**
https://www.ijcai.org/proceedings/2025/1175.pdf

- Covers iterative feedback and inter-task knowledge transfer
- Focus is on task-level feedback (correct/incorrect), not aggregate performance statistics

**Multi-Agent Debate Frameworks:**
- Agents critique each other's reasoning to reach consensus
- This is PEER feedback, not self-performance-statistics
- Shown to reduce hallucinations and improve mathematical reasoning

**No published production case study** was found where agent prompts include statements
like "your accuracy is 62%" or "your proposals are adopted X% of the time." This
appears to be a gap in the literature.

---

## 6. A/B Testing: WITH vs WITHOUT Self-Calibration Data

**No direct ablation study found** comparing agent performance with vs. without
historical self-performance statistics in prompts.

The closest analogs:

- **Self-Refine ablations** show that self-feedback helps on some tasks but hurts
  on others (see "Pride and Prejudice" above)
- **Persona ablations** show expert personas are ~neutral while negative personas
  degrade performance
- **Metacognitive prompting ablations** show process-reflection helps, but this is
  about reasoning structure, not performance statistics
- **Confidence calibration ablations** (Rewarding Doubt) show training-time
  calibration works but prompting-time calibration is unreliable

---

## Synthesis: What the Evidence Actually Says

### Things that HELP:
1. **Structured self-reflection during reasoning** (metacognitive prompting, 5-stage
   frameworks) -- 10-110% improvement depending on task
2. **External peer feedback** (multi-agent debate, critique) -- reduces hallucinations
3. **Training-time calibration** (Rewarding Doubt RL) -- produces genuine uncertainty
   awareness
4. **Accurate external assessment** fed into self-refine loops -- reduces self-bias

### Things that HURT:
1. **Negative framing** ("you are bad at X") -- reliably degrades accuracy
2. **Unchecked self-reflection** -- amplifies self-bias, makes models more overconfident
3. **Coupled accuracy-calibration optimization** -- improving calibration can degrade
   reasoning accuracy

### Things with NO CLEAR EFFECT:
1. **Positive expert personas** ("you are an expert at X") -- mostly neutral
2. **Verbalized confidence scores** -- models can say them but don't act on them

### The Gap:
**Nobody has published a rigorous study on feeding aggregate performance statistics
("your accuracy is 62%") into agent prompts.** This specific intervention sits in
unmapped territory between persona-prompting (studied, mostly neutral/harmful) and
metacognitive prompting (studied, helpful but about process not statistics).

---

## Recommendations for Multi-Agent System Design

Based on the evidence:

1. **Do NOT inject negative performance framing.** "Your proposals are rejected 38%
   of the time" will likely degrade output. The self-fulfilling prophecy is real
   for negative framing.

2. **Positive framing is probably neutral.** "Your proposals are adopted 62% of the
   time" is unlikely to help or hurt much, based on expert-persona research.

3. **Process reflection > statistics.** Instead of "your accuracy is X%", prompt
   agents to evaluate their own reasoning step-by-step. This is what actually works.

4. **Use peer feedback instead.** Multi-agent debate and critique outperform
   self-assessment. Let agents challenge each other rather than contemplating their
   own track records.

5. **If you must use performance data, make it external and specific.** "This
   specific type of proposal tends to fail when X" is more useful than "your overall
   adoption rate is Y%." Accurate external feedback reduces self-bias (ACL 2024).

6. **Beware the accuracy-calibration tradeoff.** Making an agent more calibrated
   about its uncertainty can reduce its actual task performance. Treat these as
   separate optimization targets.

7. **The verbalization-action gap is real.** Even if an agent correctly states its
   uncertainty, it may not USE that information to improve decisions. Don't assume
   that awareness leads to better behavior.

---

---

## Addendum: Additional Quantitative Evidence (March 2026 Search)

### Prompt Sentiment Quantitative Impact

**"Prompt Sentiment: The Catalyst for LLM Change" (arXiv 2503.13510, 2025)**

Direct measurement of how emotional framing affects factual accuracy:
- Neutral prompts: 92.3% factual accuracy (baseline)
- Positive prompts: 89.7% accuracy (~2.8% decline)
- Negative prompts: 84.5% accuracy (~8.4% decline vs neutral)

Response length asymmetry:
- Positive prompts: 8.1% longer than neutral
- Negative prompts: 17.6% shorter (disengagement/terseness)

Subjective domains amplify sentiment. Objective domains resist it.
https://arxiv.org/html/2503.13510

### Reflexion Ablation (Quantitative)

On HumanEval Rust (50 hardest problems):
- Base model: 0.60 pass@1
- Self-reflection omitted: 0.60 (no improvement)
- Test generation omitted: 0.52 (worse than baseline)
- Full Reflexion: 0.68

On HotPotQA: CoT 0.60 -> CoT+memory 0.68 -> CoT+Reflexion 0.80

**Critical:** Reflection alone without concrete test signals = ZERO improvement.
Reflection only works when grounded in specific, verifiable task feedback.

### Self-Refine Ablation (Quantitative)

Code Optimization task:
- Specific feedback: 27.5
- Generic feedback: 26.0
- No feedback: 24.8

~2.7 point gain from specific vs generic, ~1.2 from generic vs none.

### Multi-Expert Prompting

Multi-expert aggregation outperforms single expert prompting by 8.69% on
truthfulness (ChatGPT), while simple "you are an expert" role prompts show
minimal effect on factual tasks.

### Threat-Based Manipulation (arXiv 2507.21133, 2025)

3,390 experimental responses across Claude, GPT-4, Gemini under 6 threat
conditions across 10 task domains. Complex responses: both vulnerabilities
and unexpected performance changes. No simple "threats always help/hurt" story.

### LLM Honesty Survey (TMLR 2025)

Covers self-knowledge evaluation including recognition of known/unknown,
calibration, and selective prediction. Core finding: LLMs have limited ability
to recognize what they know vs don't know without external scaffolding.
https://github.com/SihengLi99/LLM-Honesty-Survey

### Scaling Multi-Agent Systems (arXiv 2512.08296, Dec 2025)

Performance is normalized to best single-agent baseline to measure coordination
gain or loss. This is EVALUATION methodology -- no evidence of feeding these
normalized scores back into agent prompts at runtime.

### Key Gap Confirmed

As of March 2026, no published ablation study directly tests:
"Agent with aggregate performance stats in prompt" vs "Agent without."

The evidence converges from adjacent fields:
- Negative framing hurts (~8% accuracy loss)
- Aggregate stats are not actionable (no concrete improvement path)
- Task-specific feedback helps (but must be grounded in specifics)
- Process reflection outperforms data reflection

---

## Key Papers (Citation List)

1. Basil et al. "Playing Pretend: Expert Personas Don't Improve Factual Accuracy" (2025, SSRN)
2. Xu et al. "Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement" (ACL 2024)
3. Stangel & Bani-Harouni. "Rewarding Doubt" (ICLR 2026)
4. Ma et al. "Large Language Models Have Intrinsic Meta-Cognition" (EMNLP 2025)
5. Wang et al. "Metacognitive Prompting Improves Understanding in LLMs" (NAACL 2024)
6. Yang. "On Verbalized Confidence Scores for LLMs" (2024)
7. "Decoupling Reasoning and Confidence" (2025)
8. "Are LLM Decisions Faithful to Verbal Confidence?" (2026)
9. "A Survey on the Feedback Mechanism of LLM-based AI Agents" (IJCAI 2025)
10. "Mind the Confidence Gap" (2025)
11. "Prompt Sentiment: The Catalyst for LLM Change" (arXiv 2503.13510, 2025)
12. "Emotional Framing as a Control Channel" (NeurIPS 2025)
13. Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" (NeurIPS 2023)
14. Madaan et al. "Self-Refine: Iterative Refinement with Self-Feedback" (2023)
15. "A Survey on the Honesty of Large Language Models" (TMLR 2025)
16. "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296, 2025)
17. "Threat-Based Manipulation in Large Language Models" (arXiv 2507.21133, 2025)
18. "Multi-Expert Prompting" (arXiv 2411.00492, 2024)
19. "ExpertPrompting" (arXiv 2305.14688, 2023)
20. "Should We Respect LLMs? Prompt Politeness" (NAACL 2024)
