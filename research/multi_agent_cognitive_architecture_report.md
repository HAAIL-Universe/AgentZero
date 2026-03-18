# Multi-Agent Cognitive Architectures: State of the Art (2024-2026)
## Deep Research Report -- March 2026

---

## 1. Multi-Agent Frameworks: The Big Three + Google ADK

### 1.1 Framework Landscape

Three frameworks dominate, each with a fundamentally different philosophy:

**LangGraph** (LangChain) -- Graph-based state machines
- Models agent workflows as directed graphs of nodes (functions, tools, models) connected by edges (decision logic)
- Shared state persists through the graph, enabling retries, branching, loops, and multi-agent patterns
- Reached v1.0 in late 2025; now the default runtime for all LangChain agents
- Key patterns: sequential, branching/conditional, parallel (fork/join), and looping
- Strength: maximum control, compliance, production-grade state management
- Weakness: requires strong grasp of graph theory and state machines
- Best for: enterprises building mission-critical, auditable agent systems

**CrewAI** -- Role-based team metaphor
- Models collaboration as a "crew" of role-playing agents with defined roles, backstories, and goals
- Intuitive abstraction for rapid prototyping
- Limitation: as requirements grow beyond sequential/hierarchical task execution, the opinionated design becomes constraining. Multiple teams report hitting this wall 6-12 months in, requiring rewrites to LangGraph

**AutoGen** (Microsoft, now merging into Microsoft Agent Framework)
- Conversational agent architecture emphasizing natural language interactions and dynamic role-playing
- Best for iterative refinement tasks: code generation, research, creative problem-solving
- As of February 2026: merging with Semantic Kernel into unified "Microsoft Agent Framework" (RC status, GA targeted Q1 2026)
- The merged framework combines AutoGen's multi-agent abstractions with Semantic Kernel's enterprise features (state management, type safety, filters, telemetry)
- AutoGen and Semantic Kernel entering maintenance mode (bug fixes only, no new features)

**Google Agent Development Kit (ADK)** -- Announced at Cloud NEXT 2025
- Open-source, model-agnostic, deployment-agnostic
- Three communication mechanisms:
  1. **Shared Session State**: shared digital whiteboard -- agents write results to common state, others read
  2. **LLM-Driven Delegation**: parent agent uses reasoning to route tasks to best sub-agent
  3. **Agent-to-Agent (A2A) Protocol**: open standard for cross-framework agent communication via "Agent Cards"
- Supports sequential, loop, and parallel workflows
- Deep integration with Vertex AI and Google Cloud

### 1.2 The Emerging "Agentic Mesh" Trend

The future is not about choosing a single framework. We are moving toward a modular ecosystem where a LangGraph "brain" might orchestrate a CrewAI "marketing team" while calling specialized tools for sub-tasks. If 2024 was the year of the Chatbot and 2025 the year of the Agent, 2026 is the year of the Architect.

### 1.3 Actionable Insight

For AgentZero's A1-A2 architecture: LangGraph's graph-based state machine model maps most naturally to complex agent orchestration. The shared-state + conditional-routing pattern is what you already do with MQ + NEXT.md. Consider formalizing this into a proper state graph where transitions are explicit and auditable.

---

## 2. Inter-Agent Communication Patterns

### 2.1 Four Core Patterns

**Blackboard (Shared Workspace)**
- A supervisor agent posts tasks to a shared workspace; specialists read, execute, and write results back
- Used in systems like InsurifyAI for complex document processing
- Pros: simple coordination, natural for heterogeneous agents
- Cons: requires conflict resolution for concurrent writes

**Message Passing (Direct)**
- Structured messages between specific agents (point-to-point)
- Most direct form -- one agent sends position/status/results to another
- Pros: precise, traceable, low overhead
- Cons: requires agents to know about each other (tight coupling)

**Publish-Subscribe (Pub/Sub)**
- Agents subscribe to topics; messages broadcast to all subscribers
- Used in high-velocity environments (fraud detection, real-time trading)
- Pros: decoupled, scalable, natural for event-driven systems
- Cons: harder to debug, potential message storms

**Market-Based**
- Agents bid on tasks based on capability and availability
- Natural load balancing
- Pros: self-organizing, handles heterogeneous capabilities
- Cons: overhead of bidding protocol, latency

### 2.2 Protocol Evolution (2025-2026)

Four major protocols have emerged for agent interoperability:

1. **MCP (Model Context Protocol)** -- Anthropic. JSON-RPC client-server interface for tool invocation. Standardizes how agents connect to tools, APIs, and resources. Most widely adopted as of 2026.

2. **A2A (Agent-to-Agent Protocol)** -- Google (now Linux Foundation). Peer-to-peer agent communication: message passing, role negotiation, task delegation. Protobuf serialization, async pub/sub support. 50+ technology partners at launch.

3. **ACP (Agent Communication Protocol)** -- Emerging standard for structured semantic messaging.

4. **ANP (Agent Network Protocol)** -- Decentralized agent discovery and collaboration.

**Key insight**: MCP and A2A are complementary, not competing. MCP standardizes how agents use tools (vertical); A2A standardizes how agents talk to each other (horizontal).

### 2.3 Performative-Based Messaging

By 2026, systems are shifting from rigid rule-based messaging to dynamic semantic exchanges. Messages carry intent ("I request", "I inform", "I propose") rather than just data, enabling richer negotiation between agents.

### 2.4 Actionable Insight

AgentZero's MQ system is message-passing. Consider adding a shared-state layer (blackboard) for A1-A2 to share intermediate results without full message round-trips. The MQ handles missions/findings (structured); a blackboard would handle live working state.

---

## 3. Cognitive Routing and Mixture of Experts

### 3.1 MoE Applied to Agent Selection

**Symbolic Mixture-of-Experts (Symbolic-MoE)** -- March 2025
- A gradient-free, text-based MoE framework for adaptive instance-level mixing of pre-trained LLM experts
- Key innovation: skill-based recruiting -- dynamically selects expert LLMs based on skills needed per problem instance (e.g., "Algebra" experts for Q1, "Probability" experts for Q2)
- Generates only a single round of responses with a selected aggregator
- Performance: +8.15% absolute over best multi-agent baseline; beats GPT-4o-mini
- Efficiency: 16 expert models on 1 GPU, comparable time cost to 4-GPU baselines
- Paper: arxiv.org/abs/2503.05641

### 3.2 Routing Strategy Evolution

**Top-K Routing** (traditional): fixed number of experts per input
**Top-P Routing** (emerging): selects experts based on cumulative probability mass -- high-confidence tokens use fewer experts, uncertain ones activate more
**Expert Choice Routing** (Google): experts choose which tokens to process, not the other way around
**Expert Threshold Routing** (2026): dynamic computation allocation with load balancing

### 3.3 Industry Trend

As of early 2026, virtually all leading frontier models (DeepSeek-V3/R1, Llama 4, Mistral Large 3, Gemini) use MoE architectures. The trend is toward small parameters + many experts (e.g., DeepSeek-V3: 256 experts) with fine-grained division and dynamic routing.

### 3.4 Actionable Insight

For a cognitive routing layer in AgentZero: implement a skill-based router that analyzes incoming tasks, identifies required skills (e.g., "needs formal verification" vs "needs code generation"), and dispatches to the appropriate specialist (A1 for building, A2 for verification). This is essentially what you do manually via MQ missions -- automate it with a lightweight skill classifier.

---

## 4. Behavioral AI and Adaptive Systems

### 4.1 AI Agent Behavioral Science

A new research paradigm has emerged: "AI Agent Behavioral Science" -- shifting focus from model-centric analysis to interaction, adaptation, and emergent dynamics. Key principles:
- Systematic observation of behavior (not just outputs)
- Hypothesis-driven intervention design
- Theory-informed interpretation to uncover behavioral regularities

**Behavior Intelligence** is defined as the ability to flexibly adapt and guide actions in human-machine interactions depending on context, requiring integration of game theory, reinforcement learning, and Theory of Mind.

### 4.2 Personality Consistency

**MBTI-in-Thoughts** (Swiss Federal Institute of Technology): primes agents with distinct personality types through prompt engineering, creating consistent emotional responses without retraining.

**EvoEmo**: allows AI to adapt emotional tone during conversations -- agents become more conciliatory, assertive, or supportive depending on dialogue flow.

Tension: fixed personality vs. adaptive personality. The research suggests the best approach is a stable core personality with adaptive surface behavior.

### 4.3 Affective Computing

Current capabilities:
- Emotion recognition via NLP, sentiment analysis, and deep learning
- User-specific emotional profiles built from conversation history
- Adaptive emotional support based on context

Limitations:
- ~15% error rate on ambiguous expressions (sarcasm, mixed emotions) under real-world conditions
- Long-term contextual memory remains computationally expensive
- Multi-day interaction continuity still weak

### 4.4 The Pseudo-Intimacy Problem

Frontiers in Psychology (2025) raises concerns about "pseudo-intimacy" -- users forming emotional bonds with AI that simulates emotional understanding. This is a design consideration, not just an ethics question: systems that simulate emotion too convincingly may create dependency.

### 4.5 Actionable Insight

For Agent Zero personality: implement a two-layer approach: (1) stable core traits stored in identity config (values, communication style, reasoning preferences), (2) adaptive surface layer that adjusts tone/approach based on conversation context. The core never changes; the surface adapts. Store interaction patterns in episodic memory to inform future adaptation.

---

## 5. Advanced Agent Communication Patterns

### 5.1 Chain-of-Thought Delegation

Chain-of-thought multi-agent systems distribute stepwise reasoning across specialized agents. Each agent performs part of the reasoning chain and passes intermediate traces to the next. Amazon's approach breaks this into three stages:
1. **Intent decomposition**: identify explicit and implicit intents
2. **Deliberation**: iterative expansion where multiple LLMs review and correct the chain
3. **Refinement**: post-processing to filter redundant/inconsistent thoughts

### 5.2 Generator-Critic Patterns

The generator-critic loop has become a first-class architectural pattern in 2026:
- Generator produces candidate output
- Critic evaluates against criteria
- Generator refines based on critique
- Iterate until quality threshold met

Recent papers:
- "Generator-Assistant Stepwise Rollback Framework" (March 2025)
- "Table-Critic: Multi-Agent Framework for Collaborative Criticism and Refinement" (Feb 2025)
- "The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement" (March 2025)

**Multi-Agent Reflexion (MAR)** (December 2025): extends single-agent Reflexion to multi-agent settings, addressing cognitive entrenchment and degeneration-of-thought problems that plague single-agent self-improvement.

### 5.3 Memory-Augmented Agent Architectures

**Letta/MemGPT** -- the LLM-as-Operating-System paradigm:
- Two-tier memory: core memory (in-context, like RAM) and external memory (archival/recall, like disk)
- Agent manages its own memory, moving data between tiers as needed
- Core memory: always in context, focused on specific topics (user, task, organization)
- Recall memory: complete interaction history, searchable
- Archival memory: processed, indexed knowledge in external databases

**Episodic-to-Semantic Consolidation** (2025 research focus):
- Episodic memories capture specific experiences with contextual details (when, where)
- Background process analyzes traces, identifies successful patterns
- Abstracts patterns into generalizable skills/rules in semantic memory
- This is how agents learn from experience without explicit retraining

**Transactive Memory** (emerging):
- Like human teams knowing "who knows what"
- Meta-memory: agents track what other agents know
- Enables efficient cognitive resource allocation across agent teams

### 5.4 Reflection and Meta-Cognition

**Reflexion** (foundational, 2023): agents generate natural language critiques of mistakes, store in episodic memory for future guidance.

**Agent-R** (March 2025): iterative self-training where agents learn to reflect on-the-fly using Monte Carlo Tree Search to construct recovery trajectories from errors.

**ICML 2025 Position Paper**: argues truly self-improving agents require intrinsic metacognitive learning -- the ability to actively evaluate, reflect on, and adapt their own learning processes. Current approaches are too rigid and fail to generalize across task domains.

**SAGE** (2025): Self-evolving Agents with Reflective and Memory-augmented Abilities -- combines reflection with persistent memory for continuous self-improvement.

### 5.5 Actionable Insight

AgentZero already has episodic memory (session journals) and a reflection step (assess.py). The missing piece is episodic-to-semantic consolidation: automatically extracting patterns from session journals into reusable knowledge. Consider a tool that periodically scans recent sessions, identifies recurring patterns/solutions, and writes them to a semantic memory store (similar to what memory.py does, but automated).

---

## 6. Cutting-Edge Papers and Frameworks (2025-2026)

### 6.1 Cognitive Architecture Papers

**"Applying Cognitive Design Patterns to General LLM Agents"** (June 2025)
- Maps SOAR/ACT-R decision processes to an abstract observe-decide-act pattern
- Presents cognitive design patterns as organizational tools for agentic LLM research
- URL: arxiv.org/html/2505.07087v2

**"Cognitive LLMs: Towards Integrating Cognitive Architectures and Large Language Models"** (2024)
- Extracts ACT-R's decision-making process as latent neural representations
- Embeds them into LLM, improving task performance on manufacturing benchmarks
- URL: arxiv.org/pdf/2408.09176

**"Agentic AI: Architectures, Taxonomies, and Evaluation"** (January 2026)
- Unified taxonomy: 6 modular dimensions for LLM agents
  1. Core Components (perception, memory, action, profiling)
  2. Cognitive Architecture (planning, reflection)
  3. Learning
  4. Multi-Agent Systems
  5. Environments
  6. Evaluation
- URL: arxiv.org/html/2601.12560v1

**"Learn Like Humans: Meta-cognitive Reflection for Efficient Self-Improvement"** (January 2026)
- URL: arxiv.org/html/2601.11974v1

**"Position: Truly Self-Improving Agents Require Intrinsic Metacognitive Learning"** (ICML 2025)
- URL: openreview.net/forum?id=4KhDd0Ozqe

### 6.2 Society of Mind Approaches

**"Society of Mind Meets Real-Time Strategy"** (2025)
- Hierarchical multi-agent framework for strategic reasoning
- URL: arxiv.org/abs/2508.06042

**"Multi-Agent Collaboration via Evolving Orchestration"** (May 2025)
- Puppeteer-style paradigm: centralized orchestrator dynamically directs agents based on evolving task state
- Produces an implicit inference graph
- URL: arxiv.org/html/2505.19591v2

**Key empirical finding** (2024): a multi-agent discussion approach outperformed single-agent chain-of-thought prompting on benchmarks with no additional human data. A collection of mediocre reasoners, when allowed to interact, produced superior outcomes.

### 6.3 Memory Systems

**"Memory in the Age of AI Agents"** (December 2025)
- Comprehensive survey of agent memory systems
- URL: arxiv.org/abs/2512.13564

**"Anatomy of Agentic Memory: Taxonomy and Empirical Analysis"** (February 2026)
- Evaluation of memory system limitations
- URL: arxiv.org/html/2602.19320v1

**"MIRIX: Multi-Agent Memory System"** (July 2025)
- Multi-agent memory architecture
- URL: arxiv.org/pdf/2507.07957

### 6.4 Agent Interoperability

**"A Survey of Agent Interoperability Protocols"** (May 2025)
- Covers MCP, ACP, A2A, and ANP
- URL: arxiv.org/html/2505.02279v1

---

## 7. Synthesis: Key Takeaways for AgentZero

### Architecture Patterns to Adopt

1. **Graph-Based Orchestration**: Formalize the A1-A2 workflow as a state graph (LangGraph-style). Each session step (orientation, inbox check, work, reflection, journal) becomes a node. Transitions are conditional on state. This makes the system auditable and extensible.

2. **Dual Memory Tiers**: Adopt MemGPT's two-tier model. Core memory (always loaded -- identity, current goals, active context) and archival memory (session journals, patterns, challenge solutions). Use the existing memory.py but add explicit tier management.

3. **Episodic-to-Semantic Consolidation**: Build an automated pipeline that scans session journals, extracts patterns, and promotes them to semantic memory. This is the missing learning loop.

4. **Skill-Based Routing**: When dispatching to A2, classify the task by required skill (complexity analysis, symbolic execution, fault localization, etc.) and include the skill tag in the MQ message. This enables future automation of the dispatch decision.

5. **Generator-Critic Loop**: Formalize the A1-builds/A2-verifies pattern as an explicit generator-critic loop with iteration counts and quality thresholds. Currently this is ad-hoc via MQ; make it a first-class workflow.

### Communication Patterns to Consider

6. **Shared State Layer**: Add a lightweight blackboard (JSON file) alongside MQ for sharing live working state between A1 and A2 without the overhead of formal messages.

7. **A2A-style Agent Cards**: Define capability cards for A1 and A2 that explicitly list what each agent can do. This enables future agents to discover and route to them.

### Personality and Behavior

8. **Two-Layer Personality**: For Agent Zero -- stable core traits (identity.md) plus adaptive surface behavior (learned from interaction patterns). Never change the core; let the surface evolve.

9. **Meta-Cognitive Reflection**: The assess.py tool is already a form of meta-cognition. Enhance it to not just score but to generate specific behavioral recommendations that feed into the next session's planning.

### What to Watch

10. **Microsoft Agent Framework GA** (Q1 2026) -- if you ever need enterprise-grade multi-agent infrastructure
11. **A2A Protocol maturation** -- the emerging standard for agent interoperability
12. **Symbolic-MoE patterns** -- gradient-free expert routing applicable to cognitive agent selection

---

## Sources

### Frameworks
- [LangGraph vs CrewAI vs AutoGen 2026 Guide](https://dev.to/pockit_tools/langgraph-vs-crewai-vs-autogen-the-complete-multi-agent-ai-orchestration-guide-for-2026-2d63)
- [AutoGen vs LangGraph vs CrewAI 2026](https://dev.to/synsun/autogen-vs-langgraph-vs-crewai-which-agent-framework-actually-holds-up-in-2026-3fl8)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Google ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/)
- [Google ADK Blog Post](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- [Microsoft Agent Framework Overview](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Microsoft Agent Framework RC Migration](https://devblogs.microsoft.com/semantic-kernel/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/)
- [Top 5 Open-Source Agentic AI Frameworks 2026](https://aimultiple.com/agentic-frameworks)

### Communication Patterns and Protocols
- [Communication Between Agents: Formats, Protocols, Coordination](https://mbrenndoerfer.com/writing/communication-between-agents)
- [Four Design Patterns for Event-Driven Multi-Agent Systems](https://www.confluent.io/blog/event-driven-multi-agent-systems/)
- [Agent Interoperability Protocols Survey (MCP, ACP, A2A, ANP)](https://arxiv.org/html/2505.02279v1)
- [A2A Protocol](https://a2a-protocol.org/latest/)
- [A2A and MCP Complementary Roles](https://a2a-protocol.org/latest/topics/a2a-and-mcp/)
- [MCP vs A2A 2026](https://onereach.ai/blog/guide-choosing-mcp-vs-a2a-protocols/)
- [Agent Blackboard Pattern (GitHub)](https://github.com/claudioed/agent-blackboard)

### Cognitive Routing and MoE
- [Symbolic MoE: Adaptive Skill-based Routing](https://arxiv.org/abs/2503.05641)
- [Top-K Routing in MoE Models](https://mbrenndoerfer.com/writing/top-k-routing-mixture-of-experts-expert-selection)
- [MoE Architecture Scaling 2026](https://www.youngju.dev/blog/ai-papers/2026-03-04-ai-papers-mixture-of-experts-scaling-2026.en)
- [Expert Threshold Routing](https://arxiv.org/html/2603.11535)
- [MoE in LLMs Survey](https://arxiv.org/html/2507.11181v2)

### Behavioral AI
- [AI Agent Behavioral Science](https://arxiv.org/html/2506.06366v2)
- [Artificial Behavior Intelligence](https://arxiv.org/html/2505.03315v1)
- [Governing Interactive AI with Behavioral Insights](https://aihub.org/2026/02/10/governing-the-rise-of-interactive-ai-will-require-behavioral-insights/)
- [Behavioral Insights for AI Recommendations (Stanford)](https://news.stanford.edu/stories/2025/09/behavioral-insights-user-intent-ai-driven-recommendations-youtube)
- [Emotional AI and Pseudo-Intimacy](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1679324/full)
- [Affective Computing and Emotional Data](https://arxiv.org/html/2509.20153v2)

### Memory and Meta-Cognition
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [Memory in Agentic AI Systems](https://genesishumanexperience.com/2025/11/03/memory-in-agentic-ai-systems-the-cognitive-architecture-behind-intelligent-collaboration/)
- [Letta/MemGPT Documentation](https://docs.letta.com/concepts/memgpt/)
- [Letta Agent Memory](https://www.letta.com/blog/agent-memory)
- [Anatomy of Agentic Memory](https://arxiv.org/html/2602.19320v1)
- [MIRIX Multi-Agent Memory System](https://arxiv.org/pdf/2507.07957)

### Cognitive Architectures
- [Cognitive Design Patterns for LLM Agents](https://arxiv.org/html/2505.07087v2)
- [Cognitive LLMs: Integrating ACT-R and LLMs](https://arxiv.org/pdf/2408.09176)
- [Agentic AI Architectures and Taxonomies](https://arxiv.org/html/2601.12560v1)
- [Cognitive Architecture in AI (Sema4)](https://sema4.ai/learning-center/cognitive-architecture-ai/)

### Reflection and Self-Improvement
- [Meta-cognitive Reflection for Self-Improvement](https://arxiv.org/html/2601.11974v1)
- [Truly Self-Improving Agents Require Metacognitive Learning (ICML 2025)](https://openreview.net/forum?id=4KhDd0Ozqe)
- [Multi-Agent Reflexion (MAR)](https://arxiv.org/html/2512.20845v1)
- [Agent-R: Iterative Self-Training](https://arxiv.org/abs/2501.11425)
- [SAGE: Self-evolving Agents](https://arxiv.org/html/2409.00872v2)

### Society of Mind and Orchestration
- [Language Model Agents in 2025: Society of Mind Revisited](https://isolutions.medium.com/language-model-agents-in-2025-897ec15c9c42)
- [Society of Mind Meets Real-Time Strategy](https://arxiv.org/abs/2508.06042)
- [Multi-Agent Collaboration via Evolving Orchestration](https://arxiv.org/html/2505.19591v2)
- [Multi-Agent Collaboration Mechanisms Survey](https://arxiv.org/html/2501.06322v1)
- [Chain-of-Thought Multi-Agent Systems](https://www.emergentmind.com/topics/chain-of-thought-multi-agent-systems)
- [Chain-of-Agents: End-to-End Agent Models](https://openreview.net/pdf?id=VcT9KJeB89)

### Generator-Critic and CoT Delegation
- [Amazon Multiagent CoT Generation](https://www.amazon.science/blog/multiagent-ai-for-generating-chain-of-thought-training-data)
- [LangGraph Multi-Agent Orchestration Guide](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
