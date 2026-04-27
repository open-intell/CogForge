# CogForge + CogWorks Swarm

**A lightweight, memory-augmented coding & reasoning transformer with a native multi-agent software engineering swarm.**

Built from the ground up to **digest huge amounts of code without forgetting** while producing maintainable, Google-style high-quality software.

---

## Overview

**CogForge** is a ~100M parameter transformer specifically designed for long-context coding and deep reasoning.  
**CogWorks** is a team of 11 specialized agents that collaborate using structured handoffs, shared hierarchical memory, and a principled engineering workflow heavily influenced by [Google's Engineering Practices](https://google.github.io/eng-practices/).

The system combines:
- Modern transformer optimizations
- Persistent hierarchical memory (Dreamer)
- Adaptive computation
- Repo-level cross-attention (Architect)
- A native multi-agent orchestration layer

The goal is **continuous improvement of code health** rather than one-shot benchmark chasing.

---

## Key Features

### CogForge Architecture Highlights

- **Grouped-Query Attention (GQA)** with sliding window (local efficiency)
- **Rotary Positional Embeddings (RoPE)**
- **True O(T) Linear Lookback Attention** – recurrent formulation for long-range context (fixed memory)
- **Hierarchical Memory Module** – persistent external memory with gated updates and cross-attention reads
- **Adaptive Computation Time (ACT)** blocks – variable compute per token for hard reasoning
- **Dynamic Latent Reasoning Tokens** – automatically expands thinking budget on complex tasks
- **Architect Cross-Attention** – repo-level context injection at layers 4, 8, and 10
- **SwiGLU** feed-forward + **RMSNorm** throughout
- **Verifier Head** – learned correctness scoring
- **Handoff Projection Head** – recommends next agent in the swarm

**Parameter count**: ~100M (easily scalable)

### CogWorks Swarm Agents

| Agent                  | Role |
|------------------------|------|
| **Coordinator**        | Task routing, quality gates, orchestration |
| **Dreamer**            | Hierarchical memory management & consolidation |
| **Explorer**           | Repository mapping, complexity analysis, flagging |
| **Planner**            | Task decomposition into DAGs |
| **ProblemSolver**      | Deep reasoning over resources and constraints |
| **Engineer**           | Refactoring toward clean, efficient, idiomatic code |
| **BugFinder**          | Logical error detection with pattern-based scanning |
| **VulnerabilityFinder**| Security vulnerability scanning (CWE-aware) |
| **Pessimist**          | Devil's advocate – stress testing plans & code |
| **TerminalGuy**        | Sandboxed shell execution with safety checks |
| **Documentor**         | Automatic docstring and comment generation |

---

## How It Works

### 1. Memory System (The "Not Forgetting" Solution)

- **HierarchicalMemory**: 128 learnable memory slots
- **Update mechanism**: Gated additive updates with decay on recent hidden states (every ~256 tokens)
- **Read mechanism**: Lightweight cross-attention from current hidden states to memory slots
- **Consolidation**: Periodic summarization into dedicated summary slot (managed by Dreamer)
- Injected into the last `n_memory_layers` transformer blocks

This gives the model effective **infinite context** for repository-scale work while staying computationally efficient.

### 2. Swarm Orchestration Flow

Typical workflow for a user task:

1. **Coordinator** receives task → delegates to **Planner** + **Explorer** in parallel
2. **Dreamer** injects relevant episodic + repo context
3. **Planner** produces a DAG of subtasks
4. **ProblemSolver** annotates the DAG with resource-aware strategy
5. **Engineer** → **Pessimist** (critique) → **BugFinder** + **VulnerabilityFinder** (review)
6. **TerminalGuy** executes tests/commands
7. **Documentor** adds documentation
8. **Coordinator** applies quality gate using verifier proxy score
9. **Dreamer** consolidates learnings into long-term memory

All communication happens via **HandoffMessage** objects logged in `SharedMemoryStore`.

### 3. Google Engineering Practices Integration

The swarm is explicitly designed around Google’s code review philosophy:
- Prioritize **overall code health** over perfection
- Favor small, incremental improvements
- Focus reviews on **design, functionality, complexity, tests, naming, comments, style**
- Be kind and explain reasoning

This bias is baked into agent behavior (especially Engineer, BugFinder, Pessimist, Documentor, and Coordinator).

---

## Technical Architecture Details

### CogForge Forward Pass

```python
x = embed(input_ids)
x = prepend_latent_tokens(x)           # dynamic expansion possible
for each block:
    local GQA
    linear lookback (O(T) recurrent)
    optional hierarchical memory read (cross-attn)
    optional Architect repo cross-attn
    SwiGLU FFN
    (ACT wrapping on deeper layers)
memory.update() → memory.consolidate()
logits = lm_head(x)
