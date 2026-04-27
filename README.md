# CogForge + CogWorks Swarm

**A memory-augmented coding & reasoning transformer with a native multi-agent software engineering team and execution-guided search.**

Designed from the ground up to **digest massive codebases without forgetting** while producing high-quality, maintainable, Google-style code through structured collaboration and adaptive test-time compute.

---

## Overview

**CogForge** is a ~100M parameter transformer optimized for long-context coding and deep reasoning.

**CogWorks** is a team of 13 specialized agents that work together using clean handoffs, shared hierarchical memory, and principled engineering practices.

**CogSearch** is an execution-guided Monte Carlo Tree Search (MCTS) layer that enables systematic exploration of code solutions with internal (VerifierHead) and external (TerminalGuy) evaluation.

The entire system is heavily influenced by **Google's Engineering Practices**, emphasizing continuous improvement of code health over one-shot perfection.

---

## Key Features

### CogForge Architecture

- Grouped-Query Attention (GQA) with Sliding Window
- Rotary Positional Embeddings (RoPE)
- True O(T) Linear Lookback Attention (recurrent formulation)
- **Hierarchical Memory Module** (Dreamer-managed persistent memory with gated updates + cross-attention reads)
- Adaptive Computation Time (ACT) blocks with dynamic latent reasoning tokens
- Architect Cross-Attention for repo-level context injection
- SwiGLU + RMSNorm
- VerifierHead for learned correctness scoring
- Handoff projection head for swarm coordination

### CogSearch – Execution-Guided MCTS

- UCT-based node selection (exploration vs exploitation)
- Multi-branch expansion driven by ProblemSolver
- Two-level evaluation:
  - Level 1: VerifierHead (fast internal scoring)
  - Level 2: TerminalGuy (compilation + pytest execution)
- Reward model: syntax error (-1.0), logic flaw (0.1), passes tests (1.0)
- Automatic DPO pair collection for future preference fine-tuning
- Backpropagation of rewards through the code-state tree

### CogWorks Swarm Agents (13 total)

| Agent                | Role |
|----------------------|------|
| **Coordinator**      | Task routing, quality gates, orchestration |
| **Dreamer**          | Hierarchical memory management & consolidation |
| **Explorer**         | Repository mapping + complexity flagging |
| **Archeologist**     | Git history & temporal intent analysis (volatile/stale/caution zones) |
| **Nexus**            | Dependency auditing, API RAG cache, hallucination detection |
| **Planner**          | Task decomposition into DAGs |
| **ProblemSolver**    | Deep expert reasoning + MCTS expansion |
| **Engineer**         | Clean, efficient refactoring |
| **BugFinder**        | Logical error detection |
| **VulnerabilityFinder** | Security scanning (CWE-aware) |
| **Pessimist**        | Devil's advocate & stress testing |
| **TerminalGuy**      | Sandboxed command execution (MCTS oracle) |
| **Documentor**       | Docstrings, comments, and README generation |

**New capability**: **Agent Group Cloning** — Agents can dynamically clone themselves (exact same weights) for extra-long or high-stakes reasoning tasks, run in parallel, and merge results via verifier scoring and Dreamer consolidation.

---

## How It Works

### Memory System (Core "No Forgetting" Mechanism)

- **HierarchicalMemory**: 128 learnable slots with gated additive updates and cross-attention reads
- Periodic updates from recent hidden states + full-context consolidation
- Injected into deeper transformer blocks
- Managed centrally by the **Dreamer** agent

This gives the swarm effective **infinite context** for large repositories.

### Swarm Workflow

1. **Coordinator** receives task → launches **Planner** + **Explorer** in parallel
2. **Archeologist** builds temporal map from git history (flags volatile/stale/caution zones)
3. **Nexus** audits dependencies and builds RAG cache for APIs
4. **Dreamer** injects relevant memory context
5. **Planner** creates DAG → **ProblemSolver** refines strategy
6. **Engineer** → **Pessimist** critique → **BugFinder** + **VulnerabilityFinder** review
7. **TerminalGuy** runs tests/commands
8. **Documentor** adds documentation
9. **Coordinator** applies quality gate
10. **Dreamer** consolidates learnings

For hard reasoning tasks, agents can **self-clone** into groups (e.g. 4–8 identical instances) to explore diverse paths in parallel.

### CogSearch MCTS

When deep code generation is needed, **ProblemSolver** can invoke CogSearch:
- Selects promising partial solutions using UCT
- Expands branches with diverse strategies
- Evaluates with VerifierHead + actual execution
- Backpropagates rewards
- Produces best code + DPO training pairs

---

## Technical Highlights

- **True linear-time long-range attention** via recurrent Linear Lookback
- **Dynamic latent tokens** that expand on complex tasks
- **Agent cloning / group reasoning** for test-time compute scaling
- **Execution-guided search** (CogSearch) combining neural scoring and real execution
- **Temporal awareness** via Archeologist (git history intent)
- **Dependency safety** via Nexus RAG cache
- Structured, observable communication via `HandoffMessage`
- Google Engineering Practices baked into agent behavior

---

## Project Structure
cogforge/
├── config.py
├── model.py                 # CogForge transformer
├── memory.py
├── mcts.py                  # CogSearch implementation
├── agents/
│   ├── base.py
│   ├── coordinator.py
│   ├── dreamer.py
│   ├── archeologist.py
│   ├── nexus.py
│   ├── problem_solver.py
│   └── ...
├── swarm.py                 # CogWorksSwarm orchestrator
├── utils.py
└── main.py
text---

## Usage

```python
from cogforge.model import CogForge, CogForgeConfig
from cogforge.swarm import CogWorksSwarm

config = CogForgeConfig()
model = CogForge(config)
swarm = CogWorksSwarm(model=model, config=config)

# Full swarm pipeline
results = swarm.run(
    task="Refactor the authentication module for better security and testability",
    repo_root="."
)

# Or use CogSearch directly for hard generation tasks
search_result = swarm.cog_search(
    prompt="Write a fast prime-checking function with early exit optimization",
    iterations=12,
    run_tests=True,
    k_expansions=4
)

print("Best code reward:", search_result["best_reward"])
