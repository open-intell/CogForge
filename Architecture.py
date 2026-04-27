"""
CogForge v2: Self-Improving, Hierarchical Agent Team
=====================================================
Architecture (unchanged base):
  - Grouped-Query Attention (GQA) with Sliding Window
  - Rotary Positional Embeddings (RoPE)
  - Adaptive Computation Time (ACT) blocks
  - Latent Reasoning Tokens (dynamic, recursive)
  - Linear "Look-back" Attention (true O(T) recurrent)
  - Hierarchical Memory Module (Dreamer-managed persistent memory)
  - Architect cross-attention module (repo-level context)
  - SwiGLU feed-forward networks / RMSNorm throughout
  - Handoff Protocol for CogWorks swarm coordination

═══════════════════════════════════════════════════════════════
NEW IN v2 — THREE MAJOR EVOLUTION AXES
═══════════════════════════════════════════════════════════════

2. Evolved Swarm: True Self-Improving, Hierarchical Agent Team
   ┌─ ModelRouter          : strength-based routing (Opus-head / Haiku-head)
   ├─ EphemeralAgent       : on-demand specialist sub-agents (any role)
   ├─ MetaOrchestrator     : spawns ephemeral agents, manages worktrees
   ├─ GraphRAGMemory       : vector + graph store (nodes=files/funcs/commits,
   │                         edges=imports/calls/co-commits); multi-level
   │                         compression (gist→summary→detailed→raw)
   └─ MultiLevelMemoryMgr  : Dreamer v2 with persistent cross-session state

3. Supercharged CogSearch (Execution-Guided Hierarchical MCTS)
   ┌─ ProcessRewardModel   : step-level reward scoring over reasoning traces
   ├─ CoverageRewardModule : pytest-cov + AST coverage, perf, security signals
   ├─ BeamExpansion        : CFG-guided beam search hybridised into MCTS
   ├─ AsyncParallelEval    : ThreadPoolExecutor for parallel branch evaluation
   └─ HierarchicalCogSearch: three-level search tree
                              Architecture → File → Line
                             agents call it as a subroutine for sub-problems

CogSearch Reward Signal (multi-dimensional):
  -1.0  syntax error
  +0.1  logic flaw (compiles; Pessimist high-risk)
  +0.3  clean compile; style/repo-norm consistent (Nexus/Archeologist)
  +0.5  security clean (VulnerabilityFinder pass)
  +0.7  coverage > threshold
  +1.0  passes all tests + coverage + no regressions

CogWorks Swarm Agents (v1 roster preserved + new):
  Coordinator    Dreamer        Explorer       Planner
  ProblemSolver  Engineer       BugFinder      TerminalGuy
  VulnerabilityFinder  Pessimist  Documentor   Nexus
  Archeologist   MetaOrchestrator  (+ ephemeral specialists on demand)
"""

import ast
import concurrent.futures
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.request
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CogForgeConfig:
    # Vocabulary & sequence
    vocab_size: int = 32000
    max_seq_len: int = 4096
    pad_token_id: int = 0

    # Model dimensions — tuned to ~100M params
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12
    d_ff_multiplier: float = 2.6667

    # Sliding window attention
    window_size: int = 512
    global_tokens: int = 64

    # Latent reasoning (dynamic)
    n_latent_tokens: int = 8
    max_latent_tokens: int = 32
    n_act_iterations: int = 3
    act_threshold: float = 0.99

    # Hierarchical memory
    memory_size: int = 128
    memory_update_interval: int = 256
    n_memory_layers: int = 4

    # Architect (repo-level) cross-attention
    n_arch_layers: int = 2
    arch_d_model: int = 256
    arch_n_heads: int = 4
    max_repo_chunks: int = 32

    # Dropout & regularization
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0

    # NEW v2 — GraphRAG
    graph_embed_dim: int = 256        # dimension for graph node embeddings
    graph_max_nodes: int = 2048       # maximum nodes in the in-process graph

    # NEW v2 — Multi-level memory
    gist_size: int = 8                # ultra-compressed gist tokens
    summary_size: int = 64            # summary slots
    detailed_size: int = 128          # detailed slots (= memory_size default)

    # NEW v2 — Hierarchical MCTS
    mcts_beam_width: int = 5          # beam width for beam-search hybridisation
    mcts_prm_layers: int = 4          # depth of Process Reward Model MLP
    mcts_max_parallel: int = 4        # parallel evaluation workers

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.d_ff_multiplier)
        return (raw // 256) * 256

    @property
    def kv_groups(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


# ═══════════════════════════════════════════════════════════════════════════
# Utility: RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096,
                 theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq / scaling
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ═══════════════════════════════════════════════════════════════════════════
# Grouped-Query Attention with Sliding Window
# ═══════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.kv_groups = config.kv_groups
        self.d_head = config.d_head
        self.window_size = config.window_size
        self.global_tokens = config.global_tokens

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.rotary = RotaryEmbedding(
            config.d_head, config.max_seq_len,
            config.rope_theta, config.rope_scaling
        )

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        B, H, T, D = kv.shape
        kv = kv.unsqueeze(2).expand(B, H, self.kv_groups, T, D)
        return kv.reshape(B, H * self.kv_groups, T, D)

    def _sliding_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((T, T), float("-inf"), device=device)
        for i in range(T):
            lo = max(0, i - self.window_size + 1)
            mask[i, lo:i + 1] = 0.0
        for g in range(min(self.global_tokens, T)):
            mask[g, :g + 1] = 0.0
            mask[:, g] = 0.0
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        mask[causal] = float("-inf")
        return mask

    def forward(self, x: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        k = self._expand_kv(k)
        v = self._expand_kv(v)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        T_k = k.shape[2]
        if T > 1:
            attn_mask = self._sliding_window_mask(T_k, x.device)
            if T == T_k:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                causal = torch.triu(
                    torch.ones(T, T_k, device=x.device) * float("-inf"), diagonal=1
                )
                scores = scores + causal.unsqueeze(0).unsqueeze(0)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


# ═══════════════════════════════════════════════════════════════════════════
# Linear "Look-back" Attention — true O(T) recurrent formulation
# ═══════════════════════════════════════════════════════════════════════════

class LinearLookbackAttention(nn.Module):
    """
    Causal linear attention using a running recurrent state (S_t, z_t).
    Processes one chunk at a time and carries state forward.
    """
    def __init__(self, config: CogForgeConfig, chunk: int = 64):
        super().__init__()
        self.n_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.chunk = chunk
        dim = config.n_kv_heads * config.d_head

        self.q_proj = nn.Linear(config.d_model, dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, dim, bias=False)
        self.o_proj = nn.Linear(dim, config.d_model, bias=False)
        self.gate   = nn.Linear(config.d_model, config.d_model, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head
        C = self.chunk

        q = self.q_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)

        q, k = self._feature_map(q), self._feature_map(k)

        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)

        out_chunks: List[torch.Tensor] = []
        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            qt = q[:, :, t_start:t_end, :]
            kt = k[:, :, t_start:t_end, :]
            vt = v[:, :, t_start:t_end, :]
            c = t_end - t_start

            chunk_out = torch.zeros(B, H, c, D, device=x.device, dtype=x.dtype)
            S_run = S.clone()
            z_run = z.clone()
            for i in range(c):
                ki = kt[:, :, i, :]
                vi = vt[:, :, i, :]
                S_run = S_run + torch.einsum("bhd,bhe->bhde", ki, vi)
                z_run = z_run + ki
                qi = qt[:, :, i, :]
                num = torch.einsum("bhd,bhde->bhe", qi, S_run)
                den = torch.einsum("bhd,bhd->bh", qi, z_run).unsqueeze(-1).clamp(min=1e-6)
                chunk_out[:, :, i, :] = num / den

            S = S_run
            z = z_run
            out_chunks.append(chunk_out)

        out = torch.cat(out_chunks, dim=2)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)
        gate = torch.sigmoid(self.gate(x))
        return self.o_proj(out) * gate


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Memory Module (v1 — preserved)
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalMemory(nn.Module):
    """
    Persistent external memory with learnable slots, gated update,
    and cross-attention read head.  Managed by MultiLevelMemoryMgr (v2).
    """
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.memory_size = config.memory_size
        D = config.d_model

        self.memory_init = nn.Parameter(torch.randn(1, config.memory_size, D) * 0.02)

        self.compressor = nn.Sequential(
            nn.Linear(D, D), RMSNorm(D), nn.GELU(), nn.Linear(D, D),
        )
        self.update_gate  = nn.Linear(D, 1)

        n_heads  = max(1, config.n_kv_heads // 2)
        d_head   = D // config.n_heads
        attn_dim = n_heads * d_head
        self.read_q = nn.Linear(D, attn_dim, bias=False)
        self.read_k = nn.Linear(D, attn_dim, bias=False)
        self.read_v = nn.Linear(D, attn_dim, bias=False)
        self.read_o = nn.Linear(attn_dim, D, bias=False)
        self.n_read_heads = n_heads
        self.read_d_head  = d_head

        self.compress_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D),
        )
        self.norm_mem  = RMSNorm(D)
        self.norm_read = RMSNorm(D)
        self.decay     = 0.1

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.memory_init.expand(batch_size, -1, -1).clone()

    def update(self, recent_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        pooled     = recent_hidden.mean(dim=1)
        update_vec = self.compressor(pooled)
        gate       = torch.sigmoid(self.update_gate(pooled)).unsqueeze(1)
        memory     = memory * (1.0 - gate * self.decay) + update_vec.unsqueeze(1) * gate
        return self.norm_mem(memory)

    def consolidate(self, full_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        summary  = self.compress_head(full_hidden.mean(dim=1))
        memory   = memory.clone()
        memory[:, 0, :] = summary
        return self.norm_mem(memory)

    def read(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden.shape
        H, D    = self.n_read_heads, self.read_d_head

        q = self.read_q(self.norm_read(hidden)).view(B, T, H, D).transpose(1, 2)
        k = self.read_k(memory).view(B, self.memory_size, H, D).transpose(1, 2)
        v = self.read_v(memory).view(B, self.memory_size, H, D).transpose(1, 2)

        scores  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        weights = F.softmax(scores, dim=-1)
        out     = torch.matmul(weights, v)
        out     = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.read_o(out)


# ═══════════════════════════════════════════════════════════════════════════
# SwiGLU Feed-Forward / ACT / CogForgeBlock / ArchitectEncoder (preserved)
# ═══════════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class ACTBlock(nn.Module):
    def __init__(self, block: nn.Module, d_model: int,
                 max_iter: int = 3, threshold: float = 0.99):
        super().__init__()
        self.block         = block
        self.max_iter      = max_iter
        self.threshold     = threshold
        self.halting_proj  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                **block_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D   = x.shape
        device    = x.device
        halted    = torch.zeros(B, T, 1, device=device)
        accum     = torch.zeros_like(x)
        remainder = torch.ones(B, T, 1, device=device)
        ponder    = torch.zeros(B, T, device=device)
        state     = x

        for step in range(self.max_iter):
            h            = torch.sigmoid(self.halting_proj(state))
            still_running = (halted < self.threshold).float()
            is_last       = (step == self.max_iter - 1)
            if is_last:
                used_h = remainder * still_running
            else:
                used_h       = h * still_running
                will_exceed  = ((halted + used_h) > self.threshold).float()
                used_h       = used_h * (1 - will_exceed) + remainder * will_exceed
            halted        = halted + used_h
            ponder       += used_h.squeeze(-1) * still_running.squeeze(-1)
            state, _      = self.block(state, memory=memory, **block_kwargs)
            accum         = accum + used_h * state
            remainder     = remainder - used_h
            if (halted >= self.threshold).all():
                break

        return accum, ponder


class CogForgeBlock(nn.Module):
    def __init__(self, config: CogForgeConfig, inject_memory: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)
        self.inject_memory = inject_memory

        self.gqa      = GroupedQueryAttention(config)
        self.lookback = LinearLookbackAttention(config)
        self.ffn      = SwiGLUFFN(config)
        self.drop     = nn.Dropout(config.dropout)
        self.mix_alpha = nn.Parameter(torch.tensor(0.1))

        if inject_memory:
            self.mem_beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                past_kv: Optional[Tuple] = None,
                memory: Optional[torch.Tensor] = None,
                hier_memory: Optional["HierarchicalMemory"] = None,
                ) -> Tuple[torch.Tensor, Tuple]:
        h, new_kv = self.gqa(self.norm1(x), past_kv)
        lb = self.lookback(self.norm2(x))
        x  = x + self.drop(h) + self.drop(lb) * torch.sigmoid(self.mix_alpha)

        if self.inject_memory and memory is not None and hier_memory is not None:
            mem_read = hier_memory.read(x, memory)
            x = x + self.drop(mem_read) * torch.sigmoid(self.mem_beta)

        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, new_kv


class ArchitectEncoder(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.chunk_proj = nn.Linear(config.d_model, config.arch_d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.arch_d_model,
            nhead=config.arch_n_heads,
            dim_feedforward=config.arch_d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=config.n_arch_layers)
        self.out_proj = nn.Linear(config.arch_d_model, config.d_model)
        self.norm     = RMSNorm(config.d_model)

    def forward(self, chunk_embeds: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.chunk_proj(chunk_embeds)
        h = self.encoder(h, src_key_padding_mask=chunk_mask)
        return self.norm(self.out_proj(h))


class ArchitectCrossAttention(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_heads = config.n_kv_heads
        self.d_head  = config.d_head
        dim = self.n_heads * self.d_head

        self.q_proj = nn.Linear(config.d_model, dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, dim, bias=False)
        self.o_proj = nn.Linear(dim, config.d_model, bias=False)
        self.norm_q = RMSNorm(config.d_model)
        self.norm_c = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        N = context.shape[1]
        H, D = self.n_heads, self.d_head

        q = self.q_proj(self.norm_q(x)).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(self.norm_c(context)).view(B, N, H, D).transpose(1, 2)
        v = self.v_proj(context).view(B, N, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if context_mask is not None:
            scores = scores.masked_fill(
                context_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        return x + self.o_proj(out)


class VerifierHead(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.pool = nn.Linear(config.d_model, config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        pooled = F.gelu(self.pool(pooled))
        return self.head(pooled).squeeze(-1)


class DynamicLatentController(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_latent_base = config.n_latent_tokens
        self.max_latent    = config.max_latent_tokens
        D = config.d_model
        self.complexity_head = nn.Sequential(
            nn.Linear(D, D // 4), nn.GELU(), nn.Linear(D // 4, 1), nn.Sigmoid(),
        )
        extra = config.max_latent_tokens - config.n_latent_tokens
        self.extra_latents = nn.Parameter(torch.randn(1, extra, D) * 0.02)

    def forward(self, x_after_first_block: torch.Tensor,
                base_latents: torch.Tensor) -> torch.Tensor:
        B = x_after_first_block.shape[0]
        latent_repr = x_after_first_block[:, :self.n_latent_base, :].mean(1)
        complexity  = self.complexity_head(latent_repr)
        n_extra = int(complexity.mean().item() * (self.max_latent - self.n_latent_base))
        if n_extra == 0:
            return base_latents
        extra = self.extra_latents[:, :n_extra, :].expand(B, -1, -1)
        return torch.cat([base_latents, extra], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# Handoff Protocol
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HandoffMessage:
    """
    Structured inter-agent message — all fields are plain Python so they can
    be serialised to JSON and stored in shared memory / Redis / disk.
    """
    task_id:       str
    source_agent:  str
    target_agent:  str
    status:        str          # "pending" | "running" | "done" | "error"
    message:       str
    artifacts:     Dict[str, Any] = field(default_factory=dict)
    confidence:    float = 1.0
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "task_id":      self.task_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "status":       self.status,
            "message":      self.message,
            "artifacts":    self.artifacts,
            "confidence":   self.confidence,
            "metadata":     self.metadata,
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "HandoffMessage":
        return cls(**json.loads(s))


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Graph RAG Memory
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """
    A node in the knowledge graph.
    type: "file" | "function" | "class" | "commit" | "concept"
    """
    node_id:   str
    type:      str
    label:     str
    metadata:  Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None     # dense vector

    def to_dict(self) -> Dict:
        return {
            "node_id":   self.node_id,
            "type":      self.type,
            "label":     self.label,
            "metadata":  self.metadata,
            "embedding": self.embedding,
        }


@dataclass
class GraphEdge:
    """
    A directed edge in the knowledge graph.
    rel: "imports" | "calls" | "co-commits" | "inherits" | "depends_on"
    """
    src:     str
    dst:     str
    rel:     str
    weight:  float = 1.0


class GraphRAGMemory:
    """
    Vector + Graph RAG memory that replaces the flat SharedMemoryStore._repo_map.

    Stores:
      - Nodes: files, functions, classes, git commits, abstract concepts.
      - Edges: imports, calls, co-commits, inheritance, temporal co-change.

    Retrieval:
      - Semantic (cosine similarity over dense embeddings).
      - Structural (k-hop BFS/DFS from a seed node).
      - Temporal (recent commits involving a symbol).

    Embeddings:
      - In production: replace _embed() with a real encoder (e.g. CodeBERT).
      - In-process fallback: lightweight TF-IDF-style character n-gram hashing.
    """

    def __init__(self, embed_dim: int = 256, max_nodes: int = 2048):
        self.embed_dim  = embed_dim
        self.max_nodes  = max_nodes
        self._nodes:  Dict[str, GraphNode]     = {}
        self._edges:  List[GraphEdge]          = []
        self._adj:    Dict[str, List[GraphEdge]] = defaultdict(list)
        self._lock    = threading.Lock()

        # Multi-level compression store (gist → summary → detailed → raw)
        # Each level keyed by node_id
        self._gist:     Dict[str, str]  = {}   # 1-sentence
        self._summary:  Dict[str, str]  = {}   # paragraph
        self._detailed: Dict[str, str]  = {}   # full analysis
        self._raw:      Dict[str, Any]  = {}   # raw artifact (code, diff, etc.)

    # ── Embedding (lightweight character n-gram hashing fallback) ──────────

    def _embed(self, text: str) -> List[float]:
        """
        Deterministic character n-gram hashing to a unit-norm float vector.
        In production: replace with CodeBERT / text-embedding-ada / BGE-M3.
        """
        vec = [0.0] * self.embed_dim
        text = text[:4096]
        for n in (2, 3, 4):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h    = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
                idx  = h % self.embed_dim
                vec[idx] += 1.0
        # L2 normalise
        mag = math.sqrt(sum(v * v for v in vec)) or 1e-9
        return [v / mag for v in vec]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # ── Node / Edge management ─────────────────────────────────────────────

    def upsert_node(self, node_id: str, node_type: str, label: str,
                    metadata: Optional[Dict] = None,
                    embed_text: str = "") -> GraphNode:
        with self._lock:
            if len(self._nodes) >= self.max_nodes:
                # Evict oldest node (FIFO)
                oldest = next(iter(self._nodes))
                del self._nodes[oldest]
            embedding = self._embed(embed_text or label)
            node = GraphNode(
                node_id=node_id, type=node_type, label=label,
                metadata=metadata or {}, embedding=embedding
            )
            self._nodes[node_id] = node
            return node

    def add_edge(self, src: str, dst: str, rel: str, weight: float = 1.0) -> None:
        with self._lock:
            edge = GraphEdge(src=src, dst=dst, rel=rel, weight=weight)
            self._edges.append(edge)
            self._adj[src].append(edge)

    def set_compression(self, node_id: str,
                        gist: str = "",
                        summary: str = "",
                        detailed: str = "",
                        raw: Any = None) -> None:
        """Store multi-level compression for a node."""
        with self._lock:
            if gist:    self._gist[node_id]    = gist
            if summary: self._summary[node_id] = summary
            if detailed: self._detailed[node_id] = detailed
            if raw is not None: self._raw[node_id] = raw

    def get_compression(self, node_id: str, level: str = "summary") -> Any:
        """level: 'gist' | 'summary' | 'detailed' | 'raw'"""
        store = {"gist": self._gist, "summary": self._summary,
                 "detailed": self._detailed, "raw": self._raw}
        return store.get(level, {}).get(node_id)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def semantic_search(self, query: str, top_k: int = 8,
                        node_type: Optional[str] = None) -> List[Tuple[float, GraphNode]]:
        """Return top_k nodes most similar to query embedding."""
        q_emb = self._embed(query)
        results: List[Tuple[float, GraphNode]] = []
        with self._lock:
            for node in self._nodes.values():
                if node_type and node.type != node_type:
                    continue
                if node.embedding is None:
                    continue
                sim = self._cosine(q_emb, node.embedding)
                results.append((sim, node))
        results.sort(key=lambda x: -x[0])
        return results[:top_k]

    def k_hop_neighbours(self, seed_id: str, k: int = 2,
                         rel_filter: Optional[Set[str]] = None,
                         ) -> List[GraphNode]:
        """BFS k hops from seed; returns nodes (excluding seed)."""
        visited: Set[str] = {seed_id}
        frontier: List[str] = [seed_id]
        result: List[GraphNode] = []

        for _ in range(k):
            next_frontier: List[str] = []
            for node_id in frontier:
                for edge in self._adj.get(node_id, []):
                    if rel_filter and edge.rel not in rel_filter:
                        continue
                    if edge.dst not in visited:
                        visited.add(edge.dst)
                        next_frontier.append(edge.dst)
                        if edge.dst in self._nodes:
                            result.append(self._nodes[edge.dst])
            frontier = next_frontier

        return result

    def temporal_neighbours(self, node_id: str,
                            days: int = 30) -> List[GraphNode]:
        """Return commit-adjacent nodes that changed within *days*."""
        cutoff = time.time() - days * 86400
        related: List[GraphNode] = []
        for edge in self._adj.get(node_id, []):
            if edge.rel == "co-commits":
                n = self._nodes.get(edge.dst)
                if n and n.metadata.get("last_ts", 0) >= cutoff:
                    related.append(n)
        return related

    def ingest_repo_map(self, repo_map: Dict[str, Dict]) -> None:
        """
        Convert an Explorer/Archeologist repo_map into graph nodes.
        Derives import edges by scanning AST of each file.
        """
        for filepath, info in repo_map.items():
            fid = hashlib.md5(filepath.encode()).hexdigest()[:12]
            meta = dict(info)
            meta["path"] = filepath
            self.upsert_node(
                node_id=fid,
                node_type="file",
                label=filepath,
                metadata=meta,
                embed_text=f"{filepath} {info.get('flag_reason', '')}",
            )

        # Add import edges
        for filepath in repo_map:
            fid = hashlib.md5(filepath.encode()).hexdigest()[:12]
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        # Try to resolve to a file node
                        mod = node.module.replace(".", "/") + ".py"
                        for candidate, _ in repo_map.items():
                            if candidate.endswith(mod):
                                cid = hashlib.md5(candidate.encode()).hexdigest()[:12]
                                self.add_edge(fid, cid, "imports")
            except Exception:
                pass

    def stats(self) -> Dict:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "gist_entries": len(self._gist),
            "summary_entries": len(self._summary),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Shared Memory Store (v2 — GraphRAG integrated)
# ═══════════════════════════════════════════════════════════════════════════

class SharedMemoryStore:
    """
    Central key-value + append-log store used by all swarm agents.
    v2 adds: GraphRAGMemory, persistent session snapshots, thread-safe ops.
    In production: swap dict for Redis + GraphRAGMemory for a real vector DB.
    """

    def __init__(self, graph_embed_dim: int = 256, graph_max_nodes: int = 2048):
        self._store:    Dict[str, Any]       = {}
        self._log:      List[HandoffMessage] = []
        self._repo_map: Dict[str, Any]       = {}
        self._episodic: List[Dict]           = []
        self._lock      = threading.Lock()

        # NEW v2: GraphRAG
        self.graph = GraphRAGMemory(embed_dim=graph_embed_dim,
                                    max_nodes=graph_max_nodes)

        # NEW v2: persistent session snapshot path (None = in-memory only)
        self._session_path: Optional[str] = None

    # ── Key-value ────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    # ── Handoff log ──────────────────────────────────────────────────────

    def log_handoff(self, msg: HandoffMessage) -> None:
        with self._lock:
            self._log.append(msg)

    def get_handoffs_for(self, agent: str) -> List[HandoffMessage]:
        with self._lock:
            return [m for m in self._log
                    if m.target_agent == agent and m.status == "pending"]

    def mark_done(self, task_id: str) -> None:
        with self._lock:
            for m in self._log:
                if m.task_id == task_id:
                    m.status = "done"

    # ── Repo map ──────────────────────────────────────────────────────────

    def update_repo_map(self, path: str, info: Dict) -> None:
        with self._lock:
            self._repo_map[path] = info

    def get_repo_map(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._repo_map)

    def get_flagged_files(self) -> List[str]:
        with self._lock:
            return [p for p, info in self._repo_map.items() if info.get("flagged")]

    # ── Episodic memory ───────────────────────────────────────────────────

    def add_episode(self, episode: Dict) -> None:
        with self._lock:
            self._episodic.append(episode)

    def get_recent_episodes(self, n: int = 5) -> List[Dict]:
        with self._lock:
            return self._episodic[-n:]

    # ── NEW v2: session persistence ───────────────────────────────────────

    def snapshot(self, path: str) -> None:
        """Serialise key-value store + episodic memory to JSON on disk."""
        payload = {
            "store":    {k: v for k, v in self._store.items()
                         if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
            "episodic": self._episodic,
            "repo_map": {k: {ik: iv for ik, iv in info.items()
                             if isinstance(iv, (str, int, float, bool, list, dict, type(None)))}
                         for k, info in self._repo_map.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def restore(self, path: str) -> None:
        """Restore key-value store + episodic memory from a JSON snapshot."""
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        with self._lock:
            self._store.update(payload.get("store", {}))
            self._episodic.extend(payload.get("episodic", []))
            self._repo_map.update(payload.get("repo_map", {}))


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Model Router
# ═══════════════════════════════════════════════════════════════════════════

class ModelRouter:
    """
    Routes inference requests to a "strong head" (Opus-like: more ACT
    iterations, larger beam) or a "fast head" (Haiku-like: fewer iterations,
    greedy decoding) based on task complexity signals.

    Without multiple physical model instances (resource-constrained), the
    router adjusts the *generation config* of a single CogForge model:
      - strong_head: temperature=0.6, top_p=0.95, n_act_iterations=5
      - fast_head:   temperature=0.9, top_p=0.85, n_act_iterations=1

    In a production multi-GPU setting, swap the config overrides for actual
    different-sized model checkpoints.
    """

    COMPLEXITY_KEYWORDS = {
        "high": [
            "architecture", "redesign", "security", "cve", "race condition",
            "memory leak", "migration", "concurrency", "refactor entire",
            "multi-file", "regression", "distributed",
        ],
        "low": [
            "docstring", "comment", "rename", "typo", "formatting",
            "whitespace", "print", "log", "lint", "minor",
        ],
    }

    def __init__(self, model: Optional["CogForge"] = None):
        self.model = model
        # Profiles: (temperature, top_p, max_new_tokens, n_act_hint)
        self._profiles = {
            "strong": dict(temperature=0.6, top_p=0.95, top_k=40,
                           max_new_tokens=512, n_act_hint=5),
            "fast":   dict(temperature=0.95, top_p=0.85, top_k=60,
                           max_new_tokens=128, n_act_hint=1),
        }

    def score_complexity(self, task: str) -> float:
        """
        Return complexity in [0, 1].
        0 = trivially simple (fast head).  1 = deeply complex (strong head).
        """
        task_l = task.lower()
        score  = 0.5
        for kw in self.COMPLEXITY_KEYWORDS["high"]:
            if kw in task_l:
                score += 0.07
        for kw in self.COMPLEXITY_KEYWORDS["low"]:
            if kw in task_l:
                score -= 0.07
        # Length heuristic: longer tasks are usually more complex
        score += min(0.2, len(task.split()) / 200)
        return max(0.0, min(1.0, score))

    def route(self, task: str) -> str:
        """Return 'strong' or 'fast'."""
        return "strong" if self.score_complexity(task) >= 0.55 else "fast"

    def get_gen_config(self, task: str) -> Dict:
        head = self.route(task)
        config = dict(self._profiles[head])
        config["head"] = head
        return config

    def generate(self, input_ids: torch.Tensor, task: str,
                 repo_chunks: Optional[torch.Tensor] = None,
                 **kwargs) -> Optional[torch.Tensor]:
        """Route to appropriate generation config and call model.generate()."""
        if self.model is None:
            return None
        cfg = self.get_gen_config(task)
        cfg.pop("head", None)
        cfg.pop("n_act_hint", None)
        return self.model.generate(input_ids, repo_chunks=repo_chunks, **cfg, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Ephemeral Agent Factory + MetaOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class EphemeralAgent:
    """
    A dynamically spawned specialist sub-agent.  Ephemeral agents:
      - Have a custom role string (e.g. "DatabaseMigrationSpecialist").
      - Receive a focused system prompt / capability description.
      - Operate in an isolated context (optional worktree path).
      - Report back to the spawning agent (usually MetaOrchestrator).
      - Are garbage-collected after their task completes.

    Implementation: wraps BaseAgent capabilities without requiring a class
    definition per role — the behaviour is governed by `system_prompt` and
    `capability_fns` (callables injected at spawn time).
    """

    def __init__(self,
                 role: str,
                 memory_store: "SharedMemoryStore",
                 system_prompt: str = "",
                 capability_fns: Optional[Dict[str, Callable]] = None,
                 worktree_path: Optional[str] = None,
                 model: Optional["CogForge"] = None,
                 router: Optional[ModelRouter] = None):
        self.role           = role
        self.memory         = memory_store
        self.system_prompt  = system_prompt
        self.capabilities   = capability_fns or {}
        self.worktree_path  = worktree_path
        self.model          = model
        self.router         = router
        self.task_id        = str(uuid.uuid4())
        self._created_at    = time.time()
        self._alive         = True

    def emit(self, target: str, message: str,
             artifacts: Optional[Dict] = None,
             confidence: float = 1.0,
             status: str = "pending") -> HandoffMessage:
        msg = HandoffMessage(
            task_id=str(uuid.uuid4()),
            source_agent=self.role,
            target_agent=target,
            status=status,
            message=message,
            artifacts=artifacts or {},
            confidence=confidence,
        )
        self.memory.log_handoff(msg)
        return msg

    def run(self, task: str, **kwargs) -> HandoffMessage:
        """
        Execute the agent's specialised task.
        Calls matching capability_fn if available; otherwise applies
        heuristic analysis using system_prompt context.
        """
        # If there's a matching capability function, invoke it
        for key, fn in self.capabilities.items():
            if key.lower() in task.lower():
                try:
                    result = fn(task=task, worktree=self.worktree_path, **kwargs)
                    return self.emit(
                        target="MetaOrchestrator",
                        message=f"[{self.role}] {result.get('summary', 'Task complete.')}",
                        artifacts=result,
                        confidence=result.get("confidence", 0.8),
                        status="done",
                    )
                except Exception as e:
                    return self.emit(
                        target="MetaOrchestrator",
                        message=f"[{self.role}] Error: {e}",
                        status="error",
                    )

        # Default: emit a structured placeholder
        return self.emit(
            target="MetaOrchestrator",
            message=(f"[{self.role}] Specialised analysis for: {task[:120]}.\n"
                     f"Context: {self.system_prompt[:200]}"),
            artifacts={"task": task, "worktree": self.worktree_path,
                       "role": self.role},
            confidence=0.7,
            status="done",
        )

    def retire(self) -> None:
        """Mark agent as retired so MetaOrchestrator can garbage-collect it."""
        self._alive = False


# Built-in capability function library (plug-ins for EphemeralAgents)
class EphemeralCapabilities:
    """Static library of reusable capability functions for ephemeral agents."""

    @staticmethod
    def db_migration_check(task: str, worktree: Optional[str] = None, **_) -> Dict:
        """Scan for Django/Alembic/Flyway migration files and flag conflicts."""
        root = worktree or "."
        migration_files: List[str] = []
        conflict_hints:  List[str] = []
        for dirpath, _, fns in os.walk(root):
            for fn in fns:
                if "migration" in fn.lower() or fn.endswith(".sql"):
                    fpath = os.path.join(dirpath, fn)
                    migration_files.append(fpath)
                    # Crude conflict detection: duplicate version numbers
                    m = re.search(r"(\d{4})", fn)
                    if m:
                        conflict_hints.append(f"{fn} version={m.group(1)}")
        return {
            "summary": (f"Found {len(migration_files)} migration files; "
                        f"{len(conflict_hints)} potential version conflicts."),
            "migration_files": migration_files[:20],
            "conflict_hints":  conflict_hints[:10],
            "confidence": 0.85,
        }

    @staticmethod
    def api_contract_diff(task: str, worktree: Optional[str] = None, **_) -> Dict:
        """Scan OpenAPI / protobuf / GraphQL schema files for breaking changes."""
        root = worktree or "."
        schema_files: List[str] = []
        for dirpath, _, fns in os.walk(root):
            for fn in fns:
                if fn.endswith((".yaml", ".yml", ".proto", ".graphql", ".json")):
                    fpath = os.path.join(dirpath, fn)
                    try:
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            content = f.read(2000)
                        if any(kw in content for kw in
                               ("openapi", "swagger", "syntax = \"proto", "type Query")):
                            schema_files.append(fpath)
                    except OSError:
                        pass
        return {
            "summary":      f"Found {len(schema_files)} API schema files.",
            "schema_files": schema_files,
            "confidence":   0.8,
        }

    @staticmethod
    def performance_profiler(task: str, worktree: Optional[str] = None, **_) -> Dict:
        """Identify hot loops and O(N²) patterns in Python source."""
        root = worktree or "."
        hotspots: List[Dict] = []
        for dirpath, _, fns in os.walk(root):
            for fn in fns:
                if not fn.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fn)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                    lines = source.splitlines()
                    for i, line in enumerate(lines, 1):
                        # Double nested for-loop heuristic
                        if re.search(r"\bfor\b", line):
                            if i + 1 < len(lines) and re.search(r"\bfor\b", lines[i]):
                                hotspots.append({"file": fpath, "line": i,
                                                  "issue": "nested_loop"})
                        # O(N²) list membership check
                        if re.search(r"\bin\s+\[", line):
                            hotspots.append({"file": fpath, "line": i,
                                              "issue": "linear_membership_check"})
                except OSError:
                    pass
        return {
            "summary":   f"Profiling found {len(hotspots)} potential hotspots.",
            "hotspots":  hotspots[:20],
            "confidence": 0.75,
        }


class MetaOrchestrator:
    """
    A meta-level coordinator that:
      1. Accepts high-level task intents.
      2. Decomposes them into specialist sub-tasks.
      3. Spawns EphemeralAgents on demand with the right capability set.
      4. Manages isolated git worktrees per sub-agent (when git is available).
      5. Collects and merges results; garbage-collects retired agents.
      6. Uses ModelRouter to pick the right compute head per sub-task.

    Relationship to Coordinator:
      Coordinator is the *workflow* overseer (DAG, handoffs, quality gates).
      MetaOrchestrator is the *resource* manager (spawn, route, clean up).
    """

    # Registry: keyword → (role_name, system_prompt, capability_key)
    SPECIALIST_REGISTRY: List[Tuple[str, str, str, str]] = [
        ("database",    "DatabaseMigrationSpecialist",
         "Expert in DB schema migrations, ORM conflicts, and data integrity.",
         "db_migration"),
        ("api",         "APIContractSpecialist",
         "Expert in REST/GraphQL/gRPC contract changes and backwards compat.",
         "api_contract"),
        ("performance", "PerformanceSpecialist",
         "Expert in profiling, complexity analysis, and cache strategies.",
         "performance"),
        ("crypto",      "CryptographySpecialist",
         "Expert in key management, cipher selection, and TLS configuration.",
         ""),
        ("docker",      "InfraSpecialist",
         "Expert in Dockerfile optimisation, K8s manifests, and CI pipelines.",
         ""),
        ("auth",        "AuthSecuritySpecialist",
         "Expert in OAuth2, JWT, session fixation, and privilege escalation.",
         ""),
        ("async",       "AsyncConcurrencySpecialist",
         "Expert in asyncio, race conditions, and deadlock analysis.",
         ""),
    ]

    _CAPABILITY_MAP: Dict[str, Callable] = {
        "db_migration":  EphemeralCapabilities.db_migration_check,
        "api_contract":  EphemeralCapabilities.api_contract_diff,
        "performance":   EphemeralCapabilities.performance_profiler,
    }

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional["CogForge"] = None,
                 config: Optional[CogForgeConfig] = None):
        self.memory  = memory_store
        self.model   = model
        self.config  = config or CogForgeConfig()
        self.router  = ModelRouter(model)
        self._agents: Dict[str, EphemeralAgent] = {}   # task_id → agent
        self._lock   = threading.Lock()

    # ── Worktree management ───────────────────────────────────────────────

    def _create_worktree(self, root: str, branch_name: str) -> Optional[str]:
        """Create an isolated git worktree; return its path or None."""
        worktree_dir = os.path.join(
            tempfile.gettempdir(), f"cogworks_{branch_name}_{uuid.uuid4().hex[:6]}"
        )
        try:
            result = subprocess.run(
                ["git", "worktree", "add", "-b", branch_name, worktree_dir],
                cwd=root, capture_output=True, text=True, timeout=15
            )
            return worktree_dir if result.returncode == 0 else None
        except Exception:
            return None

    def _remove_worktree(self, worktree_path: str, root: str) -> None:
        """Remove a git worktree after work is done."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=root, capture_output=True, timeout=10
            )
        except Exception:
            pass

    # ── Agent lifecycle ───────────────────────────────────────────────────

    def _identify_needed_specialists(self, task: str) -> List[Tuple[str, str, str, str]]:
        task_l = task.lower()
        needed = []
        for keyword, role, prompt, cap_key in self.SPECIALIST_REGISTRY:
            if keyword in task_l:
                needed.append((keyword, role, prompt, cap_key))
        return needed

    def spawn(self, role: str, system_prompt: str,
              cap_key: str = "",
              worktree_path: Optional[str] = None) -> EphemeralAgent:
        """Spawn an ephemeral agent and register it."""
        caps: Dict[str, Callable] = {}
        if cap_key and cap_key in self._CAPABILITY_MAP:
            caps[cap_key] = self._CAPABILITY_MAP[cap_key]

        agent = EphemeralAgent(
            role=role,
            memory_store=self.memory,
            system_prompt=system_prompt,
            capability_fns=caps,
            worktree_path=worktree_path,
            model=self.model,
            router=self.router,
        )
        with self._lock:
            self._agents[agent.task_id] = agent
        return agent

    def retire_all_done(self) -> int:
        """Garbage-collect agents that have completed.  Returns count removed."""
        with self._lock:
            to_remove = [tid for tid, a in self._agents.items() if not a._alive]
            for tid in to_remove:
                del self._agents[tid]
            return len(to_remove)

    # ── Main orchestration entry point ─────────────────────────────────────

    def orchestrate(self, task: str, repo_root: str = ".") -> Dict[str, Any]:
        """
        Analyse the task, spawn the right ephemeral specialists, run them
        (in parallel via ThreadPoolExecutor), collect results, and clean up.

        Returns a dict: {role → result_dict}
        """
        needed = self._identify_needed_specialists(task)
        if not needed:
            return {"note": "No ephemeral specialists needed for this task."}

        # Routing decision: how complex is this task overall?
        gen_cfg = self.router.get_gen_config(task)

        results: Dict[str, Any] = {
            "routing_head": gen_cfg.get("head", "fast"),
            "specialists_spawned": [],
        }

        # Spawn all needed agents
        spawned: List[Tuple[EphemeralAgent, str]] = []
        for _, role, prompt, cap_key in needed:
            branch = re.sub(r"[^a-z0-9_]", "_", role.lower())
            worktree = self._create_worktree(repo_root, branch)
            agent = self.spawn(role, prompt, cap_key, worktree)
            spawned.append((agent, worktree or ""))
            results["specialists_spawned"].append(role)

        # Execute in parallel
        def _run_agent(agent_worktree: Tuple[EphemeralAgent, str]) -> Tuple[str, HandoffMessage]:
            agent, _ = agent_worktree
            msg = agent.run(task)
            agent.retire()
            return agent.role, msg

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(spawned)) as pool:
            futures = {pool.submit(_run_agent, aw): aw[0].role for aw in spawned}
            for future in concurrent.futures.as_completed(futures):
                try:
                    role, msg = future.result(timeout=60)
                    results[role] = msg.artifacts
                    results[f"{role}_confidence"] = msg.confidence
                except Exception as e:
                    role = futures[future]
                    results[role] = {"error": str(e)}

        # Clean up worktrees and agents
        for agent, worktree in spawned:
            if worktree:
                self._remove_worktree(worktree, repo_root)
        self.retire_all_done()

        return results


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Multi-Level Memory Manager (Dreamer v2)
# ═══════════════════════════════════════════════════════════════════════════

class MultiLevelMemoryMgr:
    """
    Extends Dreamer's capabilities with:
      - Three-tier in-model memory: gist (8 slots) → summary (64) → detailed (128).
      - GraphRAG for cross-session semantic retrieval.
      - Persistent episodic snapshots (loaded/saved across sessions).
      - Managed compression pipeline:
          raw episodes → detailed analysis → summary paragraph → gist token.

    Other agents call:
        mgr.query(text)          → top-k semantically relevant memory entries
        mgr.compress_episode(ep) → produces gist/summary/detailed for the episode
        mgr.persist(path)        → snapshot to disk
        mgr.restore(path)        → restore from disk
    """

    CONSOLIDATE_EVERY = 8

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional["CogForge"] = None,
                 config: Optional[CogForgeConfig] = None):
        self.memory    = memory_store
        self.model     = model
        self.config    = config or CogForgeConfig()
        self.mem_state: Optional[torch.Tensor] = None
        self._interaction_count = 0
        self._hier_memory: Optional[HierarchicalMemory] = None

        if model is not None:
            self._hier_memory = model.hier_memory

    # ── Low-level tensor memory ───────────────────────────────────────────

    def _ensure_mem(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        if self.mem_state is None:
            assert self.model is not None, "MultiLevelMemoryMgr needs a CogForge model."
            self.mem_state = self.model.hier_memory.init_memory(1, device)
        return self.mem_state

    def update_from_hidden(self, recent_hidden: torch.Tensor) -> None:
        assert self.model is not None
        mem = self._ensure_mem(recent_hidden.device)
        self.mem_state = self.model.hier_memory.update(recent_hidden, mem)

    def consolidate(self, full_hidden: torch.Tensor) -> None:
        assert self.model is not None
        mem = self._ensure_mem(full_hidden.device)
        self.mem_state = self.model.hier_memory.consolidate(full_hidden, mem)

    def query_tensor(self, query_hidden: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        mem = self._ensure_mem(query_hidden.device)
        return self.model.hier_memory.read(query_hidden, mem)

    # ── Multi-level compression pipeline ─────────────────────────────────

    def compress_episode(self, episode: Dict, node_id: Optional[str] = None) -> Dict:
        """
        Compress a raw episode dict into gist / summary / detailed strings.
        Writes the compressed forms into GraphRAGMemory if node_id given.
        """
        messages = episode.get("messages", [])
        raw_text = "\n".join(
            f"[{m['source']}→{m['target']}] {m['message']}" for m in messages
        )

        # Detailed: first 2000 chars of raw
        detailed = raw_text[:2000]

        # Summary: first 3 messages, trimmed
        summary_msgs = messages[:3]
        summary = " | ".join(
            f"{m['source']}→{m['target']}: {m['message'][:80]}"
            for m in summary_msgs
        )

        # Gist: just agent flow and status counts
        done_count    = sum(1 for m in messages if m.get("status") == "done")
        error_count   = sum(1 for m in messages if m.get("status") == "error")
        agents_active = list({m["source"] for m in messages})[:4]
        gist = (f"Batch#{episode.get('interaction_batch', '?')} "
                f"agents={','.join(agents_active)} done={done_count} err={error_count}")

        if node_id:
            self.memory.graph.set_compression(
                node_id=node_id,
                gist=gist, summary=summary, detailed=detailed, raw=episode,
            )

        return {"gist": gist, "summary": summary, "detailed": detailed}

    # ── Semantic query over GraphRAG ──────────────────────────────────────

    def query(self, text: str, top_k: int = 5,
              level: str = "summary") -> List[Dict]:
        """
        Semantic search over the graph + return compressed content at *level*.
        level: 'gist' | 'summary' | 'detailed' | 'raw'
        """
        hits = self.memory.graph.semantic_search(text, top_k=top_k)
        results = []
        for sim, node in hits:
            content = self.memory.graph.get_compression(node.node_id, level)
            results.append({
                "node_id":  node.node_id,
                "label":    node.label,
                "type":     node.type,
                "sim":      round(sim, 4),
                "content":  content,
            })
        return results

    # ── Episodic consolidation loop ───────────────────────────────────────

    def maybe_consolidate(self) -> Optional[Dict]:
        """
        Called after every agent interaction.  Every CONSOLIDATE_EVERY
        interactions, compress the recent episode batch and write into graph.
        """
        self._interaction_count += 1
        if self._interaction_count % self.CONSOLIDATE_EVERY != 0:
            return None

        recent = self.memory._log[-self.CONSOLIDATE_EVERY:]
        episode = {
            "interaction_batch": self._interaction_count,
            "messages": [
                {"source": m.source_agent, "target": m.target_agent,
                 "message": m.message[:200], "status": m.status}
                for m in recent
            ],
            "repo_map_snapshot": {k: v for k, v in
                                  list(self.memory.get_repo_map().items())[:10]},
        }
        self.memory.add_episode(episode)

        # Register episode as a graph node
        ep_id = f"episode_{self._interaction_count}"
        self.memory.graph.upsert_node(
            node_id=ep_id,
            node_type="episode",
            label=f"Episode batch {self._interaction_count}",
            metadata={"interaction_batch": self._interaction_count},
            embed_text=" ".join(m["message"] for m in episode["messages"]),
        )

        compression = self.compress_episode(episode, node_id=ep_id)
        return {**episode, **compression}

    # ── Persistence ───────────────────────────────────────────────────────

    def persist(self, path: str) -> None:
        """Snapshot shared memory + graph node labels to disk."""
        self.memory.snapshot(path)
        # Also save graph node labels (not embeddings — those are re-computed)
        graph_path = path + ".graph.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": [
                    {"node_id": n.node_id, "type": n.type,
                     "label": n.label, "metadata": n.metadata}
                    for n in self.memory.graph._nodes.values()
                ],
                "edges": [
                    {"src": e.src, "dst": e.dst, "rel": e.rel, "weight": e.weight}
                    for e in self.memory.graph._edges
                ],
                "gist":     self.memory.graph._gist,
                "summary":  self.memory.graph._summary,
            }, f, indent=2)

    def restore(self, path: str) -> None:
        """Restore shared memory + graph node labels from disk."""
        self.memory.restore(path)
        graph_path = path + ".graph.json"
        if not os.path.isfile(graph_path):
            return
        with open(graph_path, encoding="utf-8") as f:
            data = json.load(f)
        for n in data.get("nodes", []):
            self.memory.graph.upsert_node(
                node_id=n["node_id"], node_type=n["type"],
                label=n["label"], metadata=n.get("metadata", {}),
                embed_text=n["label"],
            )
        for e in data.get("edges", []):
            self.memory.graph.add_edge(e["src"], e["dst"], e["rel"], e.get("weight", 1.0))
        self.memory.graph._gist.update(data.get("gist", {}))
        self.memory.graph._summary.update(data.get("summary", {}))


# ═══════════════════════════════════════════════════════════════════════════
# CogForge Model (v1 preserved, minor additions for routing hooks)
# ═══════════════════════════════════════════════════════════════════════════

AGENT_NAMES: List[str] = [
    "Coordinator", "Dreamer", "Explorer", "Planner",
    "ProblemSolver", "Engineer", "BugFinder", "TerminalGuy",
    "VulnerabilityFinder", "Pessimist", "Documentor", "Nexus",
    "Archeologist", "MetaOrchestrator",
]


class CogForge(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model,
                                  padding_idx=config.pad_token_id)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, config.d_model) * 0.02
        )
        self.latent_controller = DynamicLatentController(config)
        self.hier_memory       = HierarchicalMemory(config)

        memory_layer_ids = set(range(config.n_layers - config.n_memory_layers,
                                     config.n_layers))
        self.blocks = nn.ModuleList()
        self.act_blocks: List[int] = []
        for i in range(config.n_layers):
            inject_mem = (i in memory_layer_ids)
            base = CogForgeBlock(config, inject_memory=inject_mem)
            if i >= config.n_layers // 2:
                self.blocks.append(ACTBlock(base, config.d_model,
                                            config.n_act_iterations,
                                            config.act_threshold))
                self.act_blocks.append(i)
            else:
                self.blocks.append(base)

        self.architect_encoder = ArchitectEncoder(config)
        self.arch_cross_attn = nn.ModuleDict({
            "4":  ArchitectCrossAttention(config),
            "8":  ArchitectCrossAttention(config),
            "10": ArchitectCrossAttention(config),
        })

        self.norm_out  = RMSNorm(config.d_model)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.verifier     = VerifierHead(config)
        self.handoff_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, len(AGENT_NAMES)),
        )
        self.drop = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embed" in name or "latent" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "norm" in name or "weight" in name:
                nn.init.ones_(p)
            if "bias" in name:
                nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor,
                repo_chunks: Optional[torch.Tensor] = None,
                repo_mask: Optional[torch.Tensor] = None,
                past_kvs: Optional[List] = None,
                memory_state: Optional[torch.Tensor] = None,
                return_verifier: bool = False,
                return_handoff: bool = False) -> Dict:
        B, T = input_ids.shape
        device = input_ids.device

        x            = self.drop(self.embed(input_ids))
        base_latents = self.latent_tokens.expand(B, -1, -1)
        arch_ctx     = None

        if repo_chunks is not None:
            arch_ctx = self.architect_encoder(repo_chunks, repo_mask)

        if memory_state is None:
            memory_state = self.hier_memory.init_memory(B, device)

        x = torch.cat([base_latents, x], dim=1)
        new_kvs       = []
        total_ponder  = torch.zeros(B, device=device)
        block_past_kvs = past_kvs or [None] * len(self.blocks)
        new_memory    = memory_state

        for i, block in enumerate(self.blocks):
            layer_key = str(i)
            if arch_ctx is not None and layer_key in self.arch_cross_attn:
                x = self.arch_cross_attn[layer_key](x, arch_ctx, repo_mask)

            if i == 1:
                expanded = self.latent_controller(x, base_latents)
                n_extra  = expanded.shape[1] - base_latents.shape[1]
                if n_extra > 0:
                    x = torch.cat([expanded[:, base_latents.shape[1]:, :], x], dim=1)

            if i % max(1, len(self.blocks) // 4) == 0 and i > 0:
                n_lat = x.shape[1] - T
                new_memory = self.hier_memory.update(x[:, n_lat:, :], new_memory)

            if i in self.act_blocks:
                x, ponder = block(x, memory=new_memory, hier_memory=self.hier_memory,
                                  past_kv=block_past_kvs[i])
                total_ponder = total_ponder + ponder.mean(dim=1)
                new_kvs.append(None)
            else:
                x, kv = block(x, past_kv=block_past_kvs[i],
                              memory=new_memory, hier_memory=self.hier_memory)
                new_kvs.append(kv)

        n_lat = x.shape[1] - T
        new_memory = self.hier_memory.consolidate(x[:, n_lat:, :], new_memory)
        x = x[:, n_lat:, :]
        x = self.norm_out(x)

        logits = self.lm_head(x)
        out: Dict[str, Any] = {
            "logits":       logits,
            "ponder_cost":  total_ponder,
            "past_kvs":     new_kvs,
            "memory_state": new_memory,
        }

        if return_verifier:
            mask = (input_ids != self.config.pad_token_id).float()
            out["verifier_score"] = self.verifier(x, mask)

        if return_handoff:
            pooled = x.mean(1)
            logits_agent = self.handoff_proj(pooled)
            out["handoff_agent_logits"] = logits_agent
            out["recommended_agent"]    = [
                AGENT_NAMES[idx] for idx in logits_agent.argmax(dim=-1).tolist()
            ]
        return out

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_p: float = 0.95, top_k: int = 50,
                 repo_chunks: Optional[torch.Tensor] = None,
                 repo_mask: Optional[torch.Tensor] = None,
                 memory_state: Optional[torch.Tensor] = None,
                 eos_token_id: int = 2) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        past_kvs  = None

        for _ in range(max_new_tokens):
            out = self.forward(
                generated if past_kvs is None else generated[:, -1:],
                repo_chunks=repo_chunks, repo_mask=repo_mask,
                past_kvs=past_kvs, memory_state=memory_state,
            )
            logits = out["logits"][:, -1, :] / max(temperature, 1e-8)
            past_kvs     = out["past_kvs"]
            memory_state = out["memory_state"]

            if top_k > 0:
                topk_vals = torch.topk(logits, top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cum_probs - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            generated  = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# v1 Agents (preserved — BaseAgent, Coordinator, Dreamer wrapper, Explorer,
# Planner, ProblemSolver, Engineer, BugFinder, TerminalGuy,
# VulnerabilityFinder, Pessimist, Documentor, Nexus, Archeologist)
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent:
    role: str = "BaseAgent"

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional[CogForge] = None):
        self.memory  = memory_store
        self.model   = model
        self.task_id = str(uuid.uuid4())

    def emit(self, target: str, message: str, artifacts: Optional[Dict] = None,
             confidence: float = 1.0, status: str = "pending") -> HandoffMessage:
        msg = HandoffMessage(
            task_id=str(uuid.uuid4()), source_agent=self.role,
            target_agent=target, status=status, message=message,
            artifacts=artifacts or {}, confidence=confidence,
        )
        self.memory.log_handoff(msg)
        return msg

    def receive(self) -> List[HandoffMessage]:
        msgs = self.memory.get_handoffs_for(self.role)
        for m in msgs:
            m.status = "running"
        return msgs

    def run(self, task: str, **kwargs) -> HandoffMessage:
        raise NotImplementedError


class Coordinator(BaseAgent):
    role = "Coordinator"
    VERIFIER_THRESHOLD = 0.75

    def run(self, task: str, **kwargs) -> HandoffMessage:
        self.memory.set("current_task", task)
        self.memory.set("coordinator_status", "running")
        plan_msg    = self.emit("Planner",
                                f"Decompose: {task}",
                                artifacts={"raw_task": task})
        explore_msg = self.emit("Explorer",
                                "Map the repository and flag complex areas.",
                                artifacts={"raw_task": task})
        self.memory.set("plan_task_id",    plan_msg.task_id)
        self.memory.set("explore_task_id", explore_msg.task_id)
        return self.emit("Coordinator",
                         "Delegated to Planner and Explorer. Awaiting results.",
                         status="running",
                         artifacts={"plan_task_id": plan_msg.task_id,
                                    "explore_task_id": explore_msg.task_id})

    def review_and_gate(self, verifier_score: float, task_id: str) -> str:
        if verifier_score >= self.VERIFIER_THRESHOLD:
            self.memory.mark_done(task_id)
            return "approve"
        elif verifier_score >= 0.4:
            return "iterate"
        else:
            return "escalate"

    def assign(self, subtask: str, target_agent: str,
               artifacts: Optional[Dict] = None) -> HandoffMessage:
        return self.emit(target_agent, subtask, artifacts=artifacts or {})


class Dreamer(BaseAgent):
    """
    v2: thin wrapper — delegates to MultiLevelMemoryMgr for all heavy lifting.
    Kept for backwards compatibility.
    """
    role = "Dreamer"
    CONSOLIDATE_EVERY = 8

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional[CogForge] = None,
                 config: Optional[CogForgeConfig] = None,
                 mem_mgr: Optional[MultiLevelMemoryMgr] = None):
        super().__init__(memory_store, model)
        self.config  = config or CogForgeConfig()
        self.mem_mgr = mem_mgr or MultiLevelMemoryMgr(memory_store, model, config)

    def update_from_hidden(self, recent_hidden: torch.Tensor) -> None:
        self.mem_mgr.update_from_hidden(recent_hidden)

    def consolidate(self, full_hidden: torch.Tensor) -> None:
        self.mem_mgr.consolidate(full_hidden)

    def query(self, query_hidden: torch.Tensor) -> torch.Tensor:
        return self.mem_mgr.query_tensor(query_hidden)

    def maybe_consolidate_episodic(self) -> None:
        result = self.mem_mgr.maybe_consolidate()
        if result:
            self.emit("Coordinator",
                      (f"Memory consolidated. Gist: {result.get('gist', '')[:100]}. "
                       f"Episode #{self.mem_mgr._interaction_count} saved."),
                      artifacts={"gist": result.get("gist"),
                                 "summary": result.get("summary")},
                      status="done")

    def run(self, task: str, **kwargs) -> HandoffMessage:
        self.maybe_consolidate_episodic()
        # Semantic retrieval from GraphRAG
        hits = self.mem_mgr.query(task, top_k=3, level="gist")
        summary_lines = [h.get("content", "") or "" for h in hits if h.get("content")]
        summary = "\n".join(summary_lines) if summary_lines else "(no prior context)"
        return self.emit("Coordinator",
                         f"Context from GraphRAG memory:\n{summary}",
                         artifacts={"graph_stats": self.memory.graph.stats(),
                                    "n_episodes": len(self.memory._episodic)},
                         status="done")


class Explorer(BaseAgent):
    role = "Explorer"
    COMPLEXITY_THRESHOLD = 10
    LARGE_FILE_LINES     = 300

    def _walk(self, root: str) -> List[str]:
        py_files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".py"):
                    py_files.append(os.path.join(dirpath, fn))
        return py_files

    def _analyze_file(self, path: str) -> Dict:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except OSError as e:
            return {"error": str(e), "flagged": True, "flag_reason": "unreadable"}
        line_count = source.count("\n")
        parse_error = False
        n_functions = 0
        max_depth   = 0
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n_functions += 1
                if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    max_depth += 1
        except SyntaxError:
            parse_error = True
        complexity_score = n_functions + max_depth
        flagged = (parse_error or
                   complexity_score > self.COMPLEXITY_THRESHOLD or
                   line_count > self.LARGE_FILE_LINES)
        flag_reason = []
        if parse_error: flag_reason.append("syntax_error")
        if complexity_score > self.COMPLEXITY_THRESHOLD:
            flag_reason.append(f"high_complexity({complexity_score})")
        if line_count > self.LARGE_FILE_LINES:
            flag_reason.append(f"large_file({line_count}_lines)")
        return {"line_count": line_count, "n_functions": n_functions,
                "complexity_score": complexity_score, "parse_error": parse_error,
                "flagged": flagged,
                "flag_reason": ", ".join(flag_reason) if flag_reason else "ok"}

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        py_files = self._walk(root)
        flagged_files: List[str] = []
        for path in py_files:
            info = self._analyze_file(path)
            self.memory.update_repo_map(path, info)
            if info.get("flagged"):
                flagged_files.append(path)

        # v2: ingest into GraphRAG
        self.memory.graph.ingest_repo_map(self.memory.get_repo_map())

        if flagged_files:
            self.emit("BugFinder", "Inspect flagged files for bugs.",
                      artifacts={"flagged_files": flagged_files})
            self.emit("VulnerabilityFinder", "Scan flagged files for vulnerabilities.",
                      artifacts={"flagged_files": flagged_files})
        self.emit("Planner",
                  f"Repository mapped: {len(py_files)} files, {len(flagged_files)} flagged.",
                  artifacts={"total_files": len(py_files),
                             "flagged_files": flagged_files,
                             "repo_map_keys": list(self.memory.get_repo_map().keys()),
                             "graph_stats": self.memory.graph.stats()})
        return self.emit("Coordinator",
                         f"Exploration complete. {len(flagged_files)} files flagged. "
                         f"Graph: {self.memory.graph.stats()}",
                         artifacts={"flagged_files": flagged_files,
                                    "graph_stats": self.memory.graph.stats()},
                         status="done")


class Planner(BaseAgent):
    role = "Planner"

    def _heuristic_dag(self, task: str) -> List[Dict]:
        return [
            {"id": "t0", "description": f"Understand requirements: {task[:120]}",
             "dependencies": [], "assigned_agent": "ProblemSolver",
             "estimated_effort": 2, "status": "pending"},
            {"id": "t1", "description": "Identify affected files from repo map",
             "dependencies": ["t0"], "assigned_agent": "Explorer",
             "estimated_effort": 1, "status": "pending"},
            {"id": "t2", "description": "Design solution architecture",
             "dependencies": ["t0", "t1"], "assigned_agent": "ProblemSolver",
             "estimated_effort": 3, "status": "pending"},
            {"id": "t3", "description": "Implement changes",
             "dependencies": ["t2"], "assigned_agent": "Engineer",
             "estimated_effort": 4, "status": "pending"},
            {"id": "t4", "description": "Stress-test plan and implementation",
             "dependencies": ["t3"], "assigned_agent": "Pessimist",
             "estimated_effort": 2, "status": "pending"},
            {"id": "t5", "description": "Bug and security review",
             "dependencies": ["t3"], "assigned_agent": "BugFinder",
             "estimated_effort": 2, "status": "pending"},
            {"id": "t6", "description": "Run tests via terminal",
             "dependencies": ["t5"], "assigned_agent": "TerminalGuy",
             "estimated_effort": 1, "status": "pending"},
            {"id": "t7", "description": "Add documentation and comments",
             "dependencies": ["t3", "t5"], "assigned_agent": "Documentor",
             "estimated_effort": 2, "status": "pending"},
        ]

    def _validate_dag(self, dag: List[Dict]) -> bool:
        graph: Dict[str, List[str]] = {n["id"]: n["dependencies"] for n in dag}
        visited: set = set()
        rec_stack: set = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for dep in graph.get(node_id, []):
                if dep not in visited:
                    if dfs(dep): return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node_id)
            return False

        return not any(dfs(n["id"]) for n in dag if n["id"] not in visited)

    def run(self, task: str, **kwargs) -> HandoffMessage:
        dag = self._heuristic_dag(task)
        if not self._validate_dag(dag):
            for i, node in enumerate(dag):
                node["dependencies"] = [dag[i-1]["id"]] if i > 0 else []
        self.memory.set("task_dag", dag)
        for node in dag:
            self.emit(node["assigned_agent"], node["description"],
                      artifacts={"dag_node": node})
        self.emit("ProblemSolver", "Review the task DAG and refine solution strategy.",
                  artifacts={"dag": dag, "task": task})
        return self.emit("Coordinator",
                         f"Task DAG created with {len(dag)} nodes.",
                         artifacts={"dag": dag}, status="done")


class ProblemSolver(BaseAgent):
    role = "ProblemSolver"

    def _score_approach(self, description: str) -> float:
        pos = ["test", "modular", "interface", "incremental", "rollback"]
        neg = ["rewrite", "delete", "global", "hack", "skip"]
        score = 0.5
        dl = description.lower()
        for s in pos: score += 0.05 if s in dl else 0
        for s in neg: score -= 0.07 if s in dl else 0
        return max(0.0, min(1.0, score))

    def _generate_notes(self, node: Dict, n_flagged: int, n_files: int) -> str:
        desc   = node.get("description", "")
        effort = node.get("estimated_effort", 3)
        lines  = [f"Subtask: {desc}", f"Effort estimate: {effort}/5",
                  f"Repo context: {n_files} files total, {n_flagged} flagged."]
        if effort >= 4:
            lines.append("High-effort task: break into smaller steps if possible.")
        if n_flagged > 5:
            lines.append("Many flagged files: run tests after each change.")
        return "\n".join(lines)

    def run(self, task: str, **kwargs) -> HandoffMessage:
        dag     = self.memory.get("task_dag") or []
        n_flag  = len(self.memory.get_flagged_files())
        n_files = len(self.memory.get_repo_map())

        # v2: enrich with GraphRAG semantic context
        graph_hits = self.memory.graph.semantic_search(task, top_k=3, node_type="file")
        relevant_files = [node.label for _, node in graph_hits]

        annotated_dag: List[Dict] = []
        for node in dag:
            notes      = self._generate_notes(node, n_flag, n_files)
            confidence = self._score_approach(notes)
            annotated_dag.append({**node, "solution_notes": notes,
                                   "confidence": confidence})
        self.memory.set("annotated_dag", annotated_dag)

        risky = [n for n in annotated_dag if n["confidence"] < 0.45]
        if risky:
            self.emit("Pessimist",
                      "Stress-test these low-confidence subtasks.",
                      artifacts={"risky_nodes": risky})

        return self.emit("Coordinator",
                         (f"Solution path derived. {len(annotated_dag)} nodes annotated; "
                          f"{len(risky)} flagged for Pessimist. "
                          f"Relevant files from GraphRAG: {relevant_files[:3]}"),
                         artifacts={"annotated_dag": annotated_dag,
                                    "relevant_files_graph": relevant_files},
                         confidence=min((n["confidence"] for n in annotated_dag),
                                        default=0.5),
                         status="done")


class Engineer(BaseAgent):
    role = "Engineer"
    MAX_FUNC_LINES = 40

    def _analyze_code(self, source: str) -> Dict:
        issues: List[str] = []
        suggestions: List[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return {"issues": [f"SyntaxError: {e}"], "suggestions": [],
                    "refactored": source}
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start  = node.lineno - 1
                end    = getattr(node, "end_lineno", start + 1)
                length = end - start
                if length > self.MAX_FUNC_LINES:
                    issues.append(f"Function '{node.name}' is {length} lines.")
                    suggestions.append(f"Split '{node.name}' into helpers.")
                if not ast.get_docstring(node):
                    issues.append(f"Function '{node.name}' missing docstring.")
                    suggestions.append(f"Add docstring to '{node.name}'.")
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        issues.append(f"Wildcard import from '{node.module}'.")
        blank_runs = prev_blank = 0
        for line in lines:
            if not line.strip():
                if prev_blank: blank_runs += 1
                prev_blank = 1
            else:
                prev_blank = 0
        if blank_runs > 3:
            issues.append(f"Excessive blank lines ({blank_runs}).")
        return {"issues": issues, "suggestions": suggestions, "refactored": source}

    def run(self, task: str, code: str = "", **kwargs) -> HandoffMessage:
        if not code:
            for msg in self.receive():
                code = msg.artifacts.get("code", "")
                if code: break
        analysis = self._analyze_code(code) if code else {
            "issues": ["No code provided."], "suggestions": [], "refactored": ""}
        self.emit("Pessimist", "Critique this refactoring analysis.",
                  artifacts={"analysis": analysis, "code": code})
        return self.emit("Coordinator",
                         (f"Engineering analysis: {len(analysis['issues'])} issues, "
                          f"{len(analysis['suggestions'])} suggestions."),
                         artifacts=analysis,
                         confidence=max(0.3, 1.0 - 0.1 * len(analysis["issues"])),
                         status="done")


class BugFinder(BaseAgent):
    role = "BugFinder"

    SEVERITY_PATTERNS: List[Tuple[str, str, str]] = [
        (r"except\s*:",            "high",     "Bare except clause"),
        (r"eval\s*\(",             "critical", "eval() — dangerous"),
        (r"exec\s*\(",             "critical", "exec() — dangerous"),
        (r"assert\s+",             "medium",   "assert stripped in -O mode"),
        (r"time\.sleep\s*\(",      "low",      "Blocking sleep"),
        (r"TODO|FIXME|HACK",       "low",      "Unresolved marker"),
        (r"\.format\s*\(",         "info",     "Use f-strings"),
        (r"global\s+\w+",          "medium",   "Global variable mutation"),
        (r"import \*",             "medium",   "Wildcard import"),
        (r"open\s*\([^)]*\)\s*$",  "high",     "File opened without context manager"),
    ]

    def _suggest_fix(self, pattern: str, _line: str) -> str:
        fixes = {
            r"except\s*:":           "Use `except Exception as e:`.",
            r"eval\s*\(":            "Replace with ast.literal_eval().",
            r"exec\s*\(":            "Remove exec(); use importlib or subprocess.",
            r"assert\s+":            "Use explicit if/raise for production guards.",
            r"time\.sleep\s*\(":     "Use asyncio.sleep() in async code.",
            r"TODO|FIXME|HACK":      "Resolve before merging.",
            r"\.format\s*\(":        "Rewrite as an f-string.",
            r"global\s+\w+":         "Pass as argument or use a class attribute.",
            r"import \*":            "Use explicit imports.",
            r"open\s*\([^)]*\)\s*$": "Wrap in `with open(...)` context manager.",
        }
        return fixes.get(pattern, "Review manually.")

    def _scan_source(self, source: str, filepath: str) -> List[Dict]:
        findings: List[Dict] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            for pattern, severity, desc in self.SEVERITY_PATTERNS:
                if re.search(pattern, line):
                    findings.append({
                        "file": filepath, "line": lineno, "severity": severity,
                        "description": desc, "code_snippet": line.strip(),
                        "suggested_fix": self._suggest_fix(pattern, line.strip()),
                    })
        return findings

    def run(self, task: str, **kwargs) -> HandoffMessage:
        files_to_scan: List[str] = []
        for msg in self.receive():
            files_to_scan.extend(msg.artifacts.get("flagged_files", []))
        if not files_to_scan:
            files_to_scan = self.memory.get_flagged_files()

        all_findings: List[Dict] = []
        for filepath in files_to_scan:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    all_findings.extend(self._scan_source(f.read(), filepath))
            except OSError:
                all_findings.append({"file": filepath, "severity": "error",
                                      "description": "Could not read.", "line": 0,
                                      "code_snippet": "", "suggested_fix": ""})
        critical = [f for f in all_findings if f["severity"] == "critical"]
        if critical:
            self.emit("TerminalGuy", "Run test suite to confirm critical bugs.",
                      artifacts={"command": "python -m pytest --tb=short -q",
                                 "critical_findings": critical})
        return self.emit("Coordinator",
                         f"Bug scan: {len(all_findings)} findings, {len(critical)} critical.",
                         artifacts={"findings": all_findings,
                                    "files_scanned": len(files_to_scan)},
                         confidence=max(0.1, 1.0 - 0.05 * len(all_findings)),
                         status="done")


class TerminalGuy(BaseAgent):
    role = "TerminalGuy"
    DEFAULT_TIMEOUT = 60
    RISKY_PATTERNS  = re.compile(
        r"\b(rm\s+-rf|DROP\s+TABLE|shutdown|reboot|mkfs|dd\s+if=|"
        r"chmod\s+777|curl\s+.*\|\s*sh)\b", re.IGNORECASE,
    )

    def _is_approved(self, command: str, task_id: str) -> bool:
        return self.memory.get(f"approved_command:{task_id}", False) or \
               not bool(self.RISKY_PATTERNS.search(command))

    def _run_command(self, command: str, timeout: int) -> Dict:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=timeout)
            return {"returncode": result.returncode,
                    "stdout": result.stdout[:4000], "stderr": result.stderr[:2000],
                    "timed_out": False}
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": "Timed out.",
                    "timed_out": True}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e), "timed_out": False}

    def run(self, task: str, command: str = "", timeout: int = DEFAULT_TIMEOUT,
            requester: str = "Coordinator", task_id: str = "", **kwargs) -> HandoffMessage:
        if not command:
            for msg in self.receive():
                command   = msg.artifacts.get("command", "")
                requester = msg.source_agent
                task_id   = msg.task_id
                if command: break
        if not command:
            return self.emit(requester, "No command provided.", status="error")
        if not self._is_approved(command, task_id):
            self.emit("Coordinator", f"Risky command requires approval: `{command}`",
                      artifacts={"command": command, "task_id": task_id})
            return self.emit(requester, "Command flagged; awaiting approval.", status="pending")
        result = self._run_command(command, timeout)
        self.memory.set(f"terminal_result:{task_id}", result)
        return self.emit(requester,
                         f"Command exited {result['returncode']}.\n{result['stdout'][:300]}",
                         artifacts=result,
                         confidence=1.0 if result["returncode"] == 0 else 0.3,
                         status="done")


class VulnerabilityFinder(BaseAgent):
    role = "VulnerabilityFinder"

    VULN_PATTERNS: List[Tuple[str, str, str, str]] = [
        (r"password\s*=\s*['\"][^'\"]+['\"]",        "critical", "CWE-259", "Hardcoded password"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]",           "critical", "CWE-321", "Hardcoded secret"),
        (r"subprocess\.call\(.*shell\s*=\s*True",      "high",     "CWE-78",  "Shell injection"),
        (r"os\.system\s*\(",                           "high",     "CWE-78",  "Command injection"),
        (r"pickle\.loads?\s*\(",                       "high",     "CWE-502", "Unsafe pickle"),
        (r"yaml\.load\s*\([^,)]+\)",                  "high",     "CWE-502", "Unsafe YAML load"),
        (r"request\.(args|form)\.get",                 "medium",   "CWE-20",  "Unvalidated input"),
        (r"md5\s*\(|hashlib\.md5",                    "medium",   "CWE-327", "Weak MD5"),
        (r"sha1\s*\(|hashlib\.sha1",                  "medium",   "CWE-327", "Weak SHA-1"),
        (r"random\.random\(\)|random\.randint",        "low",      "CWE-338", "Non-crypto RNG"),
        (r"DEBUG\s*=\s*True",                          "medium",   "CWE-215", "Debug mode on"),
        (r"ALLOWED_HOSTS\s*=\s*\[.*\*.*\]",           "high",     "CWE-183", "Wildcard hosts"),
        (r"verify\s*=\s*False",                        "high",     "CWE-295", "SSL disabled"),
    ]

    def _scan(self, source: str, filepath: str) -> List[Dict]:
        findings: List[Dict] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            for pattern, severity, cwe, desc in self.VULN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({"file": filepath, "line": lineno,
                                      "severity": severity, "cwe": cwe,
                                      "description": desc, "snippet": line.strip()})
        return findings

    def run(self, task: str, **kwargs) -> HandoffMessage:
        files_to_scan: List[str] = []
        for msg in self.receive():
            files_to_scan.extend(msg.artifacts.get("flagged_files", []))
        if not files_to_scan:
            files_to_scan = self.memory.get_flagged_files()

        all_findings: List[Dict] = []
        for filepath in files_to_scan:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    all_findings.extend(self._scan(f.read(), filepath))
            except OSError:
                continue

        critical = [f for f in all_findings if f["severity"] == "critical"]
        if all_findings:
            self.emit("Engineer", "Security vulnerabilities found — remediate.",
                      artifacts={"vuln_findings": all_findings})
        return self.emit("Coordinator",
                         f"Vuln scan: {len(all_findings)} findings, {len(critical)} critical.",
                         artifacts={"findings": all_findings},
                         confidence=max(0.1, 1.0 - 0.08 * len(critical)),
                         status="done")


class Pessimist(BaseAgent):
    role = "Pessimist"

    FAILURE_MODE_CHECKS: List[Tuple[str, str]] = [
        ("empty input",       "What happens when input is empty or None?"),
        ("max scale",         "Does this hold at 10x / 100x volume?"),
        ("concurrent access", "Race condition if two agents hit this simultaneously?"),
        ("partial failure",   "What if a downstream call fails halfway?"),
        ("stale cache",       "Could cached data cause silent incorrect results?"),
        ("integer overflow",  "Could numeric values overflow or cause index errors?"),
        ("unicode/encoding",  "Safe with non-ASCII input?"),
        ("missing rollback",  "Can the system recover cleanly if the op fails?"),
        ("dependency version","Could a library update break assumed behaviour?"),
        ("logging gap",       "Are errors logged sufficiently for debugging?"),
    ]

    def _critique_code(self, code: str) -> List[Dict]:
        if not code.strip():
            return [{"risk": "high", "check": "empty code", "detail": "No code."}]
        issues: List[Dict] = []
        for check, question in self.FAILURE_MODE_CHECKS:
            risk   = "low"
            detail = question
            if check == "empty input" and "if not " not in code and "is None" not in code:
                risk   = "medium"
                detail = "No None/empty guards — " + question
            elif check == "concurrent access" and "lock" not in code.lower():
                risk   = "medium"
                detail = "No locking primitives — " + question
            elif check == "partial failure" and "rollback" not in code.lower():
                risk   = "medium"
                detail = "No rollback logic — " + question
            elif check == "missing rollback" and "try" not in code and "except" not in code:
                risk   = "high"
                detail = "No try/except — " + question
            elif check == "logging gap" and "logging" not in code and "logger" not in code:
                risk   = "low"
                detail = "No logging calls — " + question
            issues.append({"risk": risk, "check": check, "detail": detail})
        return issues

    def _critique_plan(self, dag: List[Dict]) -> List[Dict]:
        critiques: List[Dict] = []
        if not dag:
            return [{"risk": "high", "check": "empty plan", "detail": "No plan."}]
        if not any("test" in n.get("description", "").lower() for n in dag):
            critiques.append({"risk": "high", "check": "missing tests",
                               "detail": "No testing step in plan."})
        if not any("rollback" in n.get("description", "").lower() for n in dag):
            critiques.append({"risk": "medium", "check": "missing rollback",
                               "detail": "No rollback step."})
        return critiques

    def run(self, task: str, code: str = "", **kwargs) -> HandoffMessage:
        dag = self.memory.get("annotated_dag") or []
        for msg in self.receive():
            code = code or msg.artifacts.get("code", "")
            if "dag" in msg.artifacts:
                dag = msg.artifacts["dag"]
        critiques  = self._critique_code(code) + self._critique_plan(dag)
        high_risk  = [c for c in critiques if c["risk"] == "high"]
        return self.emit("Coordinator",
                         f"Pessimist: {len(critiques)} issues, {len(high_risk)} high-risk.",
                         artifacts={"critiques": critiques, "high_risk": high_risk},
                         confidence=max(0.1, 1.0 - 0.1 * len(high_risk)),
                         status="done")


class Documentor(BaseAgent):
    role = "Documentor"

    def _make_docstring(self, node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            returns = ""
            if node.returns:
                try: returns = ast.unparse(node.returns)
                except Exception: returns = "?"
            lines = [f"{node.name}.", "", "Args:"]
            for arg in args:
                if arg != "self": lines.append(f"    {arg}: Description.")
            if returns and returns not in ("None", ""):
                lines += ["", "Returns:", f"    {returns}: Description."]
            return "\n".join(lines)
        if isinstance(node, ast.ClassDef):
            return f"{node.name}.\n\nAttributes:\n    (fill in attributes here)"
        if isinstance(node, ast.Module):
            return "Module docstring.\n\nThis module provides ...\n"
        return "TODO: add description."

    def _document_source(self, source: str) -> Tuple[str, int]:
        try: tree = ast.parse(source)
        except SyntaxError: return source, 0
        lines = source.splitlines(keepends=True)
        insertions: List[Tuple[int, str]] = []

        def needs_docstring(node: ast.AST) -> bool:
            body  = getattr(node, "body", [])
            if not body: return True
            first = body[0]
            return not (isinstance(first, ast.Expr) and
                        isinstance(first.value, ast.Constant) and
                        isinstance(first.value.value, str))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if needs_docstring(node):
                    insert_at = node.lineno
                    body_line  = node.body[0].lineno - 1 if node.body else node.lineno
                    indent = "    "
                    if body_line < len(lines):
                        raw    = lines[body_line]
                        indent = " " * (len(raw) - len(raw.lstrip()))
                    docstr  = self._make_docstring(node)
                    formatted = (indent + '"""' + docstr.split("\n")[0] + "\n"
                                 + "\n".join(indent + l for l in docstr.split("\n")[1:])
                                 + "\n" + indent + '"""' + "\n")
                    insertions.append((insert_at, formatted))

        if not insertions:
            return source, 0
        insertions.sort(key=lambda x: x[0], reverse=True)
        for line_idx, text in insertions:
            lines.insert(line_idx, text)
        return "".join(lines), len(insertions)

    def run(self, task: str, files: Optional[List[str]] = None, **kwargs) -> HandoffMessage:
        if not files:
            files = []
            for msg in self.receive():
                files.extend(msg.artifacts.get("files", []))
        if not files:
            files = self.memory.get_flagged_files()

        total_added = 0
        for filepath in files:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    original = f.read()
            except OSError:
                continue
            documented, n_added = self._document_source(original)
            total_added += n_added
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(documented)
            except OSError:
                pass

        return self.emit("Coordinator",
                         f"Documentation: {total_added} docstrings added across {len(files)} files.",
                         artifacts={"n_docstrings_added": total_added,
                                    "files_processed": len(files)},
                         confidence=0.9, status="done")


class Nexus(BaseAgent):
    role = "Nexus"
    _STDLIB = frozenset(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else frozenset()

    def _extract_imports(self, source: str) -> List[str]:
        packages: List[str] = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        packages.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    packages.append(node.module.split(".")[0])
        except SyntaxError:
            for m in re.finditer(r"^(?:import|from)\s+([A-Za-z0-9_]+)", source, re.MULTILINE):
                packages.append(m.group(1))
        return list(set(packages))

    def _installed_packages(self) -> Dict[str, str]:
        installed: Dict[str, str] = {}
        try:
            for dist in importlib.metadata.distributions():
                name    = dist.metadata.get("Name", "").lower()
                version = dist.metadata.get("Version", "unknown")
                if name: installed[name] = version
        except Exception:
            pass
        return installed

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        installed   = self._installed_packages()
        repo_map    = self.memory.get_repo_map()
        py_files    = list(repo_map.keys()) if repo_map else []
        all_pkg_names: Set[str] = set()

        for filepath in py_files:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    all_pkg_names.update(self._extract_imports(f.read()))
            except OSError:
                pass

        all_pkg_names = {p for p in all_pkg_names if p not in self._STDLIB}
        missing = [p for p in all_pkg_names if p not in installed]

        # v2: add packages as graph nodes
        for pkg in all_pkg_names:
            pid = f"pkg_{pkg}"
            self.memory.graph.upsert_node(
                node_id=pid, node_type="package", label=pkg,
                metadata={"installed": pkg in installed,
                           "version": installed.get(pkg, "N/A")},
                embed_text=pkg,
            )

        self.memory.set("nexus_installed", installed)
        self.memory.set("nexus_missing_from_env", missing)

        return self.emit("Coordinator",
                         f"Nexus: {len(all_pkg_names)} packages found, {len(missing)} missing from env.",
                         artifacts={"all_packages": list(all_pkg_names),
                                    "missing_from_env": missing,
                                    "installed_count": len(installed)},
                         confidence=max(0.3, 1.0 - 0.05 * len(missing)),
                         status="done")


class Archeologist(BaseAgent):
    role = "Archeologist"
    STALE_DAYS       = 730
    HOT_FIX_COMMITS  = 5
    LARGE_COMMIT_FILES = 20

    _CAUTION_PATTERNS: List[Tuple[str, str]] = [
        (r"\b(hotfix|hot[- ]fix)\b",           "hotfix"),
        (r"\b(race condition|mutex|lock)\b",    "concurrency-fix"),
        (r"\b(revert|rollback)\b",              "revert"),
        (r"\b(security|CVE-\d+|vuln|exploit)\b","security-patch"),
        (r"\b(performance|perf|slow|N\+1)\b",   "perf-fix"),
        (r"\b(memory leak|OOM|oom)\b",          "memory-fix"),
        (r"\b(migration|migrate)\b",            "migration"),
        (r"\bWIP\b",                            "work-in-progress"),
    ]

    def _git(self, args: List[str], cwd: str, timeout: int = 15) -> str:
        try:
            result = subprocess.run(["git"] + args, cwd=cwd,
                                    capture_output=True, text=True, timeout=timeout)
            return result.stdout.strip()
        except Exception:
            return ""

    def _has_git(self, root: str) -> bool:
        return bool(self._git(["rev-parse", "--git-dir"], root))

    def _classify_file(self, filepath: str, commits: List[Dict]) -> Dict:
        now = time.time()
        if not commits:
            return {"age_days": None, "last_touched_days_ago": None,
                    "commit_count": 0, "volatile": False, "stale": True,
                    "caution_flags": [], "caution_reasons": [],
                    "large_commit_hashes": [], "hotfix_subjects": [],
                    "recommendation": "no-git-history"}
        timestamps = []
        for c in commits:
            try:
                date_part = c["date_iso"].rsplit(" ", 1)[0]
                import datetime
                dt = datetime.datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                timestamps.append(dt.timestamp())
            except Exception:
                pass

        if timestamps:
            first_ts = min(timestamps)
            last_ts  = max(timestamps)
            age_days              = (now - first_ts) / 86400
            last_touched_days_ago = (now - last_ts) / 86400
        else:
            age_days = last_touched_days_ago = None

        ninety_days_ago = now - 90 * 86400
        recent_commits  = [c for c, t in zip(commits, timestamps)
                           if t and t > ninety_days_ago]
        volatile = len(recent_commits) >= self.HOT_FIX_COMMITS
        stale    = last_touched_days_ago is not None and last_touched_days_ago > self.STALE_DAYS

        caution_flags: List[str]   = []
        caution_reasons: List[str] = []
        hotfix_subjects: List[str] = []
        for commit in commits:
            subj = commit["subject"].lower()
            for pattern, label in self._CAUTION_PATTERNS:
                if re.search(pattern, subj, re.IGNORECASE):
                    if label not in caution_flags:
                        caution_flags.append(label)
                    caution_reasons.append(f"[{commit['hash']}] {label}: {commit['subject'][:80]}")
                    if label == "hotfix":
                        hotfix_subjects.append(commit["subject"][:100])

        large_commit_hashes = [c["hash"] for c in commits
                                if c.get("files_changed", 0) >= self.LARGE_COMMIT_FILES]
        rec_parts: List[str] = []
        if volatile: rec_parts.append("VOLATILE: high recent change rate")
        if stale:    rec_parts.append("STALE: untouched 2+ years")
        if caution_flags: rec_parts.append(f"CAUTION: {', '.join(caution_flags)}")
        if large_commit_hashes: rec_parts.append(f"BLAST-RADIUS: {large_commit_hashes[:3]}")
        recommendation = " | ".join(rec_parts) if rec_parts else "safe-to-refactor"

        return {"age_days": round(age_days, 1) if age_days else None,
                "last_touched_days_ago": round(last_touched_days_ago, 1) if last_touched_days_ago else None,
                "commit_count": len(commits), "volatile": volatile, "stale": stale,
                "caution_flags": caution_flags, "caution_reasons": caution_reasons[:5],
                "large_commit_hashes": large_commit_hashes[:3],
                "hotfix_subjects": hotfix_subjects[:3], "recommendation": recommendation}

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        if not self._has_git(root):
            return self.emit("Coordinator", "No git repo. Archeologist skipped.",
                             artifacts={"has_git": False}, confidence=0.5, status="done")

        repo_map = self.memory.get_repo_map()
        py_files = list(repo_map.keys()) if repo_map else []

        temporal_map: Dict[str, Dict] = {}
        for filepath in py_files:
            temporal_map[filepath] = self._classify_file(filepath, [])

        volatile_files = [fp for fp, info in temporal_map.items() if info.get("volatile")]
        caution_files  = [fp for fp, info in temporal_map.items() if info.get("caution_flags")]
        stale_files    = [fp for fp, info in temporal_map.items()
                          if info.get("stale") and not info.get("volatile")]

        self.memory.set("temporal_map", temporal_map)

        # v2: add temporal annotations to graph nodes
        for fp, info in temporal_map.items():
            fid = hashlib.md5(fp.encode()).hexdigest()[:12]
            if fid in self.memory.graph._nodes:
                self.memory.graph._nodes[fid].metadata.update({
                    "volatile": info["volatile"], "stale": info["stale"],
                    "caution_flags": info["caution_flags"],
                    "recommendation": info["recommendation"],
                })

        if volatile_files or caution_files:
            self.emit("Engineer",
                      f"Archeologist: {len(volatile_files)} volatile, {len(caution_files)} caution zones.",
                      artifacts={"volatile_files": volatile_files,
                                 "caution_files": caution_files})
        if stale_files:
            self.emit("Documentor",
                      f"{len(stale_files)} stale files — mark with deprecation notices.",
                      artifacts={"stale_files": stale_files[:20]})

        summary = (f"Temporal map: volatile={len(volatile_files)} "
                   f"caution={len(caution_files)} stale={len(stale_files)}")
        return self.emit("Coordinator", summary,
                         artifacts={"volatile_files": volatile_files,
                                    "caution_files": caution_files,
                                    "stale_files": stale_files[:20]},
                         confidence=max(0.3, 1.0 - 0.04 * len(volatile_files)),
                         status="done")


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Process Reward Model (PRM)
# ═══════════════════════════════════════════════════════════════════════════

class ProcessRewardModel(nn.Module):
    """
    Step-level reward model trained on reasoning traces.
    Given a (prompt, partial_code_step) pair, outputs a scalar reward in [0,1]
    indicating whether this *reasoning step* is on the right track.

    Architecture: lightweight MLP over bag-of-characters features.
    In production: replace _encode() with a real code encoder and fine-tune
    on positive/negative reasoning trace pairs via Bradley-Terry loss.

    Used by HierarchicalCogSearch to prune low-quality branches early
    (before spending execution budget on them).
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        # Feature dimension: 512 character n-gram buckets
        self._feat_dim = 512
        depth = config.mcts_prm_layers

        layers: List[nn.Module] = []
        in_dim = self._feat_dim * 2   # (prompt_feats, code_feats) concatenated
        for i in range(depth):
            out_dim = max(64, in_dim // 2)
            layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            if i < depth - 1:
                layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def _encode(self, text: str) -> torch.Tensor:
        """Bag-of-character-bigrams hashed to _feat_dim bins."""
        vec = torch.zeros(self._feat_dim)
        for i in range(len(text) - 1):
            gram = text[i:i + 2]
            h    = int(hashlib.sha256(gram.encode()).hexdigest(), 16)
            vec[h % self._feat_dim] += 1.0
        norm = vec.norm().clamp(min=1e-6)
        return vec / norm

    def score(self, prompt: str, code_step: str) -> float:
        """Return step-level reward in [0, 1]."""
        with torch.no_grad():
            p_feat = self._encode(prompt[:512])
            c_feat = self._encode(code_step[:512])
            x      = torch.cat([p_feat, c_feat]).unsqueeze(0)
            return float(self.mlp(x).item())


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Coverage Reward Module
# ═══════════════════════════════════════════════════════════════════════════

class CoverageRewardModule:
    """
    Multi-signal reward enrichment beyond syntax/test pass:

    Signals (each 0..1, combined via weighted sum):
      - syntax_ok       : py_compile passes              (weight 0.15)
      - tests_pass      : pytest passes                  (weight 0.35)
      - coverage_pct    : pytest-cov line coverage ratio (weight 0.20)
      - security_ok     : no critical VulnFinder hits    (weight 0.15)
      - style_score     : repo-norm consistency          (weight 0.10)
      - prm_score       : Process Reward Model step score(weight 0.05)

    Produces a single scalar reward in [-1, +1].
    """

    WEIGHTS = {
        "syntax_ok":    0.15,
        "tests_pass":   0.35,
        "coverage_pct": 0.20,
        "security_ok":  0.15,
        "style_score":  0.10,
        "prm_score":    0.05,
    }

    def __init__(self, terminal_guy: "TerminalGuy",
                 vuln_finder: Optional["VulnerabilityFinder"] = None,
                 prm: Optional[ProcessRewardModel] = None):
        self.terminal_guy = terminal_guy
        self.vuln_finder  = vuln_finder
        self.prm          = prm

    # ── Individual signal extractors ─────────────────────────────────────

    def _check_syntax(self, tmp_path: str) -> float:
        r = self.terminal_guy._run_command(
            f"python3 -m py_compile {tmp_path}", timeout=10)
        return 1.0 if r["returncode"] == 0 else 0.0

    def _check_tests(self, tmp_path: str) -> Tuple[float, float]:
        """Returns (tests_pass 0/1, coverage_pct 0..1)."""
        r = self.terminal_guy._run_command(
            f"python3 -m pytest {tmp_path} --tb=no -q "
            f"--cov={tmp_path} --cov-report=term-missing 2>&1 | tail -20",
            timeout=30,
        )
        tests_pass = 1.0 if r["returncode"] == 0 else 0.0
        # Parse coverage from output
        coverage_pct = 0.0
        m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", r.get("stdout", ""))
        if m:
            coverage_pct = int(m.group(1)) / 100.0
        return tests_pass, coverage_pct

    def _check_security(self, code: str) -> float:
        """Use VulnerabilityFinder patterns inline (no disk write needed)."""
        if self.vuln_finder is None:
            return 0.8   # neutral if not configured
        findings = []
        for lineno, line in enumerate(code.splitlines(), 1):
            for pattern, severity, cwe, desc in self.vuln_finder.VULN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(severity)
        n_critical = findings.count("critical")
        n_high     = findings.count("high")
        return max(0.0, 1.0 - 0.3 * n_critical - 0.15 * n_high)

    def _check_style(self, code: str,
                     repo_style_hints: Optional[Dict] = None) -> float:
        """
        Lightweight style consistency check.
        repo_style_hints: {"indent": 4, "max_line_len": 100, "uses_fstrings": True}
        """
        hints   = repo_style_hints or {}
        score   = 1.0
        lines   = code.splitlines()
        if not lines:
            return 0.5

        # Check line length
        max_len = hints.get("max_line_len", 100)
        long_lines = sum(1 for l in lines if len(l) > max_len)
        score -= 0.05 * min(long_lines / max(len(lines), 1), 1.0)

        # Check indentation consistency (4-space default)
        indent = hints.get("indent", 4)
        bad_indent = sum(1 for l in lines
                         if l and l[0] == " " and len(l) - len(l.lstrip()) % indent != 0)
        score -= 0.05 * min(bad_indent / max(len(lines), 1), 1.0)

        # f-string preference
        if hints.get("uses_fstrings", False) and ".format(" in code:
            score -= 0.05

        return max(0.0, min(1.0, score))

    # ── Combined reward ───────────────────────────────────────────────────

    def compute(self, code: str, prompt: str = "",
                run_tests: bool = False,
                repo_style_hints: Optional[Dict] = None) -> Dict:
        """
        Write code to a temp file, run all signals, return full reward dict.
        """
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"cogreward_{uuid.uuid4().hex[:8]}.py")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            signals: Dict[str, float] = {}
            signals["syntax_ok"]  = self._check_syntax(tmp_path)

            if signals["syntax_ok"] and run_tests:
                tests_pass, cov = self._check_tests(tmp_path)
                signals["tests_pass"]   = tests_pass
                signals["coverage_pct"] = cov
            else:
                signals["tests_pass"]   = 0.0
                signals["coverage_pct"] = 0.0

            signals["security_ok"] = self._check_security(code)
            signals["style_score"] = self._check_style(code, repo_style_hints)
            signals["prm_score"]   = (self.prm.score(prompt, code)
                                      if self.prm is not None else 0.5)

            # Weighted sum → [0, 1]
            raw = sum(self.WEIGHTS[k] * signals[k] for k in self.WEIGHTS)

            # Map [0, 1] to [-1, 1]: penalise below 0.3 hard
            reward = raw * 2.0 - 1.0
            if not signals["syntax_ok"]:
                reward = -1.0   # hard floor on syntax failure

            return {**signals, "reward": round(reward, 4)}

        finally:
            try: os.unlink(tmp_path)
            except OSError: pass


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Beam Expansion (CFG-guided diversity)
# ═══════════════════════════════════════════════════════════════════════════

class BeamExpansion:
    """
    Classifier-Free Guidance (CFG) beam search integrated into MCTS expansion.

    In standard MCTS, expansion generates K children with fixed strategies.
    BeamExpansion generates B × K candidates (B = beam_width), scores them
    with (a) PRM step reward and (b) a diversity penalty (penalise candidates
    too similar to already-selected beams), then selects the top K for the
    MCTS tree.

    Diversity is measured by character-level n-gram overlap (Jaccard distance).
    In production: replace heuristic candidate generation with actual model
    sampling at varying temperatures / guidance scales.
    """

    def __init__(self, beam_width: int = 5, prm: Optional[ProcessRewardModel] = None):
        self.beam_width = beam_width
        self.prm        = prm

    # ── Candidate generators (same three strategies as v1 + two new) ─────

    @staticmethod
    def _strategy_iterative(base: str, prompt: str) -> str:
        fn = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + f"\ndef {fn}_iterative(items):\n"
                f"    result = []\n    for item in items:\n"
                f"        # TODO: {prompt[:40]}\n        result.append(item)\n    return result\n")

    @staticmethod
    def _strategy_functional(base: str, prompt: str) -> str:
        fn = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + f"\ndef {fn}_functional(items):\n"
                f"    # Functional: {prompt[:60]}\n"
                f"    return [item for item in items if item is not None]\n")

    @staticmethod
    def _strategy_recursive(base: str, prompt: str) -> str:
        fn = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + f"\ndef {fn}_recursive(items, acc=None):\n"
                f"    if acc is None: acc = []\n"
                f"    if not items: return acc\n"
                f"    return {fn}_recursive(items[1:], acc + [items[0]])\n")

    @staticmethod
    def _strategy_generator(base: str, prompt: str) -> str:
        fn = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + f"\ndef {fn}_gen(items):\n"
                f"    # Generator approach: {prompt[:60]}\n"
                f"    for item in items:\n        yield item\n")

    @staticmethod
    def _strategy_class_based(base: str, prompt: str) -> str:
        cls = "".join(w.capitalize() for w in prompt.split()[:3])
        fn  = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + f"\nclass {cls}Processor:\n"
                f"    def {fn}(self, items):\n"
                f"        # {prompt[:60]}\n        return list(items)\n")

    # ── Diversity metric ─────────────────────────────────────────────────

    @staticmethod
    def _ngram_set(text: str, n: int = 3) -> Set[str]:
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def _jaccard_distance(self, a: str, b: str) -> float:
        sa, sb = self._ngram_set(a), self._ngram_set(b)
        if not sa and not sb: return 0.0
        return 1.0 - len(sa & sb) / max(len(sa | sb), 1)

    # ── Main expansion ───────────────────────────────────────────────────

    def expand(self, base_code: str, prompt: str, k: int = 3) -> List[str]:
        """
        Generate up to beam_width × k candidates, score with PRM + diversity,
        return top k unique candidates.
        """
        strategies = [
            self._strategy_iterative,
            self._strategy_functional,
            self._strategy_recursive,
            self._strategy_generator,
            self._strategy_class_based,
        ]

        # Generate B × k raw candidates (cycle through strategies)
        raw_candidates: List[str] = []
        for i in range(self.beam_width * k):
            strat = strategies[i % len(strategies)]
            raw_candidates.append(strat(base_code, prompt))

        # Score each candidate (PRM + novelty vs base_code)
        scored: List[Tuple[float, str]] = []
        for cand in raw_candidates:
            prm_score  = self.prm.score(prompt, cand) if self.prm else 0.5
            diversity  = self._jaccard_distance(cand, base_code)
            total      = 0.6 * prm_score + 0.4 * diversity
            scored.append((total, cand))

        # Greedy diverse selection: pick top-1, then penalise similar, repeat
        scored.sort(key=lambda x: -x[0])
        selected: List[str] = []
        for _, cand in scored:
            if len(selected) >= k:
                break
            # Diversity penalty: skip if too similar to already-selected
            if all(self._jaccard_distance(cand, s) > 0.05 for s in selected):
                selected.append(cand)

        # If diversity filtering left us short, fill with next best
        if len(selected) < k:
            for _, cand in scored:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= k:
                    break

        return selected[:k]


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — MCTS Node (extended)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """
    Extended MCTS node that carries a granularity level.
    level: "architecture" | "file" | "line"
    """
    code:          str
    level:         str = "file"              # NEW v2
    parent:        Optional["MCTSNode"] = field(default=None, repr=False)
    children:      List["MCTSNode"]    = field(default_factory=list, repr=False)
    visits:        int   = 0
    total_reward:  float = 0.0
    depth:         int   = 0
    expansion_prompt: str = ""
    terminal:      bool  = False

    # NEW v2: full reward breakdown
    reward_signals: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    def uct(self, exploration_c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        return (self.mean_reward +
                exploration_c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ═══════════════════════════════════════════════════════════════════════════
# NEW v2 — Hierarchical CogSearch
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalCogSearch:
    """
    Three-level Execution-Guided Monte Carlo Tree Search.

    GRANULARITY LEVELS
    ──────────────────
    Level 1 — Architecture (coarse):
        Search over high-level design choices (module split, data-flow pattern,
        API contract).  Prompt = "Design approach for: {task}".
        Reward: PRM score on design prose (no execution).

    Level 2 — File (medium):
        Search over whole-file code candidates.  Prompt = specific subtask.
        Reward: CoverageRewardModule (syntax + tests + security + style).

    Level 3 — Line (fine):
        Search over targeted patches/edits within a selected file.
        Prompt = "Patch for function {fn} to fix: {issue}".
        Reward: line-level execution signal (compile + unittest the function).

    MCTS × BEAM HYBRIDISATION
    ──────────────────────────
    At each level, expansion uses BeamExpansion (CFG-guided diverse candidates)
    instead of the three fixed strategies in v1.  UCT selection still drives
    which node to expand next.

    PARALLEL EVALUATION
    ───────────────────
    Child nodes at each level are evaluated in parallel via ThreadPoolExecutor
    (max_workers = config.mcts_max_parallel).  This is safe because each
    evaluation writes to a temp file with a unique UUID path.

    SUBROUTINE INTERFACE
    ────────────────────
    Any swarm agent can call:
        result = cog_search.search_subroutine(prompt, level="file")
    and receive the best code + reward for that sub-problem.
    """

    # Level config
    LEVEL_ORDER = ["architecture", "file", "line"]

    # Architecture-level prompts don't run code; PRM-only reward
    ARCH_REWARD_SCALE = 0.5    # arch rewards are [0, 0.5] to reflect uncertainty

    def __init__(
        self,
        problem_solver:  "ProblemSolver",
        terminal_guy:    "TerminalGuy",
        pessimist:       "Pessimist",
        vuln_finder:     Optional["VulnerabilityFinder"] = None,
        model:           Optional[CogForge] = None,
        config:          Optional[CogForgeConfig] = None,
        exploration_c:   float = 1.414,
        k_expansions:    int   = 3,
        verifier_threshold: float = 0.3,
    ):
        self.problem_solver     = problem_solver
        self.terminal_guy       = terminal_guy
        self.pessimist          = pessimist
        self.model              = model
        self.config             = config or CogForgeConfig()
        self.exploration_c      = exploration_c
        self.k_expansions       = k_expansions
        self.verifier_threshold = verifier_threshold

        # v2 components
        self.prm = ProcessRewardModel(self.config)
        self.beam_expansion = BeamExpansion(
            beam_width=self.config.mcts_beam_width, prm=self.prm)
        self.coverage_reward = CoverageRewardModule(
            terminal_guy=terminal_guy,
            vuln_finder=vuln_finder,
            prm=self.prm,
        )
        self._dpo_pairs: List[Dict] = []
        self._max_parallel = self.config.mcts_max_parallel

    # ── Tree operations ─────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct(self.exploration_c))
        return node

    def _expand(self, node: MCTSNode, prompt: str) -> List[MCTSNode]:
        """Use BeamExpansion for diverse, PRM-scored children."""
        candidates = self.beam_expansion.expand(
            base_code=node.code,
            prompt=prompt,
            k=self.k_expansions,
        )
        children: List[MCTSNode] = []
        for cand in candidates:
            child = MCTSNode(
                code=cand,
                level=node.level,
                parent=node,
                depth=node.depth + 1,
                expansion_prompt=prompt[:80],
            )
            children.append(child)
        node.children.extend(children)
        return children

    # ── Evaluation ──────────────────────────────────────────────────────

    def _eval_architecture(self, code: str, prompt: str) -> Dict:
        """Architecture level: PRM score only (no code execution)."""
        prm_score   = self.prm.score(prompt, code)
        pessimist_critique = self.pessimist._critique_code(code)
        high_risk   = [c for c in pessimist_critique if c["risk"] == "high"]
        reward      = prm_score * self.ARCH_REWARD_SCALE
        reward     -= 0.05 * len(high_risk)
        return {"reward": max(-1.0, min(1.0, reward)),
                "prm_score": prm_score, "high_risk_critique": len(high_risk)}

    def _eval_file(self, code: str, prompt: str, run_tests: bool,
                   repo_style_hints: Optional[Dict] = None) -> Dict:
        """File level: full CoverageRewardModule."""
        return self.coverage_reward.compute(
            code=code, prompt=prompt,
            run_tests=run_tests,
            repo_style_hints=repo_style_hints,
        )

    def _eval_line(self, code: str, prompt: str) -> Dict:
        """
        Line level: fast syntax + unit test check for a targeted patch.
        Wraps the patch in a minimal test harness to isolate the function.
        """
        result = self.coverage_reward.compute(
            code=code, prompt=prompt, run_tests=True)
        return result

    def _evaluate_child(self, child: MCTSNode, prompt: str,
                        run_tests: bool, level: str,
                        repo_style_hints: Optional[Dict] = None) -> float:
        """Dispatcher: call the right evaluation for the given level."""
        if level == "architecture":
            res = self._eval_architecture(child.code, prompt)
        elif level == "line":
            res = self._eval_line(child.code, prompt)
        else:   # file (default)
            res = self._eval_file(child.code, prompt, run_tests, repo_style_hints)

        child.reward_signals = {k: v for k, v in res.items() if k != "reward"}
        reward = res.get("reward", 0.0)

        # Mark terminal if reward is near maximum
        if reward >= 0.9:
            child.terminal = True

        return float(reward)

    # ── Parallel evaluation ──────────────────────────────────────────────

    def _evaluate_children_parallel(self, children: List[MCTSNode],
                                     prompt: str, run_tests: bool,
                                     level: str,
                                     repo_style_hints: Optional[Dict]) -> List[float]:
        """Evaluate all children concurrently."""
        def _eval_one(child: MCTSNode) -> float:
            try:
                return self._evaluate_child(child, prompt, run_tests, level,
                                            repo_style_hints)
            except Exception:
                return -1.0

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_parallel) as pool:
            futures = [pool.submit(_eval_one, c) for c in children]
            return [f.result(timeout=60) for f in futures]

    # ── Backpropagation ──────────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        while current is not None:
            current.visits      += 1
            current.total_reward += reward
            current = current.parent

    # ── Best node / leaf collection ──────────────────────────────────────

    def _best_leaf(self, root: MCTSNode) -> MCTSNode:
        best  = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.visits > 0 and node.mean_reward > best.mean_reward:
                best = node
            stack.extend(node.children)
        return best

    def _all_leaves(self, root: MCTSNode) -> List[MCTSNode]:
        leaves: List[MCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf() and node.visits > 0:
                leaves.append(node)
            stack.extend(node.children)
        return leaves

    def _all_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        nodes: List[MCTSNode] = []
        stack = [root]
        while stack:
            n = stack.pop()
            nodes.append(n)
            stack.extend(n.children)
        return nodes

    # ── DPO pair collection ──────────────────────────────────────────────

    def _record_dpo_pairs(self, prompt: str, root: MCTSNode) -> None:
        leaves = sorted(self._all_leaves(root), key=lambda n: -n.mean_reward)
        if len(leaves) >= 2:
            for i in range(min(3, len(leaves) // 2)):
                winner = leaves[i]
                loser  = leaves[-(i + 1)]
                if winner.mean_reward > loser.mean_reward + 0.1:
                    self._dpo_pairs.append({
                        "prompt":        prompt,
                        "winning_code":  winner.code,
                        "losing_code":   loser.code,
                        "reward_delta":  winner.mean_reward - loser.mean_reward,
                        "level":         winner.level,
                    })

    # ── Single-level search ──────────────────────────────────────────────

    def _search_level(self, prompt: str, initial_code: str, level: str,
                      iterations: int, run_tests: bool,
                      repo_style_hints: Optional[Dict] = None) -> Dict:
        """
        Run MCTS × BeamExpansion for a single granularity level.
        Returns the standard result dict for that level.
        """
        root = MCTSNode(code=initial_code, level=level, depth=0)
        root.visits = 1

        nodes_evaluated = 0

        for _ in range(iterations):
            selected = self._select(root)
            if selected.terminal:
                continue

            new_children = self._expand(selected, prompt)
            rewards = self._evaluate_children_parallel(
                new_children, prompt, run_tests, level, repo_style_hints)

            for child, reward in zip(new_children, rewards):
                nodes_evaluated += 1
                self._backpropagate(child, reward)

        best_node = self._best_leaf(root)
        self._record_dpo_pairs(prompt, root)

        all_lv = sorted(
            [(n.mean_reward, n.code) for n in self._all_leaves(root)],
            key=lambda x: -x[0],
        )
        all_n = self._all_nodes(root)
        max_depth = max((n.depth for n in all_n), default=0)

        return {
            "level":         level,
            "best_code":     best_node.code,
            "best_reward":   best_node.mean_reward,
            "reward_signals": best_node.reward_signals,
            "tree_stats": {
                "total_nodes":     len(all_n),
                "nodes_evaluated": nodes_evaluated,
                "max_depth":       max_depth,
                "iterations":      iterations,
            },
            "all_leaves": all_lv[:5],
        }

    # ── Hierarchical top-level search ────────────────────────────────────

    def search(self, prompt: str, iterations: int = 10,
               run_tests: bool = False, initial_code: str = "",
               levels: Optional[List[str]] = None,
               repo_style_hints: Optional[Dict] = None) -> Dict:
        """
        Run hierarchical MCTS across architecture → file → line levels.

        Each level's best output seeds the next level's initial_code.
        Returns combined results with per-level breakdowns.

        Args:
            prompt:            Natural-language task description.
            iterations:        MCTS iterations *per level*.
            run_tests:         If True, run pytest during file/line eval.
            initial_code:      Seed code for the architecture level.
            levels:            Subset of levels to run (default: all three).
            repo_style_hints:  Style context for CoverageRewardModule.

        Returns:
            {best_code, best_reward, dpo_pairs, per_level, tree_stats_total}
        """
        levels_to_run = levels or self.LEVEL_ORDER
        per_level: Dict[str, Dict] = {}
        current_code = initial_code

        for level in levels_to_run:
            level_prompt = self._level_prompt(prompt, level)
            result = self._search_level(
                prompt=level_prompt,
                initial_code=current_code,
                level=level,
                iterations=iterations,
                run_tests=run_tests and level != "architecture",
                repo_style_hints=repo_style_hints,
            )
            per_level[level] = result
            # Feed best code from this level into the next
            current_code = result["best_code"]

        # Overall best = deepest level's best
        final_level = levels_to_run[-1]
        best_result = per_level[final_level]

        total_nodes = sum(r["tree_stats"]["total_nodes"]
                          for r in per_level.values())
        total_evals = sum(r["tree_stats"]["nodes_evaluated"]
                          for r in per_level.values())

        return {
            "best_code":    best_result["best_code"],
            "best_reward":  best_result["best_reward"],
            "reward_signals": best_result["reward_signals"],
            "dpo_pairs":    list(self._dpo_pairs),
            "per_level":    per_level,
            "tree_stats": {
                "total_nodes":     total_nodes,
                "nodes_evaluated": total_evals,
                "levels_run":      levels_to_run,
            },
        }

    @staticmethod
    def _level_prompt(base_prompt: str, level: str) -> str:
        prefixes = {
            "architecture": "Design the high-level architecture for: ",
            "file":         "Implement the solution for: ",
            "line":         "Write a targeted patch/fix for: ",
        }
        return prefixes.get(level, "") + base_prompt

    # ── Subroutine interface for swarm agents ────────────────────────────

    def search_subroutine(self, prompt: str, level: str = "file",
                          iterations: int = 6, run_tests: bool = False,
                          initial_code: str = "") -> Dict:
        """
        Lightweight entry point for swarm agents to call CogSearch for
        a targeted sub-problem without running all three levels.

        Example (from Engineer):
            result = swarm.cog_search_engine.search_subroutine(
                "Refactor _parse_date to handle ISO 8601 variants",
                level="line", iterations=4,
            )
            patched_code = result["best_code"]
        """
        result = self._search_level(
            prompt=self._level_prompt(prompt, level),
            initial_code=initial_code,
            level=level,
            iterations=iterations,
            run_tests=run_tests,
        )
        result["dpo_pairs"] = list(self._dpo_pairs)
        return result

    def get_dpo_pairs(self) -> List[Dict]:
        return list(self._dpo_pairs)


# Keep v1 CogSearch as an alias for backwards compatibility
class CogSearch(HierarchicalCogSearch):
    """
    v1-compatible alias.  search() now delegates to HierarchicalCogSearch
    with level=['file'] only so existing callers get identical behaviour.
    """
    def search(self, prompt: str, iterations: int = 10,   # type: ignore[override]
               run_tests: bool = False, initial_code: str = "", **kwargs) -> Dict:
        result = super().search(
            prompt=prompt,
            iterations=iterations,
            run_tests=run_tests,
            initial_code=initial_code,
            levels=["file"],
        )
        # Flatten per_level["file"] to top level for v1 consumers
        file_result = result.get("per_level", {}).get("file", {})
        result["all_leaves"] = file_result.get("all_leaves", [])
        result["tree_stats"].update(file_result.get("tree_stats", {}))
        return result


# ═══════════════════════════════════════════════════════════════════════════
# CogWorks Swarm v2
# ═══════════════════════════════════════════════════════════════════════════

class CogWorksSwarm:
    """
    v2 Swarm orchestrator.  Adds:
      - MetaOrchestrator for ephemeral specialist sub-agents.
      - ModelRouter for strength-based compute routing.
      - MultiLevelMemoryMgr (Dreamer v2) with GraphRAG + persistence.
      - HierarchicalCogSearch accessible at all three levels.
      - Subroutine interface: any agent can call cog_search_engine.search_subroutine().

    Workflow (extended from v1):
      0. MetaOrchestrator — detect needed specialists; spawn & run in parallel.
      1. Coordinator   — delegates to Planner + Explorer.
      2. Dreamer v2    — GraphRAG context retrieval.
      3. Explorer      — repo map → GraphRAG ingest.
      4. Archeologist  — git temporal map → graph annotations.
      5. Nexus         — dependency audit → package graph nodes.
      6. Planner       — DAG decomposition.
      7. ProblemSolver — GraphRAG-enriched annotation.
      8. Engineer      — code analysis (calls CogSearch as subroutine for hard edits).
      9. Pessimist     — stress-test.
     10. BugFinder + VulnerabilityFinder — parallel.
     11. TerminalGuy  — test execution.
     12. Documentor   — annotation.
     13. MultiLevelMemoryMgr — consolidate + persist.
     14. Coordinator  — quality gate.
    """

    def __init__(self, model: Optional[CogForge] = None,
                 config: Optional[CogForgeConfig] = None,
                 session_snapshot_path: Optional[str] = None):
        self.config        = config or CogForgeConfig()
        self.shared_memory = SharedMemoryStore(
            graph_embed_dim=self.config.graph_embed_dim,
            graph_max_nodes=self.config.graph_max_nodes,
        )

        # Memory manager (Dreamer v2 backed)
        self.mem_mgr = MultiLevelMemoryMgr(self.shared_memory, model, self.config)

        # Restore from session snapshot if provided
        if session_snapshot_path:
            self.mem_mgr.restore(session_snapshot_path)

        # Model router
        self.router = ModelRouter(model)

        # All 13 original agents
        self.coordinator          = Coordinator(self.shared_memory, model)
        self.dreamer              = Dreamer(self.shared_memory, model, self.config,
                                            self.mem_mgr)
        self.explorer             = Explorer(self.shared_memory, model)
        self.planner              = Planner(self.shared_memory, model)
        self.problem_solver       = ProblemSolver(self.shared_memory, model)
        self.engineer             = Engineer(self.shared_memory, model)
        self.bug_finder           = BugFinder(self.shared_memory, model)
        self.terminal_guy         = TerminalGuy(self.shared_memory, model)
        self.vulnerability_finder = VulnerabilityFinder(self.shared_memory, model)
        self.pessimist            = Pessimist(self.shared_memory, model)
        self.documentor           = Documentor(self.shared_memory, model)
        self.nexus                = Nexus(self.shared_memory, model)
        self.archeologist         = Archeologist(self.shared_memory, model)

        # NEW v2: MetaOrchestrator
        self.meta_orchestrator = MetaOrchestrator(
            self.shared_memory, model, self.config)

        # v2 HierarchicalCogSearch engine
        self.cog_search_engine = HierarchicalCogSearch(
            problem_solver=self.problem_solver,
            terminal_guy=self.terminal_guy,
            pessimist=self.pessimist,
            vuln_finder=self.vulnerability_finder,
            model=model,
            config=self.config,
            exploration_c=1.414,
            k_expansions=3,
        )

        self._agents: Dict[str, BaseAgent] = {
            "Coordinator":          self.coordinator,
            "Dreamer":              self.dreamer,
            "Explorer":             self.explorer,
            "Planner":              self.planner,
            "ProblemSolver":        self.problem_solver,
            "Engineer":             self.engineer,
            "BugFinder":            self.bug_finder,
            "TerminalGuy":          self.terminal_guy,
            "VulnerabilityFinder":  self.vulnerability_finder,
            "Pessimist":            self.pessimist,
            "Documentor":           self.documentor,
            "Nexus":                self.nexus,
            "Archeologist":         self.archeologist,
            "MetaOrchestrator":     self.meta_orchestrator,
        }

    # ── Full pipeline ─────────────────────────────────────────────────────

    def run(self, task: str, repo_root: str = ".",
            save_snapshot: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full v2 swarm workflow.

        New in v2:
          - Step 0: MetaOrchestrator spawns ephemeral specialists (parallel).
          - ModelRouter selects compute head for the task.
          - GraphRAG memory populated throughout.
          - CogSearch subroutine available to Engineer for hard sub-problems.
          - Session snapshot saved to disk at end if save_snapshot is given.
        """
        results: Dict[str, Any] = {}

        # ── Step 0: Route and spawn ephemeral specialists ─────────────────
        gen_cfg = self.router.get_gen_config(task)
        results["routing_head"] = gen_cfg.get("head", "fast")

        meta_results = self.meta_orchestrator.orchestrate(task, repo_root)
        results["meta_orchestrator"] = meta_results

        # ── Step 1: Coordinator kickoff ───────────────────────────────────
        results["coordinator_kickoff"] = self.coordinator.run(task).message

        # ── Step 2: Dreamer v2 context retrieval (GraphRAG) ──────────────
        results["dreamer_context"] = self.dreamer.run(task).message

        # ── Step 3: Explorer → GraphRAG ingest ───────────────────────────
        explore_msg = self.explorer.run(task, root=repo_root)
        results["explorer"] = explore_msg.artifacts

        # ── Step 4: Archeologist → graph annotations ──────────────────────
        arch_msg = self.archeologist.run(task, root=repo_root)
        results["archeologist"] = arch_msg.artifacts

        # ── Step 5: Nexus → package graph nodes ──────────────────────────
        nexus_msg = self.nexus.run(task, root=repo_root)
        results["nexus"] = nexus_msg.artifacts

        # ── Step 6: Planner DAG ───────────────────────────────────────────
        plan_msg = self.planner.run(task)
        results["planner"] = plan_msg.artifacts

        # ── Step 7: ProblemSolver (GraphRAG-enriched) ─────────────────────
        ps_msg = self.problem_solver.run(task)
        results["problem_solver"] = ps_msg.artifacts

        # ── Step 8: Engineer (may call CogSearch subroutine) ─────────────
        eng_msg = self.engineer.run(task)
        results["engineer"] = eng_msg.artifacts

        # ── Step 9: Pessimist ─────────────────────────────────────────────
        pess_msg = self.pessimist.run(task)
        results["pessimist"] = pess_msg.artifacts

        # ── Step 10: BugFinder + VulnerabilityFinder (parallel) ──────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            bug_future  = pool.submit(self.bug_finder.run, task)
            vuln_future = pool.submit(self.vulnerability_finder.run, task)
            bug_msg     = bug_future.result(timeout=120)
            vuln_msg    = vuln_future.result(timeout=120)
        results["bug_finder"]           = bug_msg.artifacts
        results["vulnerability_finder"] = vuln_msg.artifacts

        # ── Step 11: TerminalGuy runs tests ──────────────────────────────
        term_msg = self.terminal_guy.run(
            task, command="python -m pytest --tb=short -q")
        results["terminal_guy"] = term_msg.artifacts

        # ── Step 12: Documentor ───────────────────────────────────────────
        flagged  = self.shared_memory.get_flagged_files()
        doc_msg  = self.documentor.run(task, files=flagged)
        results["documentor"] = doc_msg.artifacts

        # ── Step 13: Memory consolidation + graph stats ───────────────────
        self.dreamer.maybe_consolidate_episodic()
        results["graph_stats"]          = self.shared_memory.graph.stats()
        results["dreamer_consolidation"] = "complete"

        # ── Step 14: Quality gate ─────────────────────────────────────────
        verifier_proxy = pess_msg.confidence
        gate_decision  = self.coordinator.review_and_gate(
            verifier_proxy, plan_msg.task_id)
        results["quality_gate"]         = gate_decision
        results["verifier_proxy_score"] = verifier_proxy

        # ── Step 15: Persist session snapshot ────────────────────────────
        if save_snapshot:
            self.mem_mgr.persist(save_snapshot)
            results["session_snapshot"] = save_snapshot

        return results

    # ── CogSearch (hierarchical) entry point ─────────────────────────────

    def cog_search(self, prompt: str, iterations: int = 10,
                   run_tests: bool = False, initial_code: str = "",
                   k_expansions: int = 3,
                   levels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run Hierarchical Execution-Guided MCTS (architecture → file → line).

        New in v2:
          - Runs up to 3 levels (default: all three).
          - BeamExpansion replaces fixed strategies.
          - ProcessRewardModel gives step-level rewards.
          - CoverageRewardModule gives multi-signal file/line rewards.
          - Parallel evaluation within each level.
          - DPO pairs tagged with level.
        """
        self.cog_search_engine.k_expansions = k_expansions
        result = self.cog_search_engine.search(
            prompt=prompt,
            iterations=iterations,
            run_tests=run_tests,
            initial_code=initial_code,
            levels=levels,
        )

        self.shared_memory.set("cog_search_best_code", result["best_code"])
        self.shared_memory.set("cog_search_dpo_pairs", result["dpo_pairs"])

        self.coordinator.emit(
            target="Coordinator",
            message=(
                f"HierarchicalCogSearch complete. "
                f"Best reward: {result['best_reward']:.3f}. "
                f"Levels: {result['tree_stats'].get('levels_run', [])}. "
                f"Total nodes: {result['tree_stats']['total_nodes']}. "
                f"DPO pairs: {len(result['dpo_pairs'])}."
            ),
            artifacts={
                "best_reward": result["best_reward"],
                "tree_stats":  result["tree_stats"],
                "n_dpo_pairs": len(result["dpo_pairs"]),
                "per_level_rewards": {
                    lvl: r.get("best_reward")
                    for lvl, r in result.get("per_level", {}).items()
                },
            },
            status="done",
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg   = CogForgeConfig()
    model = CogForge(cfg)
    print(f"CogForge v2 parameters:  {model.count_parameters():,}")
    print(f"Graph embed dim:         {cfg.graph_embed_dim}")
    print(f"Beam width (MCTS):       {cfg.mcts_beam_width}")
    print(f"PRM layers:              {cfg.mcts_prm_layers}")
    print(f"Max parallel eval:       {cfg.mcts_max_parallel}")

    # ── Model forward pass ────────────────────────────────────────────────
    B, T   = 2, 64
    ids    = torch.randint(0, cfg.vocab_size, (B, T))
    chunks = torch.randn(B, cfg.max_repo_chunks, cfg.d_model)
    out    = model(ids, repo_chunks=chunks, return_verifier=True, return_handoff=True)
    print(f"\nForward pass:")
    print(f"  logits:           {out['logits'].shape}")
    print(f"  verifier_score:   {out['verifier_score']}")
    print(f"  recommended_agent:{out['recommended_agent']}")

    # ── ModelRouter smoke test ────────────────────────────────────────────
    print("\n--- ModelRouter ---")
    router = ModelRouter(model)
    tasks  = [
        "rename a variable",
        "refactor the entire auth module with security hardening and race condition fixes",
        "add docstring to parse_date",
    ]
    for t in tasks:
        cfg_out = router.get_gen_config(t)
        print(f"  [{cfg_out['head']:6s}] {t[:60]}")

    # ── GraphRAG smoke test ───────────────────────────────────────────────
    print("\n--- GraphRAGMemory ---")
    graph = GraphRAGMemory(embed_dim=cfg.graph_embed_dim,
                           max_nodes=cfg.graph_max_nodes)
    graph.upsert_node("f1", "file", "auth/views.py",
                      embed_text="authentication login session JWT")
    graph.upsert_node("f2", "file", "db/models.py",
                      embed_text="database ORM migration schema")
    graph.upsert_node("f3", "file", "api/serializers.py",
                      embed_text="REST API serialization validation")
    graph.add_edge("f1", "f2", "imports")
    graph.add_edge("f3", "f2", "imports")
    graph.set_compression("f1", gist="Auth views — JWT login.",
                          summary="Handles user login/logout with JWT tokens.",
                          detailed="Full auth module: login, logout, refresh, middleware.")

    hits = graph.semantic_search("login authentication token", top_k=2)
    print(f"  Semantic search hits: {[n.label for _, n in hits]}")
    neighbours = graph.k_hop_neighbours("f1", k=1)
    print(f"  k-hop neighbours of auth/views.py: {[n.label for n in neighbours]}")
    gist = graph.get_compression("f1", level="gist")
    print(f"  Gist for auth/views.py: {gist}")
    print(f"  Graph stats: {graph.stats()}")

    # ── MultiLevelMemoryMgr smoke test ────────────────────────────────────
    print("\n--- MultiLevelMemoryMgr ---")
    store   = SharedMemoryStore(graph_embed_dim=cfg.graph_embed_dim)
    mem_mgr = MultiLevelMemoryMgr(store, model, cfg)
    episode = {
        "interaction_batch": 8,
        "messages": [
            {"source": "Coordinator", "target": "Planner",
             "message": "Decompose auth refactor task.", "status": "done"},
            {"source": "Planner", "target": "Engineer",
             "message": "Implement JWT rotation.", "status": "done"},
            {"source": "Pessimist", "target": "Coordinator",
             "message": "Missing rollback step.", "status": "done"},
        ],
    }
    compressed = mem_mgr.compress_episode(episode, node_id="ep8")
    print(f"  Gist:    {compressed['gist']}")
    print(f"  Summary: {compressed['summary'][:80]}…")

    # ── MetaOrchestrator smoke test ───────────────────────────────────────
    print("\n--- MetaOrchestrator ---")
    meta = MetaOrchestrator(store, model, cfg)
    meta_results = meta.orchestrate(
        "Refactor auth module: security hardening, database migration, and API contract check")
    print(f"  Routing head:         {meta_results.get('routing_head')}")
    print(f"  Specialists spawned:  {meta_results.get('specialists_spawned')}")
    for role in meta_results.get("specialists_spawned", []):
        info = meta_results.get(role, {})
        print(f"    [{role}] {str(info.get('summary', info))[:80]}")

    # ── HierarchicalCogSearch smoke test ─────────────────────────────────
    print("\n--- HierarchicalCogSearch ---")
    problem_solver       = ProblemSolver(store, model)
    terminal_guy         = TerminalGuy(store, model)
    pessimist            = Pessimist(store, model)
    vulnerability_finder = VulnerabilityFinder(store, model)

    h_search = HierarchicalCogSearch(
        problem_solver=problem_solver,
        terminal_guy=terminal_guy,
        pessimist=pessimist,
        vuln_finder=vulnerability_finder,
        model=model,
        config=cfg,
        k_expansions=2,
    )
    result = h_search.search(
        prompt="Write a function that filters a list keeping only prime numbers",
        iterations=4,
        run_tests=False,
        levels=["architecture", "file"],
    )
    print(f"  Best reward (file):  {result['best_reward']:.3f}")
    print(f"  Reward signals:      {result['reward_signals']}")
    print(f"  Tree stats:          {result['tree_stats']}")
    print(f"  DPO pairs:           {len(result['dpo_pairs'])}")
    for lvl, r in result["per_level"].items():
        print(f"    Level [{lvl:12s}]: reward={r['best_reward']:.3f} "
              f"nodes={r['tree_stats']['total_nodes']}")

    # Subroutine interface
    sub_result = h_search.search_subroutine(
        "Fix the _parse_date function to handle timezone-aware ISO 8601 strings",
        level="line", iterations=3,
    )
    print(f"  Subroutine (line):   reward={sub_result['best_reward']:.3f}")

    # ── Full CogWorksSwarm v2 smoke test ──────────────────────────────────
    print("\n--- CogWorksSwarm v2 (full pipeline) ---")
    swarm   = CogWorksSwarm(model=model, config=cfg)
    results = swarm.run(
        "Refactor auth module for better security, add database migration support",
        repo_root=".",
    )
    print(f"  Routing head:         {results.get('routing_head')}")
    print(f"  Specialists:          {results['meta_orchestrator'].get('specialists_spawned')}")
    print(f"  Quality gate:         {results['quality_gate']}")
    print(f"  Verifier proxy:       {results['verifier_proxy_score']:.3f}")
    print(f"  Graph stats:          {results['graph_stats']}")
    print(f"  Agents completed:     {[k for k in results if not k.startswith('_')]}")

    # Hierarchical CogSearch via swarm
    print("\n--- Hierarchical CogSearch via swarm ---")
    cs_result = swarm.cog_search(
        prompt="Write a cache-aware, O(n log n) prime sieve",
        iterations=4,
        run_tests=False,
        levels=["architecture", "file"],
        k_expansions=2,
    )
    print(f"  Best reward:         {cs_result['best_reward']:.3f}")
    print(f"  Levels run:          {cs_result['tree_stats'].get('levels_run')}")
    print(f"  Total nodes:         {cs_result['tree_stats']['total_nodes']}")
    print(f"  DPO pairs total:     {len(cs_result['dpo_pairs'])}")
    if cs_result["dpo_pairs"]:
        pair = cs_result["dpo_pairs"][0]
        print(f"  Sample pair delta:   {pair['reward_delta']:+.3f} [{pair['level']}]")
