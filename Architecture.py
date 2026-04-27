"""
CogForge: A Coding & Reasoning Transformer (~100M parameters)
=============================================================
Architecture:
  - Grouped-Query Attention (GQA) with Sliding Window
  - Rotary Positional Embeddings (RoPE)
  - Adaptive Computation Time (ACT) blocks
  - Latent Reasoning Tokens (dynamic, recursive)
  - Linear "Look-back" Attention for long-range context (true O(T) recurrent)
  - Hierarchical Memory Module (Dreamer-managed persistent memory)
  - Architect cross-attention module (repo-level context)
  - SwiGLU feed-forward networks
  - RMSNorm throughout
  - Handoff Protocol for CogWorks swarm coordination

CogSearch: Execution-Guided Monte Carlo Tree Search
  - UCT-based node selection (exploitation vs exploration)
  - Multi-branch expansion via ProblemSolver
  - Two-level evaluation: VerifierHead (internal) + TerminalGuy (external)
  - Reward: -1.0 syntax error | 0.1 logic flaw | 1.0 passes tests
  - Backpropagation through code-state tree
  - DPO pair collection: (prompt, winning_code, losing_code)

CogWorks Swarm Agents:
  - Coordinator    : Overseer, task routing, quality gates
  - Dreamer        : Hierarchical memory management and consolidation
  - Explorer       : Repository mapping and flagging (CPG-aware)
  - Planner        : Task decomposition into DAGs
  - ProblemSolver  : Deep expert reasoning (drives MCTS expansion)
  - Engineer       : Code efficiency and clean-form refactoring
  - BugFinder      : Deep inspection and logical error hunting
  - TerminalGuy    : Sandboxed tool/command executor (MCTS oracle)
  - VulnerabilityFinder : Security specialist
  - Pessimist      : Devil's advocate, stress-tester (MCTS reward signal)
  - Documentor     : Comments, docstrings, READMEs
  - Nexus          : Dependency & API specialist, RAG over docs
  - Archeologist   : Git history & intent analyst, temporal latent map
"""

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Config
# ===========================================================================

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
    max_latent_tokens: int = 32       # upper bound for dynamic expansion
    n_act_iterations: int = 3
    act_threshold: float = 0.99

    # Hierarchical memory
    memory_size: int = 128            # number of memory slots
    memory_update_interval: int = 256 # tokens between memory updates
    n_memory_layers: int = 4          # how many blocks get memory read injection

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


# ===========================================================================
# Utility: RMSNorm
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ===========================================================================
# Rotary Positional Embeddings (RoPE)
# ===========================================================================

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


# ===========================================================================
# Grouped-Query Attention with Sliding Window
# ===========================================================================

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


# ===========================================================================
# Linear "Look-back" Attention  —  true O(T) recurrent formulation
# ===========================================================================

class LinearLookbackAttention(nn.Module):
    """
    Causal linear attention using a running recurrent state (S_t, z_t).
    Avoids the O(T^2 D^2) memory of the cumsum-over-outer-products approach
    by processing one chunk at a time and carrying state forward.

    For training we process the whole sequence in chunks of size `chunk`
    and accumulate gradients through the recurrent state.
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

        q = self.q_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, D).permute(0, 2, 1, 3)

        q, k = self._feature_map(q), self._feature_map(k)

        # Running state: S (B,H,D,D), z (B,H,D)
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)

        out_chunks: List[torch.Tensor] = []
        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            qt = q[:, :, t_start:t_end, :]   # (B,H,c,D)
            kt = k[:, :, t_start:t_end, :]
            vt = v[:, :, t_start:t_end, :]
            c = t_end - t_start

            # Within-chunk causal outputs using prefix-scan over chunk
            chunk_out = torch.zeros(B, H, c, D, device=x.device, dtype=x.dtype)
            S_run = S.clone()
            z_run = z.clone()
            for i in range(c):
                ki = kt[:, :, i, :]   # (B,H,D)
                vi = vt[:, :, i, :]
                # update running state with this token
                S_run = S_run + torch.einsum("bhd,bhe->bhde", ki, vi)
                z_run = z_run + ki
                # read from running state
                qi = qt[:, :, i, :]
                num = torch.einsum("bhd,bhde->bhe", qi, S_run)
                den = torch.einsum("bhd,bhd->bh", qi, z_run).unsqueeze(-1).clamp(min=1e-6)
                chunk_out[:, :, i, :] = num / den

            # Carry final state forward to next chunk
            S = S_run
            z = z_run
            out_chunks.append(chunk_out)

        out = torch.cat(out_chunks, dim=2)  # (B,H,T,D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

        gate = torch.sigmoid(self.gate(x))
        return self.o_proj(out) * gate


# ===========================================================================
# Hierarchical Memory Module
# ===========================================================================

class HierarchicalMemory(nn.Module):
    """
    Persistent external memory with:
      - memory_size learnable slots (B, memory_size, d_model)
      - Compressor: pools recent hidden states → memory update vector
      - Gated additive update with exponential decay
      - Cross-attention read head: query current hidden → memory slots
      - Compression head: distills current context into a summary token
        stored back into memory at consolidation time

    Dreamer calls .update() every N tokens; every block calls .read().
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.memory_size = config.memory_size
        D = config.d_model

        # Learnable initial memory slots (broadcast over batch)
        self.memory_init = nn.Parameter(torch.randn(1, config.memory_size, D) * 0.02)

        # Compressor: recent hidden → update vector
        self.compressor = nn.Sequential(
            nn.Linear(D, D),
            RMSNorm(D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Per-slot update gate
        self.update_gate = nn.Linear(D, 1)

        # Read head: cross-attention from hidden to memory
        n_heads = max(1, config.n_kv_heads // 2)
        d_head  = D // config.n_heads
        attn_dim = n_heads * d_head
        self.read_q  = nn.Linear(D, attn_dim, bias=False)
        self.read_k  = nn.Linear(D, attn_dim, bias=False)
        self.read_v  = nn.Linear(D, attn_dim, bias=False)
        self.read_o  = nn.Linear(attn_dim, D, bias=False)
        self.n_read_heads = n_heads
        self.read_d_head  = d_head

        # Compression head: summarizes context window → summary embedding
        self.compress_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        self.norm_mem  = RMSNorm(D)
        self.norm_read = RMSNorm(D)
        self.decay     = 0.1   # memory decay rate on update

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a fresh memory state for a new sequence."""
        return self.memory_init.expand(batch_size, -1, -1).clone()

    def update(self, recent_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        recent_hidden: (B, recent_T, D)  — the last chunk of hidden states
        memory:        (B, memory_size, D)
        Returns updated memory of same shape.
        """
        # Pool recent hidden to a single update vector
        pooled = recent_hidden.mean(dim=1)                          # (B, D)
        update_vec = self.compressor(pooled)                         # (B, D)

        # Scalar gate per slot (broadcast update across all slots equally;
        # could be made slot-specific with a larger projection)
        gate = torch.sigmoid(self.update_gate(pooled))              # (B, 1)
        gate = gate.unsqueeze(1)                                     # (B, 1, 1)

        # Gated additive update with decay
        memory = memory * (1.0 - gate * self.decay) + update_vec.unsqueeze(1) * gate
        return self.norm_mem(memory)

    def consolidate(self, full_hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Distill the full context into a summary token and write it into a
        dedicated memory slot (slot 0 is reserved for the latest summary).
        full_hidden: (B, T, D)
        """
        summary = self.compress_head(full_hidden.mean(dim=1))       # (B, D)
        memory = memory.clone()
        memory[:, 0, :] = summary
        return self.norm_mem(memory)

    def read(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Cross-attend from current hidden states to memory slots.
        hidden: (B, T, D)
        memory: (B, memory_size, D)
        Returns: (B, T, D) — memory-enriched residual
        """
        B, T, _ = hidden.shape
        H, D = self.n_read_heads, self.read_d_head

        q = self.read_q(self.norm_read(hidden)).view(B, T, H, D).transpose(1, 2)
        k = self.read_k(memory).view(B, self.memory_size, H, D).transpose(1, 2)
        v = self.read_v(memory).view(B, self.memory_size, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.read_o(out)


# ===========================================================================
# SwiGLU Feed-Forward
# ===========================================================================

class SwiGLUFFN(nn.Module):
    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.drop(self.down_proj(gate * up))


# ===========================================================================
# Adaptive Computation Time (ACT) Wrapper
# ===========================================================================

class ACTBlock(nn.Module):
    def __init__(self, block: nn.Module, d_model: int,
                 max_iter: int = 3, threshold: float = 0.99):
        super().__init__()
        self.block = block
        self.max_iter = max_iter
        self.threshold = threshold
        self.halting_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                **block_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device = x.device

        halted = torch.zeros(B, T, 1, device=device)
        accum_output = torch.zeros_like(x)
        remainder = torch.ones(B, T, 1, device=device)
        ponder_cost = torch.zeros(B, T, device=device)

        state = x
        for step in range(self.max_iter):
            h = torch.sigmoid(self.halting_proj(state))
            still_running = (halted < self.threshold).float()

            is_last = (step == self.max_iter - 1)
            if is_last:
                used_h = remainder * still_running
            else:
                used_h = h * still_running
                will_exceed = ((halted + used_h) > self.threshold).float()
                used_h = used_h * (1 - will_exceed) + remainder * will_exceed

            halted = halted + used_h
            ponder_cost += (used_h.squeeze(-1) * still_running.squeeze(-1))

            state, _ = self.block(state, memory=memory, **block_kwargs)
            accum_output = accum_output + used_h * state
            remainder = remainder - used_h

            if (halted >= self.threshold).all():
                break

        return accum_output, ponder_cost


# ===========================================================================
# Base Transformer Block (with memory read injection)
# ===========================================================================

class CogForgeBlock(nn.Module):
    """
    Single transformer block with:
      - Pre-norm GQA (sliding window)
      - Pre-norm Linear look-back attention (true O(T) recurrent)
      - Optional HierarchicalMemory read injection
      - Pre-norm SwiGLU FFN
    """

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
        x = x + self.drop(h) + self.drop(lb) * torch.sigmoid(self.mix_alpha)

        # Inject hierarchical memory read if available
        if self.inject_memory and memory is not None and hier_memory is not None:
            mem_read = hier_memory.read(x, memory)
            x = x + self.drop(mem_read) * torch.sigmoid(self.mem_beta)

        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, new_kv


# ===========================================================================
# Architect Encoder (Repo-Level Context)
# ===========================================================================

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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_arch_layers)
        self.out_proj = nn.Linear(config.arch_d_model, config.d_model)
        self.norm = RMSNorm(config.d_model)

    def forward(self, chunk_embeds: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.chunk_proj(chunk_embeds)
        h = self.encoder(h, src_key_padding_mask=chunk_mask)
        h = self.out_proj(h)
        return self.norm(h)


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


# ===========================================================================
# Verifier Head (V(R))
# ===========================================================================

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


# ===========================================================================
# Dynamic Latent Token Controller
# ===========================================================================

class DynamicLatentController(nn.Module):
    """
    Decides whether to expand the latent token budget based on task complexity.
    Uses a small MLP that scores the initial hidden state after the first block
    and optionally appends additional latent tokens up to max_latent_tokens.
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_latent_base   = config.n_latent_tokens
        self.max_latent      = config.max_latent_tokens
        D = config.d_model
        self.complexity_head = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid(),
        )
        # Extra latent token pool
        extra = config.max_latent_tokens - config.n_latent_tokens
        self.extra_latents = nn.Parameter(torch.randn(1, extra, D) * 0.02)

    def forward(self, x_after_first_block: torch.Tensor,
                base_latents: torch.Tensor) -> torch.Tensor:
        """
        x_after_first_block: (B, n_latent + T, D)
        base_latents:        (B, n_latent, D)
        Returns augmented latents (B, n_latent_chosen, D).
        """
        B = x_after_first_block.shape[0]
        # Score complexity from the initial latent representations
        latent_repr = x_after_first_block[:, :self.n_latent_base, :].mean(1)  # (B, D)
        complexity = self.complexity_head(latent_repr)                         # (B, 1)

        # Number of extra tokens proportional to complexity
        n_extra = int((complexity.mean().item()) *
                      (self.max_latent - self.n_latent_base))
        if n_extra == 0:
            return base_latents
        extra = self.extra_latents[:, :n_extra, :].expand(B, -1, -1)
        return torch.cat([base_latents, extra], dim=1)


# ===========================================================================
# Handoff Protocol
# ===========================================================================

@dataclass
class HandoffMessage:
    """
    Structured inter-agent message.  All fields are plain Python so they can
    be serialised to JSON and stored in the shared memory / Redis / disk.
    """
    task_id:       str
    source_agent:  str
    target_agent:  str
    status:        str                  # "pending" | "running" | "done" | "error"
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
        d = json.loads(s)
        return cls(**d)


# ===========================================================================
# Shared Memory Store  (in-process; swap for Redis/vector DB in production)
# ===========================================================================

class SharedMemoryStore:
    """
    Central key-value + append-log store used by all swarm agents.
    In production replace the dict with Redis + a vector DB for semantic lookup.
    """

    def __init__(self):
        self._store: Dict[str, Any]        = {}
        self._log:   List[HandoffMessage]  = []
        self._repo_map: Dict[str, Any]     = {}   # file → {flags, summary, complexity}
        self._episodic: List[Dict]         = []   # completed task summaries

    # ── Key-value ────────────────────────────────────────────────────────────
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    # ── Handoff log ──────────────────────────────────────────────────────────
    def log_handoff(self, msg: HandoffMessage) -> None:
        self._log.append(msg)

    def get_handoffs_for(self, agent: str) -> List[HandoffMessage]:
        return [m for m in self._log if m.target_agent == agent and m.status == "pending"]

    def mark_done(self, task_id: str) -> None:
        for m in self._log:
            if m.task_id == task_id:
                m.status = "done"

    # ── Repo map ─────────────────────────────────────────────────────────────
    def update_repo_map(self, path: str, info: Dict) -> None:
        self._repo_map[path] = info

    def get_repo_map(self) -> Dict[str, Any]:
        return self._repo_map

    def get_flagged_files(self) -> List[str]:
        return [p for p, info in self._repo_map.items() if info.get("flagged")]

    # ── Episodic memory ───────────────────────────────────────────────────────
    def add_episode(self, episode: Dict) -> None:
        self._episodic.append(episode)

    def get_recent_episodes(self, n: int = 5) -> List[Dict]:
        return self._episodic[-n:]


# ===========================================================================
# Full CogForge Model
# ===========================================================================

class CogForge(nn.Module):
    """
    CogForge with:
      - Dynamic latent tokens
      - Hierarchical memory (Dreamer-managed) injected at select layers
      - Handoff output head (structured JSON handoff per forward pass)
      - Expanded Architect cross-attention at layers 4, 8, and 10
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model,
                                  padding_idx=config.pad_token_id)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, config.d_model) * 0.02
        )

        # Dynamic latent controller
        self.latent_controller = DynamicLatentController(config)

        # Hierarchical memory module (shared across blocks, managed by Dreamer)
        self.hier_memory = HierarchicalMemory(config)

        # Determine which layers receive memory injection
        # (last n_memory_layers blocks)
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

        # Architect at layers 4, 8, 10
        self.architect_encoder = ArchitectEncoder(config)
        self.arch_cross_attn = nn.ModuleDict({
            "4":  ArchitectCrossAttention(config),
            "8":  ArchitectCrossAttention(config),
            "10": ArchitectCrossAttention(config),
        })

        self.norm_out = RMSNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.verifier = VerifierHead(config)

        # Handoff head: projects pooled hidden → small embedding used to
        # produce a structured handoff JSON target recommendation
        self.handoff_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, len(AGENT_NAMES)),  # logit per agent
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

    def forward(
        self,
        input_ids: torch.Tensor,
        repo_chunks: Optional[torch.Tensor] = None,
        repo_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[List] = None,
        memory_state: Optional[torch.Tensor] = None,   # (B, memory_size, D)
        return_verifier: bool = False,
        return_handoff: bool = False,
    ) -> Dict:
        B, T = input_ids.shape
        device = input_ids.device

        # ── Embed tokens ──────────────────────────────────────────────────
        x = self.drop(self.embed(input_ids))

        # ── Base latent tokens ────────────────────────────────────────────
        base_latents = self.latent_tokens.expand(B, -1, -1)

        # ── Architect context ─────────────────────────────────────────────
        arch_ctx = None
        if repo_chunks is not None:
            arch_ctx = self.architect_encoder(repo_chunks, repo_mask)

        # ── Hierarchical memory state ─────────────────────────────────────
        if memory_state is None:
            memory_state = self.hier_memory.init_memory(B, device)

        # ── Initial prepend of latents ────────────────────────────────────
        x = torch.cat([base_latents, x], dim=1)  # (B, n_latent + T, D)

        # ── Transformer blocks (with dynamic latent expansion after block 0) ─
        new_kvs = []
        total_ponder = torch.zeros(B, device=device)
        block_past_kvs = past_kvs or [None] * len(self.blocks)
        new_memory = memory_state

        for i, block in enumerate(self.blocks):
            layer_key = str(i)

            # Architect cross-attention injection
            if arch_ctx is not None and layer_key in self.arch_cross_attn:
                x = self.arch_cross_attn[layer_key](x, arch_ctx, repo_mask)

            # After first block, dynamically expand latent budget if needed
            if i == 1:
                expanded = self.latent_controller(x, base_latents)
                n_extra = expanded.shape[1] - base_latents.shape[1]
                if n_extra > 0:
                    x = torch.cat([expanded[:, base_latents.shape[1]:, :], x], dim=1)

            # Periodic memory update every memory_update_interval tokens
            if i % max(1, len(self.blocks) // 4) == 0 and i > 0:
                # Slice off latent prefix before updating memory
                n_lat = x.shape[1] - T
                hidden_no_latent = x[:, n_lat:, :]
                new_memory = self.hier_memory.update(hidden_no_latent, new_memory)

            if i in self.act_blocks:
                # ACT blocks: pass memory for injection inside their inner block
                x, ponder = block(x, memory=new_memory, hier_memory=self.hier_memory,
                                  past_kv=block_past_kvs[i])
                total_ponder = total_ponder + ponder.mean(dim=1)
                new_kvs.append(None)
            else:
                x, kv = block(x, past_kv=block_past_kvs[i],
                              memory=new_memory, hier_memory=self.hier_memory)
                new_kvs.append(kv)

        # ── Consolidate memory after full forward pass ─────────────────────
        n_lat = x.shape[1] - T
        hidden_no_latent = x[:, n_lat:, :]
        new_memory = self.hier_memory.consolidate(hidden_no_latent, new_memory)

        # ── Remove all latent tokens before output ─────────────────────────
        x = x[:, n_lat:, :]   # (B, T, D)
        x = self.norm_out(x)

        logits = self.lm_head(x)

        out: Dict[str, Any] = {
            "logits":       logits,
            "ponder_cost":  total_ponder,
            "past_kvs":     new_kvs,
            "memory_state": new_memory,
        }

        if return_verifier:
            attn_mask = (input_ids != self.config.pad_token_id).float()
            out["verifier_score"] = self.verifier(x, attn_mask)

        if return_handoff:
            pooled = x.mean(1)           # (B, D)
            logits_agent = self.handoff_proj(pooled)  # (B, n_agents)
            recommended_agent_idx = logits_agent.argmax(dim=-1)  # (B,)
            out["handoff_agent_logits"] = logits_agent
            out["recommended_agent"] = [AGENT_NAMES[i] for i in
                                        recommended_agent_idx.tolist()]

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repo_chunks: Optional[torch.Tensor] = None,
        repo_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()
        past_kvs = None

        for _ in range(max_new_tokens):
            out = self.forward(
                generated if past_kvs is None else generated[:, -1:],
                repo_chunks=repo_chunks,
                repo_mask=repo_mask,
                past_kvs=past_kvs,
                memory_state=memory_state,
            )
            logits = out["logits"][:, -1, :]
            past_kvs = out["past_kvs"]
            memory_state = out["memory_state"]

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                topk_vals = torch.topk(logits, top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cum_probs - sorted_probs > top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, 1)
            next_token = sorted_idx.gather(-1, sampled)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Agent name registry  (used by handoff head)
# ===========================================================================

AGENT_NAMES: List[str] = [
    "Coordinator",
    "Dreamer",
    "Explorer",
    "Planner",
    "ProblemSolver",
    "Engineer",
    "BugFinder",
    "TerminalGuy",
    "VulnerabilityFinder",
    "Pessimist",
    "Documentor",
    "Nexus",
    "Archeologist",
]


# ===========================================================================
# Base Agent
# ===========================================================================

class BaseAgent:
    """
    All CogWorks agents inherit from this.  Each agent holds a reference to
    the shared memory store and optionally a CogForge model instance.
    Agents communicate exclusively through HandoffMessage objects.
    """

    role: str = "BaseAgent"

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional[CogForge] = None):
        self.memory  = memory_store
        self.model   = model
        self.task_id = str(uuid.uuid4())

    def emit(self, target: str, message: str,
             artifacts: Optional[Dict] = None,
             confidence: float = 1.0,
             status: str = "pending") -> HandoffMessage:
        """Create and log a handoff to another agent."""
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

    def receive(self) -> List[HandoffMessage]:
        """Collect all pending handoffs directed at this agent."""
        msgs = self.memory.get_handoffs_for(self.role)
        for m in msgs:
            m.status = "running"
        return msgs

    def run(self, task: str, **kwargs) -> HandoffMessage:
        raise NotImplementedError


# ===========================================================================
# 1. Coordinator
# ===========================================================================

class Coordinator(BaseAgent):
    """
    Overseer agent.  Receives high-level user tasks, breaks them into
    top-level subtasks, delegates to Planner + Explorer in parallel,
    monitors progress via the shared memory log, and enforces quality gates
    (verifier_score threshold).  Escalates to human if stalled.
    """

    role = "Coordinator"
    VERIFIER_THRESHOLD = 0.75

    def run(self, task: str, **kwargs) -> HandoffMessage:
        # Log the incoming task
        self.memory.set("current_task", task)
        self.memory.set("coordinator_status", "running")

        # Parallel kickoff: Planner decomposes, Explorer maps the repo
        plan_msg = self.emit(
            target="Planner",
            message=f"Decompose the following task into a DAG of subtasks:\n{task}",
            artifacts={"raw_task": task},
        )
        explore_msg = self.emit(
            target="Explorer",
            message="Map the repository and flag complex or risky areas.",
            artifacts={"raw_task": task},
        )

        # Record delegation
        self.memory.set("plan_task_id", plan_msg.task_id)
        self.memory.set("explore_task_id", explore_msg.task_id)

        return self.emit(
            target="Coordinator",          # self-loop: coordinator monitors
            message="Delegated to Planner and Explorer. Awaiting results.",
            status="running",
            artifacts={"plan_task_id": plan_msg.task_id,
                       "explore_task_id": explore_msg.task_id},
        )

    def review_and_gate(self, verifier_score: float, task_id: str) -> str:
        """
        Quality gate: return 'approve', 'iterate', or 'escalate'.
        """
        if verifier_score >= self.VERIFIER_THRESHOLD:
            self.memory.mark_done(task_id)
            return "approve"
        elif verifier_score >= 0.4:
            return "iterate"
        else:
            return "escalate"

    def assign(self, subtask: str, target_agent: str,
               artifacts: Optional[Dict] = None) -> HandoffMessage:
        """Explicitly assign a named subtask to a specific agent."""
        return self.emit(
            target=target_agent,
            message=subtask,
            artifacts=artifacts or {},
        )


# ===========================================================================
# 2. Dreamer  (Memory Management Specialist)
# ===========================================================================

class Dreamer(BaseAgent):
    """
    Manages the HierarchicalMemory state across the full swarm session.
    Runs a background consolidation loop: every N agent interactions it
    compresses recent outputs into long-term memory, updates the repo map
    summaries, and stores an episodic record.

    Other agents call dreamer.query(text) to retrieve relevant past context.
    """

    role = "Dreamer"
    CONSOLIDATE_EVERY = 8    # consolidate after this many logged handoffs

    def __init__(self, memory_store: SharedMemoryStore,
                 model: Optional[CogForge] = None,
                 config: Optional[CogForgeConfig] = None):
        super().__init__(memory_store, model)
        self.config    = config or CogForgeConfig()
        # Dreamer maintains the single authoritative memory tensor on CPU
        # (moved to the model device on demand)
        self.mem_state: Optional[torch.Tensor] = None
        self._interaction_count = 0

    def _ensure_mem(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        if self.mem_state is None:
            assert self.model is not None, "Dreamer needs a CogForge model."
            self.mem_state = self.model.hier_memory.init_memory(1, device)
        return self.mem_state

    def update_from_hidden(self, recent_hidden: torch.Tensor) -> None:
        """
        Called after any agent produces hidden states (e.g., after a CogForge
        forward pass).  Updates the persistent memory state.
        """
        assert self.model is not None
        mem = self._ensure_mem(recent_hidden.device)
        self.mem_state = self.model.hier_memory.update(recent_hidden, mem)

    def consolidate(self, full_hidden: torch.Tensor) -> None:
        """Full consolidation: write a summary token into slot 0."""
        assert self.model is not None
        mem = self._ensure_mem(full_hidden.device)
        self.mem_state = self.model.hier_memory.consolidate(full_hidden, mem)

    def query(self, query_hidden: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant context from memory given a query hidden state.
        Returns (1, T, D) memory-enriched residual.
        """
        assert self.model is not None
        mem = self._ensure_mem(query_hidden.device)
        return self.model.hier_memory.read(query_hidden, mem)

    def maybe_consolidate_episodic(self) -> None:
        """
        After every CONSOLIDATE_EVERY interactions compress logged agent outputs
        into the episodic store and update repo map summaries.
        """
        self._interaction_count += 1
        if self._interaction_count % self.CONSOLIDATE_EVERY != 0:
            return

        # Collect recent agent outputs from the handoff log
        recent = self.memory._log[-self.CONSOLIDATE_EVERY:]
        episode = {
            "interaction_batch": self._interaction_count,
            "messages": [
                {"source": m.source_agent, "target": m.target_agent,
                 "message": m.message[:200], "status": m.status}
                for m in recent
            ],
            "repo_map_snapshot": dict(self.memory.get_repo_map()),
        }
        self.memory.add_episode(episode)

        # Emit a summary back to Coordinator
        self.emit(
            target="Coordinator",
            message=(f"Memory consolidated after {self._interaction_count} interactions. "
                     f"Episode saved with {len(recent)} entries."),
            artifacts={"episode_index": len(self.memory._episodic)},
            status="done",
        )

    def run(self, task: str, **kwargs) -> HandoffMessage:
        """
        Dreamer's main entry point: consolidate and emit context summary.
        """
        self.maybe_consolidate_episodic()
        recent_eps = self.memory.get_recent_episodes(3)
        summary_lines = []
        for ep in recent_eps:
            for m in ep.get("messages", []):
                summary_lines.append(f"  [{m['source']}→{m['target']}] {m['message']}")
        summary = "\n".join(summary_lines) if summary_lines else "(no prior context)"

        return self.emit(
            target="Coordinator",
            message=f"Context summary from episodic memory:\n{summary}",
            artifacts={"n_episodes": len(self.memory._episodic)},
            status="done",
        )


# ===========================================================================
# 3. Explorer
# ===========================================================================

class Explorer(BaseAgent):
    """
    Maps the repository directory structure, computes per-file metrics
    (cyclomatic complexity proxy, line count, AST parse errors), and writes
    flags into the shared repo map.  Hands off flagged files to BugFinder
    and the full map to Planner/Dreamer.
    """

    role = "Explorer"
    COMPLEXITY_THRESHOLD = 10    # flag if estimated complexity > this
    LARGE_FILE_LINES     = 300   # flag files larger than this

    def _walk(self, root: str) -> List[str]:
        """Return all .py file paths under root."""
        py_files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".py"):
                    py_files.append(os.path.join(dirpath, fn))
        return py_files

    def _analyze_file(self, path: str) -> Dict:
        """Return basic metrics for a single Python file."""
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except OSError as e:
            return {"error": str(e), "flagged": True, "flag_reason": "unreadable"}

        line_count = source.count("\n")
        parse_error = False
        n_functions = 0
        max_depth = 0

        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n_functions += 1
                # Estimate nesting depth by counting nested defs/ifs
                if isinstance(node, (ast.If, ast.For, ast.While,
                                     ast.With, ast.Try)):
                    max_depth += 1
        except SyntaxError:
            parse_error = True

        # Proxy cyclomatic complexity: n_functions + max_depth
        complexity_score = n_functions + max_depth

        flagged = (
            parse_error
            or complexity_score > self.COMPLEXITY_THRESHOLD
            or line_count > self.LARGE_FILE_LINES
        )
        flag_reason = []
        if parse_error:
            flag_reason.append("syntax_error")
        if complexity_score > self.COMPLEXITY_THRESHOLD:
            flag_reason.append(f"high_complexity({complexity_score})")
        if line_count > self.LARGE_FILE_LINES:
            flag_reason.append(f"large_file({line_count}_lines)")

        return {
            "line_count":        line_count,
            "n_functions":       n_functions,
            "complexity_score":  complexity_score,
            "parse_error":       parse_error,
            "flagged":           flagged,
            "flag_reason":       ", ".join(flag_reason) if flag_reason else "ok",
        }

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        py_files = self._walk(root)
        flagged_files: List[str] = []

        for path in py_files:
            info = self._analyze_file(path)
            self.memory.update_repo_map(path, info)
            if info.get("flagged"):
                flagged_files.append(path)

        # Hand off flagged files to BugFinder and Vulnerability Finder
        if flagged_files:
            self.emit(
                target="BugFinder",
                message="Inspect the following flagged files for bugs.",
                artifacts={"flagged_files": flagged_files},
            )
            self.emit(
                target="VulnerabilityFinder",
                message="Scan the following flagged files for security vulnerabilities.",
                artifacts={"flagged_files": flagged_files},
            )

        # Inform Planner of the full map
        self.emit(
            target="Planner",
            message=f"Repository mapped: {len(py_files)} Python files, "
                    f"{len(flagged_files)} flagged.",
            artifacts={
                "total_files":   len(py_files),
                "flagged_files": flagged_files,
                "repo_map_keys": list(self.memory.get_repo_map().keys()),
            },
        )

        return self.emit(
            target="Coordinator",
            message=f"Exploration complete. {len(flagged_files)} files flagged.",
            artifacts={"flagged_files": flagged_files},
            status="done",
        )


# ===========================================================================
# 4. Planner
# ===========================================================================

class Planner(BaseAgent):
    """
    Breaks a high-level task into a directed acyclic graph (DAG) of subtasks.
    Each node has: id, description, dependencies (list of ids), assigned_agent,
    estimated_effort (1-5), and status.

    If a CogForge model is attached, it is used to generate the DAG via
    chain-of-thought reasoning.  Otherwise a heuristic decomposition is used.
    """

    role = "Planner"

    def _heuristic_dag(self, task: str) -> List[Dict]:
        """
        Produce a sensible default DAG for a generic code task.
        Returns list of node dicts.
        """
        nodes = [
            {"id": "t0", "description": f"Understand requirements: {task[:120]}",
             "dependencies": [], "assigned_agent": "ProblemSolver",
             "estimated_effort": 2, "status": "pending"},
            {"id": "t1", "description": "Identify affected files from repo map",
             "dependencies": ["t0"], "assigned_agent": "Explorer",
             "estimated_effort": 1, "status": "pending"},
            {"id": "t2", "description": "Design solution architecture and approach",
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
        return nodes

    def _validate_dag(self, dag: List[Dict]) -> bool:
        """Check for cycles using DFS."""
        graph: Dict[str, List[str]] = {n["id"]: n["dependencies"] for n in dag}
        visited: set = set()
        rec_stack: set = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for dep in graph.get(node_id, []):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(node_id)
            return False

        return not any(dfs(n["id"]) for n in dag if n["id"] not in visited)

    def run(self, task: str, **kwargs) -> HandoffMessage:
        dag = self._heuristic_dag(task)

        # Validate no cycles
        if not self._validate_dag(dag):
            # Flatten to a linear chain as fallback
            for i, node in enumerate(dag):
                node["dependencies"] = [dag[i-1]["id"]] if i > 0 else []

        self.memory.set("task_dag", dag)

        # Dispatch each node to its assigned agent
        for node in dag:
            self.emit(
                target=node["assigned_agent"],
                message=node["description"],
                artifacts={"dag_node": node},
            )

        # Hand off refined plan to ProblemSolver for resource analysis
        self.emit(
            target="ProblemSolver",
            message="Review the task DAG and refine solution strategy.",
            artifacts={"dag": dag, "task": task},
        )

        return self.emit(
            target="Coordinator",
            message=f"Task DAG created with {len(dag)} nodes.",
            artifacts={"dag": dag},
            status="done",
        )


# ===========================================================================
# 5. Problem Solver
# ===========================================================================

class ProblemSolver(BaseAgent):
    """
    Deep expert thinker.  Analyses the task DAG, available repo map,
    episodic memory, and constraints to produce a comprehensive solution path.
    Uses the verifier head internally (if model is present) to self-score
    candidate approaches.  Runs with higher ACT iterations conceptually
    (system-prompt level instruction; the model's ACT handles it internally).
    """

    role = "ProblemSolver"

    def _score_approach(self, description: str) -> float:
        """
        Simple heuristic scoring when no model is available.
        Returns a confidence in [0, 1].
        """
        positive_signals = ["test", "modular", "interface", "incremental", "rollback"]
        negative_signals = ["rewrite", "delete", "global", "hack", "skip"]
        score = 0.5
        desc_lower = description.lower()
        for s in positive_signals:
            if s in desc_lower:
                score += 0.05
        for s in negative_signals:
            if s in desc_lower:
                score -= 0.07
        return max(0.0, min(1.0, score))

    def run(self, task: str, **kwargs) -> HandoffMessage:
        dag    = self.memory.get("task_dag") or []
        repo   = self.memory.get_repo_map()
        recent = self.memory.get_recent_episodes(3)

        # Build a solution summary
        n_flagged = len(self.memory.get_flagged_files())
        n_files   = len(repo)

        # Analyse each DAG node and annotate with solution notes
        annotated_dag: List[Dict] = []
        for node in dag:
            notes = self._generate_notes(node, n_flagged, n_files)
            confidence = self._score_approach(notes)
            annotated_dag.append({**node, "solution_notes": notes,
                                   "confidence": confidence})

        self.memory.set("annotated_dag", annotated_dag)

        # Low-confidence nodes get escalated to Pessimist immediately
        risky = [n for n in annotated_dag if n["confidence"] < 0.45]
        if risky:
            self.emit(
                target="Pessimist",
                message="Stress-test these low-confidence subtasks before proceeding.",
                artifacts={"risky_nodes": risky},
            )

        return self.emit(
            target="Coordinator",
            message=(f"Solution path derived. {len(annotated_dag)} nodes annotated; "
                     f"{len(risky)} flagged for Pessimist review."),
            artifacts={"annotated_dag": annotated_dag},
            confidence=min(n["confidence"] for n in annotated_dag) if annotated_dag else 0.5,
            status="done",
        )

    def _generate_notes(self, node: Dict, n_flagged: int, n_files: int) -> str:
        desc = node.get("description", "")
        effort = node.get("estimated_effort", 3)
        lines = [
            f"Subtask: {desc}",
            f"Effort estimate: {effort}/5",
            f"Repo context: {n_files} files total, {n_flagged} flagged.",
        ]
        if effort >= 4:
            lines.append("High-effort task: break into smaller steps if possible.")
        if n_flagged > 5:
            lines.append("Many flagged files: proceed carefully, run tests after each change.")
        return "\n".join(lines)


# ===========================================================================
# 6. Engineer
# ===========================================================================

class Engineer(BaseAgent):
    """
    Focuses on finding the cleanest, most efficient form of the code.
    Given a code artifact it:
      1. Detects style and structural issues (long functions, duplicate logic,
         non-idiomatic patterns).
      2. Produces a refactoring plan with inline justifications.
      3. Emits suggestions to Pessimist for critique before finalising.
    """

    role = "Engineer"
    MAX_FUNC_LINES = 40   # functions longer than this are flagged for split

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
                    issues.append(
                        f"Function '{node.name}' is {length} lines long (>{self.MAX_FUNC_LINES})."
                    )
                    suggestions.append(
                        f"Split '{node.name}' into smaller helper functions."
                    )
                # Check for missing docstring
                if (not ast.get_docstring(node)):
                    issues.append(f"Function '{node.name}' is missing a docstring.")
                    suggestions.append(f"Add a docstring to '{node.name}'.")

        # Check for wildcard imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        issues.append(
                            f"Wildcard import from '{node.module}' — use explicit imports."
                        )

        # Check for duplicate blank lines (simple heuristic)
        blank_runs = 0
        prev_blank = False
        for line in lines:
            if line.strip() == "":
                if prev_blank:
                    blank_runs += 1
                prev_blank = True
            else:
                prev_blank = False
        if blank_runs > 3:
            issues.append(f"Excessive blank lines ({blank_runs} consecutive runs).")

        return {"issues": issues, "suggestions": suggestions, "refactored": source}

    def run(self, task: str, code: str = "", **kwargs) -> HandoffMessage:
        if not code:
            # Try to load from artifacts in incoming handoffs
            pending = self.receive()
            for msg in pending:
                code = msg.artifacts.get("code", "")
                if code:
                    break

        analysis = self._analyze_code(code) if code else {
            "issues": ["No code provided."], "suggestions": [], "refactored": ""
        }

        # Send for stress-testing by Pessimist
        self.emit(
            target="Pessimist",
            message="Critique this refactoring analysis and check for missed issues.",
            artifacts={"analysis": analysis, "code": code},
        )

        return self.emit(
            target="Coordinator",
            message=(f"Engineering analysis complete. "
                     f"{len(analysis['issues'])} issues found, "
                     f"{len(analysis['suggestions'])} suggestions made."),
            artifacts=analysis,
            confidence=max(0.3, 1.0 - 0.1 * len(analysis["issues"])),
            status="done",
        )


# ===========================================================================
# 7. Bug Finder
# ===========================================================================

class BugFinder(BaseAgent):
    """
    Deeply inspects flagged code for logical errors, runtime pitfalls,
    and incorrect assumptions.  Produces a structured bug report with
    severity, location, explanation, and suggested fix for each finding.
    Can request TerminalGuy to run tests.
    """

    role = "BugFinder"

    SEVERITY_PATTERNS: List[Tuple[str, str, str]] = [
        # (pattern_regex, severity, description)
        (r"except\s*:",            "high",   "Bare except clause swallows all exceptions"),
        (r"eval\s*\(",             "critical","eval() is dangerous with untrusted input"),
        (r"exec\s*\(",             "critical","exec() is dangerous with untrusted input"),
        (r"assert\s+",             "medium", "assert statements stripped in optimised mode"),
        (r"time\.sleep\s*\(",      "low",    "Blocking sleep in potentially async context"),
        (r"TODO|FIXME|HACK",       "low",    "Unresolved TODO/FIXME/HACK marker"),
        (r"\.format\s*\(",         "info",   "Consider f-strings for readability"),
        (r"global\s+\w+",          "medium", "Global variable mutation — prefer encapsulation"),
        (r"import \*",             "medium", "Wildcard import pollutes namespace"),
        (r"open\s*\([^)]*\)\s*$",  "high",  "File opened without context manager (no 'with')"),
    ]

    def _scan_source(self, source: str, filepath: str) -> List[Dict]:
        findings: List[Dict] = []
        lines = source.splitlines()

        for lineno, line in enumerate(lines, start=1):
            for pattern, severity, desc in self.SEVERITY_PATTERNS:
                if re.search(pattern, line):
                    findings.append({
                        "file":     filepath,
                        "line":     lineno,
                        "severity": severity,
                        "pattern":  pattern,
                        "description": desc,
                        "code_snippet": line.strip(),
                        "suggested_fix": self._suggest_fix(pattern, line.strip()),
                    })

        return findings

    def _suggest_fix(self, pattern: str, line: str) -> str:
        fixes = {
            r"except\s*:":           "Use `except Exception as e:` or a specific exception type.",
            r"eval\s*\(":            "Replace with ast.literal_eval() or a safe parser.",
            r"exec\s*\(":            "Remove exec(); refactor to importlib or subprocess.",
            r"assert\s+":            "Replace assert with explicit if/raise for production guards.",
            r"time\.sleep\s*\(":     "Use asyncio.sleep() in async code.",
            r"TODO|FIXME|HACK":      "Resolve the outstanding issue before merging.",
            r"\.format\s*\(":        f"Rewrite as: f\"{line}\".",
            r"global\s+\w+":         "Pass value as argument or use a class attribute instead.",
            r"import \*":            "List explicit names: `from module import name1, name2`.",
            r"open\s*\([^)]*\)\s*$": "Wrap in `with open(...) as f:` context manager.",
        }
        return fixes.get(pattern, "Review manually.")

    def run(self, task: str, **kwargs) -> HandoffMessage:
        # Collect files from handoff artifacts or repo map
        pending = self.receive()
        files_to_scan: List[str] = []
        for msg in pending:
            files_to_scan.extend(msg.artifacts.get("flagged_files", []))

        if not files_to_scan:
            files_to_scan = self.memory.get_flagged_files()

        all_findings: List[Dict] = []
        for filepath in files_to_scan:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                findings = self._scan_source(source, filepath)
                all_findings.extend(findings)
            except OSError:
                all_findings.append({
                    "file": filepath, "severity": "error",
                    "description": "Could not read file.", "line": 0,
                    "code_snippet": "", "suggested_fix": "", "pattern": "",
                })

        # Request TerminalGuy to run tests if any critical findings
        critical = [f for f in all_findings if f["severity"] == "critical"]
        if critical:
            self.emit(
                target="TerminalGuy",
                message="Run the test suite to confirm critical bugs.",
                artifacts={"command": "python -m pytest --tb=short -q",
                           "critical_findings": critical},
            )

        return self.emit(
            target="Coordinator",
            message=(f"Bug scan complete: {len(all_findings)} findings across "
                     f"{len(files_to_scan)} files. "
                     f"{len(critical)} critical."),
            artifacts={"findings": all_findings,
                       "files_scanned": len(files_to_scan)},
            confidence=max(0.1, 1.0 - 0.05 * len(all_findings)),
            status="done",
        )


# ===========================================================================
# 8. Terminal Guy
# ===========================================================================

class TerminalGuy(BaseAgent):
    """
    The only agent permitted to execute shell commands.
    Runs commands in a subprocess with a configurable timeout and captures
    stdout, stderr, and return code.  Risky commands (rm, drop, shutdown …)
    require explicit Coordinator approval stored in shared memory before
    TerminalGuy will execute them.

    All outputs are stored as artifacts back into shared memory and forwarded
    to the requesting agent.
    """

    role = "TerminalGuy"
    DEFAULT_TIMEOUT = 60        # seconds
    RISKY_PATTERNS  = re.compile(
        r"\b(rm\s+-rf|DROP\s+TABLE|shutdown|reboot|mkfs|dd\s+if=|"
        r"chmod\s+777|curl\s+.*\|\s*sh)\b",
        re.IGNORECASE,
    )

    def _is_approved(self, command: str, task_id: str) -> bool:
        approved = self.memory.get(f"approved_command:{task_id}", False)
        return approved or not bool(self.RISKY_PATTERNS.search(command))

    def _run_command(self, command: str, timeout: int) -> Dict:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "returncode": result.returncode,
                "stdout":     result.stdout[:4000],   # truncate for storage
                "stderr":     result.stderr[:2000],
                "timed_out":  False,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out.",
                    "timed_out": True}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e),
                    "timed_out": False}

    def run(self, task: str, command: str = "", timeout: int = DEFAULT_TIMEOUT,
            requester: str = "Coordinator", task_id: str = "", **kwargs) -> HandoffMessage:
        if not command:
            # Pull from handoff artifacts
            pending = self.receive()
            for msg in pending:
                command = msg.artifacts.get("command", "")
                requester = msg.source_agent
                task_id = msg.task_id
                if command:
                    break

        if not command:
            return self.emit(
                target=requester,
                message="No command provided to TerminalGuy.",
                status="error",
            )

        if not self._is_approved(command, task_id):
            # Escalate to Coordinator for approval
            self.emit(
                target="Coordinator",
                message=f"Risky command requires approval before execution:\n  `{command}`",
                artifacts={"command": command, "task_id": task_id},
            )
            return self.emit(
                target=requester,
                message="Command flagged as risky; awaiting Coordinator approval.",
                status="pending",
            )

        result = self._run_command(command, timeout)
        self.memory.set(f"terminal_result:{task_id}", result)

        return self.emit(
            target=requester,
            message=(f"Command `{command}` exited with code {result['returncode']}.\n"
                     f"stdout: {result['stdout'][:300]}"),
            artifacts=result,
            confidence=1.0 if result["returncode"] == 0 else 0.3,
            status="done",
        )


# ===========================================================================
# 9. Vulnerability Finder
# ===========================================================================

class VulnerabilityFinder(BaseAgent):
    """
    Security-focused scanner.  Checks for common vulnerability patterns
    (injection, hardcoded secrets, insecure defaults, dangerous functions)
    and reports each with a severity, CWE reference, explanation, and fix.
    """

    role = "VulnerabilityFinder"

    VULN_PATTERNS: List[Tuple[str, str, str, str]] = [
        # (regex, severity, cwe, description)
        (r"password\s*=\s*['\"][^'\"]+['\"]",
         "critical", "CWE-259", "Hardcoded password in source"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]",
         "critical", "CWE-321", "Hardcoded secret/API key"),
        (r"subprocess\.call\(.*shell\s*=\s*True",
         "high",     "CWE-78",  "Shell injection risk via shell=True"),
        (r"os\.system\s*\(",
         "high",     "CWE-78",  "Potential command injection via os.system"),
        (r"pickle\.loads?\s*\(",
         "high",     "CWE-502", "Unsafe deserialisation with pickle"),
        (r"yaml\.load\s*\([^,)]+\)",
         "high",     "CWE-502", "Unsafe YAML load without Loader"),
        (r"request\.args\.get|request\.form\.get",
         "medium",   "CWE-20",  "Unvalidated user input from request"),
        (r"md5\s*\(|hashlib\.md5",
         "medium",   "CWE-327", "Weak MD5 hash algorithm"),
        (r"sha1\s*\(|hashlib\.sha1",
         "medium",   "CWE-327", "Weak SHA-1 hash algorithm"),
        (r"random\.random\(\)|random\.randint",
         "low",      "CWE-338", "Non-cryptographic RNG used"),
        (r"DEBUG\s*=\s*True",
         "medium",   "CWE-215", "Debug mode enabled in source"),
        (r"ALLOWED_HOSTS\s*=\s*\[.*\*.*\]",
         "high",     "CWE-183", "Wildcard ALLOWED_HOSTS (Django)"),
        (r"verify\s*=\s*False",
         "high",     "CWE-295", "SSL verification disabled"),
    ]

    def _scan(self, source: str, filepath: str) -> List[Dict]:
        findings: List[Dict] = []
        lines = source.splitlines()
        for lineno, line in enumerate(lines, start=1):
            for pattern, severity, cwe, desc in self.VULN_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "file":        filepath,
                        "line":        lineno,
                        "severity":    severity,
                        "cwe":         cwe,
                        "description": desc,
                        "snippet":     line.strip(),
                        "fix":         self._fix(pattern),
                    })
        return findings

    def _fix(self, pattern: str) -> str:
        fixes = {
            r"password\s*=\s*['\"][^'\"]+['\"]":
                "Store passwords in environment variables or a secrets manager.",
            r"secret\s*=\s*['\"][^'\"]+['\"]":
                "Use os.environ or a vault (e.g., HashiCorp Vault, AWS Secrets Manager).",
            r"subprocess\.call\(.*shell\s*=\s*True":
                "Use shell=False and pass args as a list.",
            r"os\.system\s*\(":
                "Replace with subprocess.run([...], check=True).",
            r"pickle\.loads?\s*\(":
                "Use json or msgpack for safe serialisation.",
            r"yaml\.load\s*\([^,)]+\)":
                "Use yaml.safe_load() instead.",
            r"request\.args\.get|request\.form\.get":
                "Validate and sanitise all user inputs with a schema library.",
            r"md5\s*\(|hashlib\.md5":
                "Use hashlib.sha256() or better.",
            r"sha1\s*\(|hashlib\.sha1":
                "Use hashlib.sha256() or better.",
            r"random\.random\(\)|random\.randint":
                "Use secrets module for security-sensitive randomness.",
            r"DEBUG\s*=\s*True":
                "Disable debug mode in production; use environment-specific config.",
            r"ALLOWED_HOSTS\s*=\s*\[.*\*.*\]":
                "List explicit hostnames in ALLOWED_HOSTS.",
            r"verify\s*=\s*False":
                "Remove verify=False; provide the correct CA bundle instead.",
        }
        return fixes.get(pattern, "Review and remediate manually per the CWE reference.")

    def run(self, task: str, **kwargs) -> HandoffMessage:
        pending = self.receive()
        files_to_scan: List[str] = []
        for msg in pending:
            files_to_scan.extend(msg.artifacts.get("flagged_files", []))
        if not files_to_scan:
            files_to_scan = self.memory.get_flagged_files()

        all_findings: List[Dict] = []
        for filepath in files_to_scan:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                all_findings.extend(self._scan(source, filepath))
            except OSError:
                continue

        critical = [f for f in all_findings if f["severity"] == "critical"]

        # Forward critical and high findings to Engineer for remediation
        if all_findings:
            self.emit(
                target="Engineer",
                message="Security vulnerabilities found — remediation required.",
                artifacts={"vuln_findings": all_findings},
            )

        return self.emit(
            target="Coordinator",
            message=(f"Vulnerability scan complete: {len(all_findings)} findings, "
                     f"{len(critical)} critical across {len(files_to_scan)} files."),
            artifacts={"findings": all_findings},
            confidence=max(0.1, 1.0 - 0.08 * len(critical)),
            status="done",
        )


# ===========================================================================
# 10. Pessimist
# ===========================================================================

class Pessimist(BaseAgent):
    """
    Devil's advocate.  Receives a plan, code change, or architectural decision
    and systematically tries to find:
      - Edge cases and boundary conditions
      - Failure modes and race conditions
      - Over-optimistic assumptions
      - Missing error handling
      - Performance cliffs
      - Incomplete test coverage

    Returns a structured critique with each issue labelled by risk level.
    """

    role = "Pessimist"

    FAILURE_MODE_CHECKS: List[Tuple[str, str]] = [
        ("empty input",         "What happens when the primary input is empty or None?"),
        ("max scale",           "Does this hold at 10x / 100x the expected volume?"),
        ("concurrent access",   "Is there a race condition if two agents/threads hit this simultaneously?"),
        ("partial failure",     "What if a downstream service call fails halfway through?"),
        ("stale cache",         "Could cached data become invalid and cause silent incorrect results?"),
        ("integer overflow",    "Are numeric values bounded? Could they overflow or cause index errors?"),
        ("unicode/encoding",    "Are all string paths safe with non-ASCII input?"),
        ("missing rollback",    "If the operation fails partway, can the system recover cleanly?"),
        ("dependency version",  "Could a library update silently break assumed behaviour?"),
        ("logging gap",         "Are errors and unexpected branches logged sufficiently for debugging?"),
    ]

    def _critique_code(self, code: str) -> List[Dict]:
        issues: List[Dict] = []
        if not code.strip():
            return [{"risk": "high", "check": "empty code", "detail": "No code to analyse."}]

        for check, question in self.FAILURE_MODE_CHECKS:
            # Heuristic: look for absence of handling patterns
            risk = "low"
            detail = question

            if check == "empty input" and "if not " not in code and "is None" not in code:
                risk = "medium"
                detail = "No visible None/empty guards — " + question

            elif check == "concurrent access" and "lock" not in code.lower() and \
                    "mutex" not in code.lower() and "threading" not in code.lower():
                risk = "medium"
                detail = "No locking primitives found — " + question

            elif check == "partial failure" and "rollback" not in code.lower() and \
                    "transaction" not in code.lower():
                risk = "medium"
                detail = "No rollback/transaction logic — " + question

            elif check == "missing rollback" and "try" not in code and "except" not in code:
                risk = "high"
                detail = "No try/except found — " + question

            elif check == "logging gap" and \
                    "logging" not in code and "logger" not in code:
                risk = "low"
                detail = "No logging calls detected — " + question

            issues.append({"risk": risk, "check": check, "detail": detail})

        return issues

    def _critique_plan(self, dag: List[Dict]) -> List[Dict]:
        critiques: List[Dict] = []
        if not dag:
            return [{"risk": "high", "check": "empty plan", "detail": "No plan provided."}]

        # Check for missing test step
        has_test = any("test" in n.get("description", "").lower() for n in dag)
        if not has_test:
            critiques.append({
                "risk": "high", "check": "missing tests",
                "detail": "No testing step in the plan. Bugs will reach production."
            })

        # Check for missing rollback step
        has_rollback = any("rollback" in n.get("description", "").lower() or
                           "revert" in n.get("description", "").lower() for n in dag)
        if not has_rollback:
            critiques.append({
                "risk": "medium", "check": "missing rollback",
                "detail": "No rollback step — if deployment fails there's no recovery path."
            })

        # Flag nodes with effort=5 and no dependencies (unrealistic)
        for n in dag:
            if n.get("estimated_effort", 0) >= 5 and not n.get("dependencies"):
                critiques.append({
                    "risk": "medium", "check": "isolated high-effort node",
                    "detail": (f"Node '{n['id']}' has max effort but no dependencies — "
                               "it may be under-specified.")
                })

        return critiques

    def run(self, task: str, code: str = "", **kwargs) -> HandoffMessage:
        pending = self.receive()
        dag  = self.memory.get("annotated_dag") or []
        code = code

        for msg in pending:
            code = code or msg.artifacts.get("code", "")
            if "dag" in msg.artifacts:
                dag = msg.artifacts["dag"]

        code_critiques = self._critique_code(code)
        plan_critiques = self._critique_plan(dag)
        all_critiques  = code_critiques + plan_critiques

        high_risk = [c for c in all_critiques if c["risk"] == "high"]

        return self.emit(
            target="Coordinator",
            message=(f"Pessimist critique complete: {len(all_critiques)} issues found, "
                     f"{len(high_risk)} high-risk."),
            artifacts={"critiques": all_critiques, "high_risk": high_risk},
            confidence=max(0.1, 1.0 - 0.1 * len(high_risk)),
            status="done",
        )


# ===========================================================================
# 11. Documentor
# ===========================================================================

class Documentor(BaseAgent):
    """
    Adds module-level, class-level, and function-level docstrings to Python
    source files.  Also updates inline comments on complex logic blocks and
    generates a brief README section summarising the module's purpose.

    Uses AST to identify missing docstrings; writes them using a template
    based on function signature inspection.  Optionally uses an attached
    CogForge model to generate richer natural-language descriptions.
    """

    role = "Documentor"

    def _make_docstring(self, node: ast.AST) -> str:
        """Generate a starter docstring from an AST node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            returns = ""
            if node.returns:
                try:
                    returns = ast.unparse(node.returns)
                except Exception:
                    returns = "?"
            lines = [
                f"{node.name}.",
                "",
                "Args:",
            ]
            for arg in args:
                if arg == "self":
                    continue
                lines.append(f"    {arg}: Description.")
            if returns and returns not in ("None", ""):
                lines.append("")
                lines.append(f"Returns:")
                lines.append(f"    {returns}: Description.")
            return "\n".join(lines)

        if isinstance(node, ast.ClassDef):
            return f"{node.name}.\n\nAttributes:\n    (fill in attributes here)"

        if isinstance(node, ast.Module):
            return "Module docstring.\n\nThis module provides ...\n"

        return "TODO: add description."

    def _document_source(self, source: str) -> Tuple[str, int]:
        """
        Parse source, find nodes missing docstrings, inject them.
        Returns (modified_source, n_added).
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        lines = source.splitlines(keepends=True)
        insertions: List[Tuple[int, str]] = []   # (line_index, text_to_insert)

        def needs_docstring(node: ast.AST) -> bool:
            body = getattr(node, "body", [])
            if not body:
                return True
            first = body[0]
            return not (isinstance(first, ast.Expr) and
                        isinstance(first.value, ast.Constant) and
                        isinstance(first.value.value, str))

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                if needs_docstring(node):
                    # Insert after the def/class line
                    insert_at = node.lineno   # 1-based; we insert after this line
                    # Detect indentation from first body statement
                    body_line = node.body[0].lineno - 1 if node.body else node.lineno
                    indent = ""
                    if body_line < len(lines):
                        raw = lines[body_line]
                        indent = " " * (len(raw) - len(raw.lstrip()))
                    else:
                        indent = "    "
                    docstr = self._make_docstring(node)
                    formatted = (indent + '"""' + docstr.split("\n")[0] + "\n"
                                 + "\n".join(indent + l for l in docstr.split("\n")[1:])
                                 + "\n" + indent + '"""' + "\n")
                    insertions.append((insert_at, formatted))

        if not insertions:
            return source, 0

        # Apply insertions in reverse order to preserve line numbers
        insertions.sort(key=lambda x: x[0], reverse=True)
        for line_idx, text in insertions:
            lines.insert(line_idx, text)

        return "".join(lines), len(insertions)

    def _generate_readme_section(self, filepath: str, source: str) -> str:
        """Produce a short markdown README section for a module."""
        module_name = os.path.basename(filepath).replace(".py", "")
        try:
            tree  = ast.parse(source)
            funcs = [n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes = [n.name for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef)]
        except SyntaxError:
            funcs, classes = [], []

        lines = [
            f"## `{module_name}`",
            "",
            f"**Functions:** {', '.join(funcs[:8]) if funcs else 'none'}",
            f"**Classes:** {', '.join(classes[:8]) if classes else 'none'}",
            "",
            "_Auto-generated by Documentor agent. Fill in purpose and usage._",
            "",
        ]
        return "\n".join(lines)

    def run(self, task: str, files: Optional[List[str]] = None, **kwargs
            ) -> HandoffMessage:
        pending = self.receive()
        if not files:
            files = []
            for msg in pending:
                files.extend(msg.artifacts.get("files", []))

        if not files:
            files = self.memory.get_flagged_files()

        total_added = 0
        readme_sections: List[str] = []

        for filepath in files:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    original = f.read()
            except OSError:
                continue

            documented, n_added = self._document_source(original)
            total_added += n_added

            # Write back (only if running with write access — skip in read-only mode)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(documented)
            except OSError:
                pass   # read-only environment; artifacts still forwarded

            readme_sections.append(self._generate_readme_section(filepath, documented))

        readme_content = "\n".join(readme_sections)
        self.memory.set("readme_sections", readme_content)

        return self.emit(
            target="Coordinator",
            message=(f"Documentation pass complete. "
                     f"{total_added} docstrings added across {len(files)} files."),
            artifacts={
                "n_docstrings_added": total_added,
                "files_processed":    len(files),
                "readme_sections":    readme_content,
            },
            confidence=0.9,
            status="done",
        )


# ===========================================================================
# 12. Nexus  —  Dependency & API Specialist
# ===========================================================================

class Nexus(BaseAgent):
    """
    Manages the 'external world' of a codebase: dependencies, package versions,
    and API documentation.  Sits between Explorer and Engineer to prevent
    hallucinated library calls.

    Core capabilities:
      - Parse requirements.txt / pyproject.toml / package.json and resolve
        installed vs declared versions.
      - Detect import statements in source and cross-reference them against
        known-installed packages (via importlib.metadata).
      - Fetch abbreviated PyPI JSON metadata to verify a package exists and
        retrieve its latest stable version (no heavy HTTP library needed).
      - Build a local RAG cache: package → {version, summary, top_symbols}.
        ProblemSolver / Engineer query this before writing any library call.
      - Flag deprecated or non-existent symbols by comparing against the
        cached API surface.
    """

    role = "Nexus"

    # Packages that are stdlib and should never be flagged as missing.
    _STDLIB = frozenset(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else frozenset()

    # Simple symbol blocklist: (package, bad_symbol, replacement)
    _DEPRECATED: List[Tuple[str, str, str]] = [
        ("pandas",  "DataFrame.append",   "pd.concat([df, new_row], ignore_index=True)"),
        ("sklearn", "cross_val_predict",  "cross_val_score (check docs for correct usage)"),
        ("flask",   "before_first_request","teardown_appcontext or app-factory pattern"),
        ("django",  "url(",               "path( / re_path("),
        ("numpy",   "np.bool",            "np.bool_"),
        ("numpy",   "np.int",             "np.int_"),
        ("numpy",   "np.float",           "np.float_"),
        ("numpy",   "np.complex",         "np.complex_"),
        ("requests","requests.get(verify=False", "provide CA bundle; never disable SSL"),
    ]

    # ── Requirements parsing ─────────────────────────────────────────────────

    def _parse_requirements(self, root: str) -> Dict[str, str]:
        """
        Return {package_name: version_spec} from requirements.txt or
        pyproject.toml (dependencies table) found under root.
        """
        declared: Dict[str, str] = {}

        req_path = os.path.join(root, "requirements.txt")
        if os.path.isfile(req_path):
            try:
                with open(req_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # e.g. "pandas>=1.3,<2.0" or "flask==2.3.2"
                        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([^;#]*)", line)
                        if m:
                            declared[m.group(1).lower()] = m.group(2).strip()
            except OSError:
                pass

        pyproject_path = os.path.join(root, "pyproject.toml")
        if os.path.isfile(pyproject_path):
            try:
                with open(pyproject_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Very lightweight TOML parse — extract [project] dependencies
                in_deps = False
                for line in content.splitlines():
                    if re.match(r"\[project\.dependencies\]|\[tool\.poetry\.dependencies\]", line):
                        in_deps = True
                        continue
                    if in_deps:
                        if line.startswith("["):
                            in_deps = False
                            continue
                        m = re.match(r'^"?([A-Za-z0-9_\-\.]+)"?\s*[=:]\s*"?([^"#\n]*)"?', line)
                        if m:
                            declared[m.group(1).lower()] = m.group(2).strip()
            except OSError:
                pass

        return declared

    def _parse_package_json(self, root: str) -> Dict[str, str]:
        """Return {package: version} from package.json if present."""
        pkg_path = os.path.join(root, "package.json")
        if not os.path.isfile(pkg_path):
            return {}
        try:
            with open(pkg_path, encoding="utf-8") as f:
                data = json.load(f)
            deps: Dict[str, str] = {}
            deps.update(data.get("dependencies", {}))
            deps.update(data.get("devDependencies", {}))
            return deps
        except (OSError, json.JSONDecodeError):
            return {}

    # ── Installed package resolution ─────────────────────────────────────────

    def _installed_packages(self) -> Dict[str, str]:
        """Return {name: version} for all packages visible to importlib.metadata."""
        installed: Dict[str, str] = {}
        try:
            for dist in importlib.metadata.distributions():
                name = dist.metadata.get("Name", "").lower()
                version = dist.metadata.get("Version", "unknown")
                if name:
                    installed[name] = version
        except Exception:
            pass
        return installed

    # ── PyPI metadata fetch ──────────────────────────────────────────────────

    def _pypi_info(self, package: str, timeout: int = 4) -> Optional[Dict]:
        """
        Fetch minimal PyPI JSON for *package*.  Returns dict with keys:
          {name, latest_version, summary, requires_python, home_page}
        or None if the package doesn't exist / network unavailable.
        """
        url = f"https://pypi.org/pypi/{package}/json"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            info = data.get("info", {})
            return {
                "name":             info.get("name", package),
                "latest_version":   info.get("version", "?"),
                "summary":          info.get("summary", "")[:200],
                "requires_python":  info.get("requires_python", "any"),
                "home_page":        info.get("home_page", ""),
            }
        except Exception:
            return None

    # ── Import scanner ───────────────────────────────────────────────────────

    def _extract_imports(self, source: str) -> List[str]:
        """Return list of top-level imported package names from Python source."""
        packages: List[str] = []
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        packages.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        packages.append(node.module.split(".")[0])
        except SyntaxError:
            # Fallback regex for files with syntax errors
            for m in re.finditer(r"^(?:import|from)\s+([A-Za-z0-9_]+)", source, re.MULTILINE):
                packages.append(m.group(1))
        return list(set(packages))

    # ── Deprecation scanner ──────────────────────────────────────────────────

    def _scan_deprecated_symbols(self, source: str) -> List[Dict]:
        """Check source for known deprecated / removed API calls."""
        issues: List[Dict] = []
        for pkg, bad_symbol, replacement in self._DEPRECATED:
            if bad_symbol in source:
                issues.append({
                    "package":     pkg,
                    "bad_symbol":  bad_symbol,
                    "replacement": replacement,
                    "severity":    "warning",
                })
        return issues

    # ── RAG cache builder ────────────────────────────────────────────────────

    def _build_rag_entry(self, package: str, installed: Dict[str, str]) -> Dict:
        """
        Build a RAG cache entry for a package.
        Prefers locally installed metadata, falls back to PyPI.
        """
        entry: Dict[str, Any] = {"package": package, "source": "unknown"}
        if package in installed:
            entry["installed_version"] = installed[package]
            entry["source"] = "local"
            try:
                dist = importlib.metadata.distribution(package)
                entry["summary"] = dist.metadata.get("Summary", "")[:200]
                entry["home_page"] = dist.metadata.get("Home-page", "")
                # Extract top-level public symbols if package is importable
                try:
                    mod = __import__(package)
                    entry["top_symbols"] = [
                        s for s in dir(mod) if not s.startswith("_")
                    ][:40]
                except Exception:
                    entry["top_symbols"] = []
            except Exception:
                pass
        else:
            # Package not installed locally; try PyPI
            pypi = self._pypi_info(package)
            if pypi:
                entry.update(pypi)
                entry["source"] = "pypi"
                entry["installed_version"] = None
            else:
                entry["source"] = "not_found"
        return entry

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        """
        Full Nexus pipeline:
          1. Parse declared dependencies (requirements.txt / pyproject.toml / package.json).
          2. Resolve installed packages.
          3. Compare declared vs installed; flag mismatches.
          4. Scan all Python source imports; build RAG cache entries.
          5. Scan for deprecated symbol usage.
          6. Emit findings + RAG cache to shared memory.
        """
        declared_py  = self._parse_requirements(root)
        declared_js  = self._parse_package_json(root)
        installed    = self._installed_packages()

        # ── Declared vs installed mismatch ──────────────────────────────────
        missing_from_env: List[str] = []
        version_flags: List[Dict]   = []
        for pkg, spec in declared_py.items():
            if pkg in self._STDLIB:
                continue
            if pkg not in installed:
                missing_from_env.append(pkg)
            else:
                # Rough check: if spec starts with ==, compare exact version
                m = re.match(r"==\s*([^\s,]+)", spec)
                if m:
                    declared_ver = m.group(1).strip()
                    inst_ver = installed[pkg]
                    if declared_ver != inst_ver:
                        version_flags.append({
                            "package": pkg,
                            "declared": declared_ver,
                            "installed": inst_ver,
                        })

        # ── Walk source files and gather imports ─────────────────────────────
        repo_map  = self.memory.get_repo_map()
        all_imports: Dict[str, List[str]] = {}   # file → [packages]
        deprecated_issues: List[Dict]     = []

        py_files = list(repo_map.keys()) if repo_map else []
        if not py_files:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.endswith(".py"):
                        py_files.append(os.path.join(dirpath, fn))

        for filepath in py_files:
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except OSError:
                continue
            imports = self._extract_imports(source)
            all_imports[filepath] = imports
            depr = self._scan_deprecated_symbols(source)
            for d in depr:
                d["file"] = filepath
            deprecated_issues.extend(depr)

        # ── Build RAG cache for all unique imported packages ─────────────────
        all_pkg_names: set = set()
        for pkgs in all_imports.values():
            all_pkg_names.update(pkgs)
        # Remove stdlib
        all_pkg_names = {p for p in all_pkg_names if p not in self._STDLIB}

        rag_cache: Dict[str, Dict] = {}
        for pkg in sorted(all_pkg_names):
            rag_cache[pkg] = self._build_rag_entry(pkg, installed)

        # Store in shared memory so ProblemSolver / Engineer can query it
        self.memory.set("nexus_rag_cache", rag_cache)
        self.memory.set("nexus_declared_py", declared_py)
        self.memory.set("nexus_deprecated_issues", deprecated_issues)
        self.memory.set("nexus_missing_from_env", missing_from_env)

        # Hallucinated (not found on PyPI or locally): not_found entries
        hallucinated = [p for p, info in rag_cache.items()
                        if info.get("source") == "not_found"]

        summary_parts = [
            f"Dependency audit complete.",
            f"Declared (py): {len(declared_py)} | Declared (js): {len(declared_js)}",
            f"Missing from env: {missing_from_env or 'none'}",
            f"Version mismatches: {len(version_flags)}",
            f"Deprecated symbol uses: {len(deprecated_issues)}",
            f"Potentially hallucinated packages: {hallucinated or 'none'}",
            f"RAG cache entries: {len(rag_cache)}",
        ]

        # Forward critical findings to Engineer and Coordinator
        if hallucinated or deprecated_issues:
            self.emit(
                target="Engineer",
                message=(f"Nexus found {len(hallucinated)} potentially hallucinated packages "
                         f"and {len(deprecated_issues)} deprecated API usages. Fix before shipping."),
                artifacts={
                    "hallucinated_packages":  hallucinated,
                    "deprecated_issues":      deprecated_issues,
                    "version_mismatches":     version_flags,
                },
            )

        confidence = max(0.2, 1.0 - 0.05 * len(hallucinated) - 0.02 * len(deprecated_issues))
        return self.emit(
            target="Coordinator",
            message="\n".join(summary_parts),
            artifacts={
                "rag_cache":              rag_cache,
                "missing_from_env":       missing_from_env,
                "hallucinated_packages":  hallucinated,
                "deprecated_issues":      deprecated_issues,
                "version_mismatches":     version_flags,
                "declared_js":            declared_js,
            },
            confidence=confidence,
            status="done",
        )

    def query_api(self, package: str, symbol: str = "") -> Dict:
        """
        Convenience method for other agents to query the RAG cache at runtime.
        Returns the cache entry for *package*, with an optional symbol check.

        Usage:
            nexus.query_api("pandas", "DataFrame.merge")
        """
        cache: Dict[str, Dict] = self.memory.get("nexus_rag_cache") or {}
        entry = cache.get(package.lower(), {})
        result = dict(entry)
        if symbol and "top_symbols" in entry:
            sym_base = symbol.split(".")[0]
            result["symbol_found"] = sym_base in entry["top_symbols"]
        elif symbol:
            result["symbol_found"] = None   # unknown — needs deeper inspection
        return result


# ===========================================================================
# 13. Archeologist  —  Git History & Intent Analyst
# ===========================================================================

class Archeologist(BaseAgent):
    """
    Transforms raw Git history into a *Temporal Latent Map*: a structured
    understanding of WHY each piece of code exists, not just what it does.

    Before Engineer refactors anything, Archeologist flags:
      - Code added to fix a specific production bug (touch-carefully zone).
      - Hot-fix files (committed frequently in recent months → volatile).
      - Stale legacy code untouched for > STALE_DAYS (candidate for removal).
      - Large commits that swept many files (risky blast-radius refactors).
      - Commit messages that reference known patterns: "fix race", "hotfix",
        "revert", "security", "CVE" → extra caution flags.

    Builds its map using pure subprocess calls to git; no external library
    required.  Degrades gracefully if the repo has no git history.
    """

    role = "Archeologist"
    STALE_DAYS       = 730    # 2 years
    HOT_FIX_COMMITS  = 5      # if a file has this many commits in 90 days → volatile
    LARGE_COMMIT_FILES = 20   # commits touching > this many files are flagged

    # Commit message patterns that warrant extra caution
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

    # ── Git helpers ──────────────────────────────────────────────────────────

    def _git(self, args: List[str], cwd: str, timeout: int = 15) -> str:
        """Run a git command; return stdout as string, empty on error."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    def _has_git(self, root: str) -> bool:
        out = self._git(["rev-parse", "--git-dir"], root)
        return bool(out)

    def _log_for_file(self, filepath: str, root: str) -> List[Dict]:
        """
        Return list of commit records for a single file:
          {hash, author, date_iso, subject, files_changed}
        Uses --follow to track renames.
        """
        sep = "|||"
        fmt = f"%H{sep}%ae{sep}%ci{sep}%s"
        out = self._git(
            ["log", "--follow", f"--pretty=format:{fmt}", "--", filepath],
            cwd=root,
        )
        if not out:
            return []

        records: List[Dict] = []
        for line in out.splitlines():
            parts = line.split(sep)
            if len(parts) < 4:
                continue
            commit_hash, author, date_iso, subject = parts[0], parts[1], parts[2], parts[3]
            # Count files changed in this commit
            stat_out = self._git(
                ["diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
                cwd=root,
            )
            files_in_commit = [l for l in stat_out.splitlines() if l.strip()]
            records.append({
                "hash":          commit_hash[:12],
                "author":        author,
                "date_iso":      date_iso.strip(),
                "subject":       subject.strip(),
                "files_changed": len(files_in_commit),
            })
        return records

    def _parse_date(self, date_iso: str) -> Optional[float]:
        """Parse ISO date string to POSIX timestamp; return None on failure."""
        # Format: "2024-03-15 14:22:01 +0000"
        try:
            import email.utils
            # Strip timezone offset and parse naive
            date_part = date_iso.rsplit(" ", 1)[0]  # remove tz
            import datetime
            dt = datetime.datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except Exception:
            return None

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _classify_file(self, filepath: str, commits: List[Dict]) -> Dict:
        """
        Produce a temporal classification for a single file given its commits.
        Returns a dict with keys:
          age_days, last_touched_days_ago, commit_count, volatile, stale,
          caution_flags, caution_reasons, large_commit_hashes,
          hotfix_subjects, recommendation
        """
        now = time.time()
        if not commits:
            return {
                "age_days": None, "last_touched_days_ago": None,
                "commit_count": 0, "volatile": False, "stale": True,
                "caution_flags": [], "caution_reasons": [],
                "large_commit_hashes": [], "hotfix_subjects": [],
                "recommendation": "no-git-history",
            }

        # Oldest commit = creation date
        timestamps = [self._parse_date(c["date_iso"]) for c in commits]
        timestamps = [t for t in timestamps if t is not None]

        if timestamps:
            first_ts = min(timestamps)
            last_ts  = max(timestamps)
            age_days            = (now - first_ts) / 86400
            last_touched_days_ago = (now - last_ts) / 86400
        else:
            age_days = last_touched_days_ago = None

        # Hot-fix volatility: commits in last 90 days
        ninety_days_ago = now - 90 * 86400
        recent_commits = [
            c for c, t in zip(commits, timestamps)
            if t and t > ninety_days_ago
        ] if timestamps else []
        volatile = len(recent_commits) >= self.HOT_FIX_COMMITS

        stale = (last_touched_days_ago is not None
                 and last_touched_days_ago > self.STALE_DAYS)

        # Caution flags from commit messages
        caution_flags: List[str]   = []
        caution_reasons: List[str] = []
        hotfix_subjects: List[str] = []
        for commit in commits:
            subj = commit["subject"].lower()
            for pattern, label in self._CAUTION_PATTERNS:
                if re.search(pattern, subj, re.IGNORECASE):
                    if label not in caution_flags:
                        caution_flags.append(label)
                    caution_reasons.append(
                        f"[{commit['hash']}] {label}: {commit['subject'][:80]}"
                    )
                    if label == "hotfix":
                        hotfix_subjects.append(commit["subject"][:100])

        # Large commit sweep detection
        large_commit_hashes = [
            c["hash"] for c in commits
            if c["files_changed"] >= self.LARGE_COMMIT_FILES
        ]

        # Build recommendation
        rec_parts: List[str] = []
        if volatile:
            rec_parts.append("VOLATILE: high recent change rate — test thoroughly after any edit")
        if stale:
            rec_parts.append("STALE: untouched for 2+ years — check if still needed")
        if caution_flags:
            rec_parts.append(f"CAUTION-FLAGS: {', '.join(caution_flags)}")
        if large_commit_hashes:
            rec_parts.append(f"BLAST-RADIUS: was part of large sweeping commits {large_commit_hashes[:3]}")
        recommendation = " | ".join(rec_parts) if rec_parts else "safe-to-refactor"

        return {
            "age_days":             round(age_days, 1) if age_days else None,
            "last_touched_days_ago": round(last_touched_days_ago, 1) if last_touched_days_ago else None,
            "commit_count":         len(commits),
            "volatile":             volatile,
            "stale":                stale,
            "caution_flags":        caution_flags,
            "caution_reasons":      caution_reasons[:5],    # top 5
            "large_commit_hashes":  large_commit_hashes[:3],
            "hotfix_subjects":      hotfix_subjects[:3],
            "recommendation":       recommendation,
        }

    def _build_temporal_map(self, root: str, py_files: List[str]) -> Dict[str, Dict]:
        """
        For every file, fetch its git log and produce a classification.
        Returns {filepath: classification_dict}.
        """
        temporal_map: Dict[str, Dict] = {}
        for filepath in py_files:
            rel = os.path.relpath(filepath, root)
            commits = self._log_for_file(rel, root)
            classification = self._classify_file(filepath, commits)
            classification["commit_log"] = [
                {"hash": c["hash"], "date": c["date_iso"][:10],
                 "subject": c["subject"][:80]}
                for c in commits[:10]
            ]
            temporal_map[filepath] = classification
        return temporal_map

    def _global_stats(self, root: str) -> Dict:
        """
        Repo-level statistics: total commits, active contributors,
        most recently changed files, most frequently changed files.
        """
        # Total commits
        total_str = self._git(["rev-list", "--count", "HEAD"], root)
        total_commits = int(total_str) if total_str.isdigit() else 0

        # Active contributors (last 90 days)
        contrib_out = self._git(
            ["shortlog", "-sn", "--since=90.days.ago", "HEAD"],
            root,
        )
        contributors = []
        for line in contrib_out.splitlines():
            m = re.match(r"\s*(\d+)\s+(.+)", line)
            if m:
                contributors.append({"commits": int(m.group(1)), "name": m.group(2)})

        # Most frequently changed files (all time top 10)
        freq_out = self._git(
            ["log", "--name-only", "--pretty=format:", "HEAD"],
            root,
        )
        freq_counter: Dict[str, int] = {}
        for line in freq_out.splitlines():
            line = line.strip()
            if line and not line.startswith(" "):
                freq_counter[line] = freq_counter.get(line, 0) + 1
        most_changed = sorted(freq_counter.items(), key=lambda x: -x[1])[:10]

        return {
            "total_commits":     total_commits,
            "active_contributors": contributors[:10],
            "most_changed_files": [
                {"file": f, "changes": c} for f, c in most_changed
            ],
        }

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self, task: str, root: str = ".", **kwargs) -> HandoffMessage:
        """
        Full Archeologist pipeline:
          1. Check for git presence.
          2. Build temporal map (per-file classification).
          3. Compute global repo stats.
          4. Identify touch-carefully zones and stale zones.
          5. Emit temporal map to shared memory; forward caution zones to Engineer.
        """
        if not self._has_git(root):
            return self.emit(
                target="Coordinator",
                message="No git repository found. Archeologist skipped.",
                artifacts={"has_git": False},
                confidence=0.5,
                status="done",
            )

        # Collect python files from shared repo map or walk fresh
        repo_map = self.memory.get_repo_map()
        py_files = list(repo_map.keys()) if repo_map else []
        if not py_files:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.endswith(".py"):
                        py_files.append(os.path.join(dirpath, fn))

        temporal_map = self._build_temporal_map(root, py_files)
        global_stats = self._global_stats(root)

        # Identify caution zones
        volatile_files = [
            fp for fp, info in temporal_map.items() if info.get("volatile")
        ]
        caution_files = [
            fp for fp, info in temporal_map.items()
            if info.get("caution_flags")
        ]
        stale_files = [
            fp for fp, info in temporal_map.items()
            if info.get("stale") and not info.get("volatile")
        ]

        # Write temporal map into shared memory so Engineer / Planner can read
        self.memory.set("temporal_map", temporal_map)
        self.memory.set("archeologist_global_stats", global_stats)

        # Annotate repo_map with temporal data
        for fp, info in temporal_map.items():
            existing = self.memory.get_repo_map().get(fp, {})
            existing["temporal"] = {
                "volatile":      info["volatile"],
                "stale":         info["stale"],
                "caution_flags": info["caution_flags"],
                "recommendation": info["recommendation"],
                "last_touched_days_ago": info["last_touched_days_ago"],
            }
            self.memory.update_repo_map(fp, existing)

        # Forward volatile + caution files to Engineer with context
        if volatile_files or caution_files:
            caution_context = {}
            for fp in set(volatile_files + caution_files):
                info = temporal_map[fp]
                caution_context[fp] = {
                    "recommendation": info["recommendation"],
                    "caution_reasons": info["caution_reasons"],
                    "hotfix_subjects": info["hotfix_subjects"],
                }
            self.emit(
                target="Engineer",
                message=(
                    f"Archeologist flagged {len(volatile_files)} volatile files and "
                    f"{len(caution_files)} caution-zone files. "
                    "Do NOT simplify hotfix-tagged code without understanding the original bug."
                ),
                artifacts={"caution_context": caution_context},
            )

        # Forward stale files to Engineer and Documentor as cleanup candidates
        if stale_files:
            self.emit(
                target="Documentor",
                message=(
                    f"{len(stale_files)} files untouched for 2+ years. "
                    "Mark with deprecation notices where appropriate."
                ),
                artifacts={"stale_files": stale_files[:20]},
            )

        summary = (
            f"Temporal map built for {len(py_files)} files. "
            f"Volatile: {len(volatile_files)} | Caution: {len(caution_files)} | Stale: {len(stale_files)}. "
            f"Repo has {global_stats['total_commits']} total commits, "
            f"{len(global_stats['active_contributors'])} active contributors (last 90d)."
        )

        confidence = max(0.3, 1.0 - 0.04 * len(volatile_files) - 0.02 * len(caution_files))
        return self.emit(
            target="Coordinator",
            message=summary,
            artifacts={
                "temporal_map":    {fp: v for fp, v in list(temporal_map.items())[:20]},
                "global_stats":    global_stats,
                "volatile_files":  volatile_files,
                "caution_files":   caution_files,
                "stale_files":     stale_files[:20],
            },
            confidence=confidence,
            status="done",
        )

    def explain_file(self, filepath: str) -> str:
        """
        Convenience method: return a human-readable intent summary for a single
        file, drawn from the temporal map already stored in shared memory.

        Usage (e.g. from Engineer before refactoring):
            context = swarm.archeologist.explain_file("auth/views.py")
        """
        temporal_map: Dict[str, Dict] = self.memory.get("temporal_map") or {}
        info = temporal_map.get(filepath)
        if not info:
            return f"No temporal data for {filepath}. Run Archeologist first."

        lines = [f"=== Archeologist report: {filepath} ==="]
        lines.append(f"  Age: {info.get('age_days', '?')} days since first commit")
        lines.append(f"  Last touched: {info.get('last_touched_days_ago', '?')} days ago")
        lines.append(f"  Commits: {info.get('commit_count', 0)}")
        lines.append(f"  Volatile: {info.get('volatile', False)} | Stale: {info.get('stale', False)}")
        lines.append(f"  Recommendation: {info.get('recommendation', 'unknown')}")
        if info.get("caution_reasons"):
            lines.append("  Caution history:")
            for reason in info["caution_reasons"]:
                lines.append(f"    • {reason}")
        if info.get("hotfix_subjects"):
            lines.append("  Hotfix commits (do not simplify without reading):")
            for subj in info["hotfix_subjects"]:
                lines.append(f"    ⚠ {subj}")
        return "\n".join(lines)


# ===========================================================================
# CogSearch  —  Execution-Guided Monte Carlo Tree Search
# ===========================================================================

@dataclass
class MCTSNode:
    """
    A single node in the CogSearch tree.
    Each node represents a code state: a partial or complete code snippet
    plus the number of times it has been visited and its accumulated reward.
    """
    code:          str
    parent:        Optional["MCTSNode"] = field(default=None, repr=False)
    children:      List["MCTSNode"]    = field(default_factory=list, repr=False)
    visits:        int   = 0
    total_reward:  float = 0.0
    depth:         int   = 0
    expansion_prompt: str = ""   # the prompt fragment that generated this code
    terminal:      bool  = False  # True if this node produced passing code

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    def uct(self, exploration_c: float = 1.414) -> float:
        """
        Upper Confidence bound applied to Trees (UCT).
        UCT = W_i/N_i  +  c * sqrt(ln(N_parent) / N_i)

        If this node has never been visited, return +inf (always explore first).
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else 1
        exploit = self.mean_reward
        explore  = exploration_c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return exploit + explore

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class CogSearch:
    """
    Execution-Guided Monte Carlo Tree Search engine for CogForge.

    Wraps the ProblemSolver (expansion), VerifierHead (internal scoring),
    TerminalGuy (external execution), and Pessimist (logic-flaw detection)
    into a principled MCTS loop.

    Algorithm (per iteration):
      1. Selection   — descend tree via UCT until a leaf node is reached.
      2. Expansion   — ProblemSolver generates K candidate next-step fragments.
      3. Evaluation  — Level-1: VerifierHead score.  Level-2: py_compile / pytest.
      4. Reward      — syntax error → -1.0 | logic flaw → 0.1 | passes tests → 1.0
      5. Backprop    — propagate reward up through all ancestor nodes.

    After all iterations, returns the highest-reward leaf (best code path) plus
    a list of DPO pairs: (prompt, winning_code, losing_code) for training.
    """

    # Reward constants
    REWARD_SYNTAX_ERROR   = -1.0
    REWARD_LOGIC_FLAW     =  0.1
    REWARD_COMPILES_CLEAN =  0.5
    REWARD_PASSES_TESTS   =  1.0

    def __init__(
        self,
        problem_solver: "ProblemSolver",
        terminal_guy:   "TerminalGuy",
        pessimist:      "Pessimist",
        model:          Optional["CogForge"] = None,
        exploration_c:  float = 1.414,
        k_expansions:   int   = 3,
        verifier_threshold: float = 0.3,
    ):
        self.problem_solver      = problem_solver
        self.terminal_guy        = terminal_guy
        self.pessimist           = pessimist
        self.model               = model
        self.exploration_c       = exploration_c
        self.k_expansions        = k_expansions
        self.verifier_threshold  = verifier_threshold
        self._dpo_pairs: List[Dict] = []   # accumulated DPO training pairs

    # ── Tree operations ──────────────────────────────────────────────────────

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Descend from root, always choosing the child with the highest UCT score,
        until a leaf (unexpanded) node is reached.
        """
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.uct(self.exploration_c))
        return node

    def _expand(self, node: MCTSNode, prompt: str) -> List[MCTSNode]:
        """
        Ask ProblemSolver to generate K candidate code fragments that extend
        the current node's code.  Returns a list of new child MCTSNodes.

        The heuristic generator produces K stylistically different completions
        (loop-based, comprehension-based, recursive) as code branch candidates.
        When a real generative model is attached, the ProblemSolver would call
        model.generate() here with nucleus sampling at varying temperatures.
        """
        base_code = node.code
        children: List[MCTSNode] = []

        # Heuristic expansion strategies (no generative model required)
        strategies = [
            ("iterative",    self._strategy_iterative),
            ("functional",   self._strategy_functional),
            ("recursive",    self._strategy_recursive),
        ]

        for i in range(min(self.k_expansions, len(strategies))):
            strategy_name, strategy_fn = strategies[i % len(strategies)]
            candidate_code = strategy_fn(base_code, prompt)
            child = MCTSNode(
                code=candidate_code,
                parent=node,
                depth=node.depth + 1,
                expansion_prompt=f"{strategy_name}: {prompt[:60]}",
            )
            children.append(child)

        node.children.extend(children)
        return children

    def _strategy_iterative(self, base: str, prompt: str) -> str:
        """Generate an iterative (for-loop) style code fragment."""
        func_name = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + "\n" +
                f"def {func_name}_iterative(items):\n"
                f"    result = []\n"
                f"    for item in items:\n"
                f"        # TODO: implement {prompt[:40]}\n"
                f"        result.append(item)\n"
                f"    return result\n")

    def _strategy_functional(self, base: str, prompt: str) -> str:
        """Generate a functional (list-comprehension) style code fragment."""
        func_name = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + "\n" +
                f"def {func_name}_functional(items):\n"
                f"    # Functional approach: {prompt[:60]}\n"
                f"    return [item for item in items if item is not None]\n")

    def _strategy_recursive(self, base: str, prompt: str) -> str:
        """Generate a recursive style code fragment."""
        func_name = re.sub(r"[^a-z0-9_]", "_", prompt.lower().split()[0][:20])
        return (base + "\n" +
                f"def {func_name}_recursive(items, acc=None):\n"
                f"    if acc is None:\n"
                f"        acc = []\n"
                f"    if not items:\n"
                f"        return acc\n"
                f"    return {func_name}_recursive(items[1:], acc + [items[0]])\n")

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _level1_verifier(self, code: str) -> float:
        """
        Internal VerifierHead evaluation.
        When a CogForge model is present, tokenise the code, run a forward
        pass, and read the verifier score.  Otherwise, fall back to a fast
        heuristic that checks for basic Python validity and style signals.
        """
        if self.model is not None:
            try:
                # Minimal tokenisation: use character IDs clamped to vocab_size
                cfg = self.model.config
                char_ids = [ord(c) % cfg.vocab_size for c in code[:cfg.max_seq_len]]
                if not char_ids:
                    return 0.0
                ids = torch.tensor([char_ids], dtype=torch.long)
                with torch.no_grad():
                    out = self.model(ids, return_verifier=True)
                score = out.get("verifier_score")
                if score is not None:
                    return float(score.mean().item())
            except Exception:
                pass

        # Heuristic fallback
        score = 0.5
        # Positive signals: has a def, has a return, has docstring, no pass
        if re.search(r"\bdef\s+\w+", code):
            score += 0.1
        if "return" in code:
            score += 0.1
        if '"""' in code or "'''" in code:
            score += 0.05
        if re.search(r"\bpass\s*$", code, re.MULTILINE):
            score -= 0.15   # bare pass = not implemented
        if re.search(r"TODO|FIXME|HACK", code):
            score -= 0.1
        # Negative: obvious truncation or missing colon
        if code.count("def ") > code.count(":"):
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _level2_execute(self, code: str, run_tests: bool = False) -> Dict:
        """
        External execution via TerminalGuy.
        Stage A: python3 -m py_compile (syntax check).
        Stage B (if run_tests): python3 -m pytest (unit tests, if any exist).

        Returns dict: {compiles: bool, tests_pass: bool | None, stderr: str}
        """
        # Write code to a temp file
        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"cogsearch_{uuid.uuid4().hex[:8]}.py"
        )
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Stage A: syntax compile check
            compile_result = self.terminal_guy._run_command(
                f"python3 -m py_compile {tmp_path}", timeout=10
            )
            compiles = (compile_result["returncode"] == 0)

            tests_pass: Optional[bool] = None
            test_stderr = ""

            if compiles and run_tests:
                # Stage B: pytest (only if a test suite exists alongside)
                test_result = self.terminal_guy._run_command(
                    f"python3 -m pytest {tmp_path} --tb=short -q", timeout=20
                )
                tests_pass = (test_result["returncode"] == 0)
                test_stderr = test_result.get("stderr", "")[:500]

            return {
                "compiles":    compiles,
                "tests_pass":  tests_pass,
                "stderr":      (compile_result.get("stderr", "") + test_stderr)[:600],
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _compute_reward(self, code: str, verifier_score: float,
                        exec_result: Dict, run_tests: bool) -> float:
        """
        Map evaluation results to a scalar reward in [-1, 1].

          - Fails to compile            → REWARD_SYNTAX_ERROR  (-1.0)
          - Compiles, verifier low      → REWARD_LOGIC_FLAW    (0.1)
          - Compiles, verifier ok       → REWARD_COMPILES_CLEAN (0.5)
          - Compiles + passes tests     → REWARD_PASSES_TESTS   (1.0)

        Pessimist high-risk issues apply a 0.2 penalty on top.
        """
        if not exec_result["compiles"]:
            return self.REWARD_SYNTAX_ERROR

        if run_tests and exec_result.get("tests_pass") is True:
            reward = self.REWARD_PASSES_TESTS
        elif verifier_score >= self.verifier_threshold:
            reward = self.REWARD_COMPILES_CLEAN
        else:
            reward = self.REWARD_LOGIC_FLAW

        # Pessimist penalty for logic flaws
        critiques = self.pessimist._critique_code(code)
        high_risk  = [c for c in critiques if c["risk"] == "high"]
        if high_risk:
            reward -= 0.2 * len(high_risk)

        return max(self.REWARD_SYNTAX_ERROR, min(self.REWARD_PASSES_TESTS, reward))

    # ── Backpropagation ──────────────────────────────────────────────────────

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Walk from *node* to the root, incrementing visits and accumulating
        total_reward at every ancestor.
        """
        current = node
        while current is not None:
            current.visits      += 1
            current.total_reward += reward
            current = current.parent

    # ── Best path extraction ─────────────────────────────────────────────────

    def _best_leaf(self, root: MCTSNode) -> MCTSNode:
        """
        BFS over all nodes; return the visited leaf with the highest mean reward.
        """
        best: MCTSNode = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.visits > 0 and node.mean_reward > best.mean_reward:
                best = node
            stack.extend(node.children)
        return best

    def _collect_all_leaves(self, root: MCTSNode) -> List[MCTSNode]:
        """Return all leaf nodes in the tree (depth-first)."""
        leaves: List[MCTSNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_leaf() and node.visits > 0:
                leaves.append(node)
            stack.extend(node.children)
        return leaves

    # ── DPO pair generation ──────────────────────────────────────────────────

    def _record_dpo_pair(self, prompt: str, winner: MCTSNode,
                         loser: MCTSNode) -> None:
        """
        Save a preference pair for offline DPO fine-tuning.
        Pairs are stored as {prompt, winning_code, losing_code, reward_delta}.
        """
        self._dpo_pairs.append({
            "prompt":        prompt,
            "winning_code":  winner.code,
            "losing_code":   loser.code,
            "reward_delta":  winner.mean_reward - loser.mean_reward,
        })

    def get_dpo_pairs(self) -> List[Dict]:
        """Return accumulated DPO training pairs from all searches so far."""
        return list(self._dpo_pairs)

    # ── Main search entry point ──────────────────────────────────────────────

    def search(
        self,
        prompt:     str,
        iterations: int  = 10,
        run_tests:  bool = False,
        initial_code: str = "",
    ) -> Dict:
        """
        Run CogSearch for *iterations* MCTS steps.

        Args:
            prompt:       Natural-language description of what to generate.
            iterations:   Number of select→expand→evaluate→backprop cycles.
            run_tests:    If True, attempt `python3 -m pytest` on each candidate.
            initial_code: Seed code to prepend to all expansions.

        Returns dict with:
            best_code:       str   — highest-reward code found.
            best_reward:     float — reward of best node.
            dpo_pairs:       list  — new DPO pairs generated during this search.
            tree_stats:      dict  — nodes explored, depth, etc.
            all_leaves:      list  — sorted (reward, code) tuples for all leaves.
        """
        root = MCTSNode(code=initial_code, depth=0)
        root.visits = 1   # mark root as already visited

        nodes_evaluated = 0

        for iteration in range(iterations):
            # ─── 1. Selection ───────────────────────────────────────────────
            selected = self._select(root)

            # If selected node is terminal (already passed tests), skip
            if selected.terminal:
                continue

            # ─── 2. Expansion ───────────────────────────────────────────────
            new_children = self._expand(selected, prompt)

            # ─── 3 & 4. Evaluate each child (Simulation + Reward) ───────────
            for child in new_children:
                nodes_evaluated += 1

                # Level 1: VerifierHead (fast, internal)
                v_score = self._level1_verifier(child.code)
                if v_score < self.verifier_threshold:
                    # Kill branch immediately — don't waste execution budget
                    reward = self.REWARD_SYNTAX_ERROR * 0.5  # softer penalty
                    self._backpropagate(child, reward)
                    continue

                # Level 2: Actual execution (external oracle)
                exec_result = self._level2_execute(child.code, run_tests)

                reward = self._compute_reward(child.code, v_score, exec_result, run_tests)

                if exec_result.get("tests_pass") is True:
                    child.terminal = True

                # ─── 5. Backpropagation ──────────────────────────────────────
                self._backpropagate(child, reward)

        # ── Collect results ──────────────────────────────────────────────────
        best_node = self._best_leaf(root)
        all_leaves = sorted(
            [(n.mean_reward, n.code) for n in self._collect_all_leaves(root)],
            key=lambda x: -x[0]
        )

        # Generate DPO pairs: pair each adjacent best/worst leaf
        leaves_by_reward = sorted(
            self._collect_all_leaves(root),
            key=lambda n: -n.mean_reward
        )
        if len(leaves_by_reward) >= 2:
            for i in range(min(3, len(leaves_by_reward) // 2)):
                winner = leaves_by_reward[i]
                loser  = leaves_by_reward[-(i + 1)]
                if winner.mean_reward > loser.mean_reward + 0.1:
                    self._record_dpo_pair(prompt, winner, loser)

        # Max depth reached in tree
        all_nodes: List[MCTSNode] = []
        stack = [root]
        while stack:
            n = stack.pop()
            all_nodes.append(n)
            stack.extend(n.children)
        max_depth = max((n.depth for n in all_nodes), default=0)

        return {
            "best_code":      best_node.code,
            "best_reward":    best_node.mean_reward,
            "best_visits":    best_node.visits,
            "dpo_pairs":      self.get_dpo_pairs(),
            "tree_stats": {
                "total_nodes":     len(all_nodes),
                "nodes_evaluated": nodes_evaluated,
                "max_depth":       max_depth,
                "iterations":      iterations,
            },
            "all_leaves": all_leaves[:10],
        }


# ===========================================================================
# CogWorks Swarm  —  orchestration entry point
# ===========================================================================

class CogWorksSwarm:
    """
    Instantiates all swarm agents around a shared memory store (and optionally
    a shared CogForge model).  Exposes a single `.run(task, repo_root)` entry
    point that mirrors the documented workflow, plus `.cog_search(prompt)` for
    Execution-Guided MCTS code generation.

    Workflow:
      1. Coordinator delegates to Planner + Explorer (parallel).
      2. Dreamer injects context.
      3. Archeologist maps git history → temporal caution zones.
      4. Nexus audits dependencies → RAG cache built.
      5. Planner outputs DAG → ProblemSolver refines.
      6. Engineer → Pessimist → BugFinder + VulnerabilityFinder.
      7. TerminalGuy executes tests.
      8. Documentor annotates.
      9. Coordinator gates on verifier score.
     10. Dreamer consolidates into long-term memory.
    """

    def __init__(self, model: Optional[CogForge] = None,
                 config: Optional[CogForgeConfig] = None):
        self.shared_memory = SharedMemoryStore()
        cfg = config or CogForgeConfig()

        self.coordinator          = Coordinator(self.shared_memory, model)
        self.dreamer              = Dreamer(self.shared_memory, model, cfg)
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

        # CogSearch MCTS engine — wires ProblemSolver, TerminalGuy, Pessimist
        self.cog_search_engine = CogSearch(
            problem_solver=self.problem_solver,
            terminal_guy=self.terminal_guy,
            pessimist=self.pessimist,
            model=model,
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
        }

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self, task: str, repo_root: str = ".") -> Dict[str, Any]:
        """
        Execute the full swarm workflow for a given task.
        Returns a summary dict of all agent outputs.
        """
        results: Dict[str, Any] = {}

        # Step 1: Coordinator kicks things off
        coord_msg = self.coordinator.run(task)
        results["coordinator_kickoff"] = coord_msg.message

        # Step 2: Dreamer provides context
        dreamer_msg = self.dreamer.run(task)
        results["dreamer_context"] = dreamer_msg.message

        # Step 3: Explorer maps the repo (builds shared repo_map)
        explore_msg = self.explorer.run(task, root=repo_root)
        results["explorer"] = explore_msg.artifacts

        # Step 4: Archeologist — git history & intent analysis
        arch_msg = self.archeologist.run(task, root=repo_root)
        results["archeologist"] = arch_msg.artifacts

        # Step 5: Nexus — dependency audit & RAG cache
        nexus_msg = self.nexus.run(task, root=repo_root)
        results["nexus"] = nexus_msg.artifacts

        # Step 6: Planner decomposes the task
        plan_msg = self.planner.run(task)
        results["planner"] = plan_msg.artifacts

        # Step 7: ProblemSolver analyses and annotates
        ps_msg = self.problem_solver.run(task)
        results["problem_solver"] = ps_msg.artifacts

        # Step 8: Engineer reviews code quality
        eng_msg = self.engineer.run(task)
        results["engineer"] = eng_msg.artifacts

        # Step 9: Pessimist stress-tests
        pess_msg = self.pessimist.run(task)
        results["pessimist"] = pess_msg.artifacts

        # Step 10: BugFinder deep inspection
        bug_msg = self.bug_finder.run(task)
        results["bug_finder"] = bug_msg.artifacts

        # Step 11: VulnerabilityFinder security scan
        vuln_msg = self.vulnerability_finder.run(task)
        results["vulnerability_finder"] = vuln_msg.artifacts

        # Step 12: TerminalGuy runs tests
        term_msg = self.terminal_guy.run(task, command="python -m pytest --tb=short -q")
        results["terminal_guy"] = term_msg.artifacts

        # Step 13: Documentor annotates
        flagged = self.shared_memory.get_flagged_files()
        doc_msg = self.documentor.run(task, files=flagged)
        results["documentor"] = doc_msg.artifacts

        # Step 14: Dreamer consolidates
        self.dreamer.maybe_consolidate_episodic()
        results["dreamer_consolidation"] = "complete"

        # Step 15: Coordinator quality gate
        verifier_proxy = pess_msg.confidence
        gate_decision = self.coordinator.review_and_gate(verifier_proxy, plan_msg.task_id)
        results["quality_gate"]          = gate_decision
        results["verifier_proxy_score"]  = verifier_proxy

        return results

    # ── CogSearch entry point ─────────────────────────────────────────────────

    def cog_search(
        self,
        prompt:       str,
        iterations:   int  = 10,
        run_tests:    bool = False,
        initial_code: str  = "",
        k_expansions: int  = 3,
    ) -> Dict[str, Any]:
        """
        Run Execution-Guided MCTS to find the best code for *prompt*.

        Each iteration:
          1. SELECT   — UCT picks the most promising partial code branch.
          2. EXPAND   — ProblemSolver generates k_expansions candidate fragments.
          3. EVALUATE — VerifierHead scores each; TerminalGuy compiles/runs them.
          4. REWARD   — syntax error → -1.0 | logic flaw → 0.1 | passes → 1.0
          5. BACKPROP — reward flows up to root through all ancestors.

        Returns dict with:
          best_code:   The highest-reward code found.
          best_reward: Reward score of best_code.
          dpo_pairs:   List of (prompt, winning_code, losing_code) training pairs.
          tree_stats:  Search statistics (nodes, depth, iterations).
          all_leaves:  Top-10 leaves sorted by reward.
        """
        self.cog_search_engine.k_expansions = k_expansions
        result = self.cog_search_engine.search(
            prompt=prompt,
            iterations=iterations,
            run_tests=run_tests,
            initial_code=initial_code,
        )

        # Store best code and DPO pairs in shared memory
        self.shared_memory.set("cog_search_best_code", result["best_code"])
        self.shared_memory.set("cog_search_dpo_pairs", result["dpo_pairs"])

        # Log result as a handoff from ProblemSolver → Coordinator
        self.coordinator.emit(
            target="Coordinator",
            message=(
                f"CogSearch complete. Best reward: {result['best_reward']:.3f}. "
                f"Explored {result['tree_stats']['total_nodes']} nodes in "
                f"{result['tree_stats']['iterations']} iterations. "
                f"Generated {len(result['dpo_pairs'])} DPO training pairs."
            ),
            artifacts={
                "best_reward": result["best_reward"],
                "tree_stats":  result["tree_stats"],
                "n_dpo_pairs": len(result["dpo_pairs"]),
            },
            status="done",
        )

        return result


# ===========================================================================
# Quick smoke test
# ===========================================================================

if __name__ == "__main__":
    cfg   = CogForgeConfig()
    model = CogForge(cfg)
    print(f"CogForge parameters: {model.count_parameters():,}")
    print(f"Memory slots: {cfg.memory_size}")
    print(f"Memory-injected layers: {cfg.n_memory_layers} (last layers)")
    print(f"Architect cross-attn at layers: 4, 8, 10")
    print(f"Dynamic latent tokens: {cfg.n_latent_tokens}–{cfg.max_latent_tokens}")

    # Tiny forward pass sanity check
    B, T = 2, 64
    ids   = torch.randint(0, cfg.vocab_size, (B, T))
    chunks = torch.randn(B, cfg.max_repo_chunks, cfg.d_model)

    out = model(ids, repo_chunks=chunks, return_verifier=True, return_handoff=True)
    print(f"\nForward pass outputs:")
    print(f"  logits:          {out['logits'].shape}")
    print(f"  ponder_cost:     {out['ponder_cost'].shape}")
    print(f"  memory_state:    {out['memory_state'].shape}")
    print(f"  verifier_score:  {out['verifier_score']}")
    print(f"  recommended_agent: {out['recommended_agent']}")

    # ── CogWorks Swarm smoke test ────────────────────────────────────────────
    print("\n--- CogWorks Swarm (full pipeline) ---")
    swarm = CogWorksSwarm(model=model, config=cfg)
    results = swarm.run("Refactor auth module for better security and testability",
                        repo_root=".")
    print(f"Quality gate decision: {results['quality_gate']}")
    print(f"Verifier proxy score:  {results['verifier_proxy_score']:.3f}")
    print(f"Agents completed:      {list(results.keys())}")

    # ── Archeologist smoke test ──────────────────────────────────────────────
    print("\n--- Archeologist ---")
    arch_artifacts = results.get("archeologist", {})
    global_stats = arch_artifacts.get("global_stats", {})
    print(f"Total commits in repo:         {global_stats.get('total_commits', 'N/A')}")
    print(f"Volatile files flagged:        {len(arch_artifacts.get('volatile_files', []))}")
    print(f"Caution-zone files:            {len(arch_artifacts.get('caution_files', []))}")
    print(f"Stale files (2+ yrs):          {len(arch_artifacts.get('stale_files', []))}")

    # ── Nexus smoke test ─────────────────────────────────────────────────────
    print("\n--- Nexus ---")
    nexus_artifacts = results.get("nexus", {})
    rag_cache = nexus_artifacts.get("rag_cache", {})
    hallucinated = nexus_artifacts.get("hallucinated_packages", [])
    deprecated   = nexus_artifacts.get("deprecated_issues", [])
    print(f"RAG cache entries:             {len(rag_cache)}")
    print(f"Hallucinated packages:         {hallucinated or 'none'}")
    print(f"Deprecated API usages:         {len(deprecated)}")
    # Query RAG for torch
    torch_info = swarm.nexus.query_api("torch", "nn.Linear")
    print(f"Nexus RAG query (torch):       source={torch_info.get('source')} "
          f"symbol_found={torch_info.get('symbol_found')}")

    # ── CogSearch MCTS smoke test ────────────────────────────────────────────
    print("\n--- CogSearch (Execution-Guided MCTS) ---")
    search_result = swarm.cog_search(
        prompt="Write a function that filters a list of integers, keeping only primes",
        iterations=6,
        run_tests=False,
        k_expansions=3,
    )
    print(f"Best reward:    {search_result['best_reward']:.3f}")
    print(f"Tree stats:     {search_result['tree_stats']}")
    print(f"DPO pairs gen:  {len(search_result['dpo_pairs'])}")
    print(f"Top leaves:")
    for reward, code_preview in search_result["all_leaves"][:3]:
        preview = code_preview.replace("\n", " ")[:80]
        print(f"  [{reward:+.2f}] {preview}…")
    if search_result["dpo_pairs"]:
        pair = search_result["dpo_pairs"][0]
        print(f"\nSample DPO pair (reward delta = {pair['reward_delta']:+.3f}):")
        print(f"  Winner preview: {pair['winning_code'][:60].strip()}…")
        print(f"  Loser  preview: {pair['losing_code'][:60].strip()}…")
