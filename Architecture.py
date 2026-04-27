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

CogWorks Swarm Agents:
  - Coordinator    : Overseer, task routing, quality gates
  - Dreamer        : Hierarchical memory management and consolidation
  - Explorer       : Repository mapping and flagging
  - Planner        : Task decomposition into DAGs
  - ProblemSolver  : Deep expert reasoning
  - Engineer       : Code efficiency and clean-form refactoring
  - BugFinder      : Deep inspection and logical error hunting
  - TerminalGuy    : Sandboxed tool/command executor
  - VulnerabilityFinder : Security specialist
  - Pessimist      : Devil's advocate, stress-tester
  - Documentor     : Comments, docstrings, READMEs
"""

import ast
import json
import math
import os
import re
import subprocess
import textwrap
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
# CogWorks Swarm  —  orchestration entry point
# ===========================================================================

class CogWorksSwarm:
    """
    Instantiates all swarm agents around a shared memory store (and optionally
    a shared CogForge model).  Exposes a single `.run(task, repo_root)` entry
    point that mirrors the documented workflow.

    Workflow:
      1. Coordinator delegates to Planner + Explorer (parallel).
      2. Dreamer injects context.
      3. Planner outputs DAG → ProblemSolver refines.
      4. Engineer → Pessimist → BugFinder + VulnerabilityFinder.
      5. TerminalGuy executes tests.
      6. Documentor annotates.
      7. Coordinator gates on verifier score.
      8. Dreamer consolidates into long-term memory.
    """

    def __init__(self, model: Optional[CogForge] = None,
                 config: Optional[CogForgeConfig] = None):
        self.shared_memory = SharedMemoryStore()
        cfg = config or CogForgeConfig()

        self.coordinator         = Coordinator(self.shared_memory, model)
        self.dreamer             = Dreamer(self.shared_memory, model, cfg)
        self.explorer            = Explorer(self.shared_memory, model)
        self.planner             = Planner(self.shared_memory, model)
        self.problem_solver      = ProblemSolver(self.shared_memory, model)
        self.engineer            = Engineer(self.shared_memory, model)
        self.bug_finder          = BugFinder(self.shared_memory, model)
        self.terminal_guy        = TerminalGuy(self.shared_memory, model)
        self.vulnerability_finder = VulnerabilityFinder(self.shared_memory, model)
        self.pessimist           = Pessimist(self.shared_memory, model)
        self.documentor          = Documentor(self.shared_memory, model)

        self._agents: Dict[str, BaseAgent] = {
            "Coordinator":         self.coordinator,
            "Dreamer":             self.dreamer,
            "Explorer":            self.explorer,
            "Planner":             self.planner,
            "ProblemSolver":       self.problem_solver,
            "Engineer":            self.engineer,
            "BugFinder":           self.bug_finder,
            "TerminalGuy":         self.terminal_guy,
            "VulnerabilityFinder": self.vulnerability_finder,
            "Pessimist":           self.pessimist,
            "Documentor":          self.documentor,
        }

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

        # Step 3: Explorer maps the repo
        explore_msg = self.explorer.run(task, root=repo_root)
        results["explorer"] = explore_msg.artifacts

        # Step 4: Planner decomposes the task
        plan_msg = self.planner.run(task)
        results["planner"] = plan_msg.artifacts

        # Step 5: ProblemSolver analyses and annotates
        ps_msg = self.problem_solver.run(task)
        results["problem_solver"] = ps_msg.artifacts

        # Step 6: Engineer reviews code quality
        eng_msg = self.engineer.run(task)
        results["engineer"] = eng_msg.artifacts

        # Step 7: Pessimist stress-tests
        pess_msg = self.pessimist.run(task)
        results["pessimist"] = pess_msg.artifacts

        # Step 8: BugFinder deep inspection
        bug_msg = self.bug_finder.run(task)
        results["bug_finder"] = bug_msg.artifacts

        # Step 9: VulnerabilityFinder security scan
        vuln_msg = self.vulnerability_finder.run(task)
        results["vulnerability_finder"] = vuln_msg.artifacts

        # Step 10: TerminalGuy runs tests
        term_msg = self.terminal_guy.run(task, command="python -m pytest --tb=short -q")
        results["terminal_guy"] = term_msg.artifacts

        # Step 11: Documentor annotates
        flagged = self.shared_memory.get_flagged_files()
        doc_msg = self.documentor.run(task, files=flagged)
        results["documentor"] = doc_msg.artifacts

        # Step 12: Dreamer consolidates
        self.dreamer.maybe_consolidate_episodic()
        results["dreamer_consolidation"] = "complete"

        # Step 13: Coordinator quality gate
        # Use pessimist confidence as a proxy for verifier score
        verifier_proxy = pess_msg.confidence
        gate_decision = self.coordinator.review_and_gate(verifier_proxy, plan_msg.task_id)
        results["quality_gate"] = gate_decision
        results["verifier_proxy_score"] = verifier_proxy

        return results


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

    # Swarm smoke test (no real repo files — just exercises the pipeline)
    print("\n--- CogWorks Swarm ---")
    swarm = CogWorksSwarm(model=model, config=cfg)
    results = swarm.run("Refactor auth module for better security and testability",
                        repo_root=".")
    print(f"Quality gate decision: {results['quality_gate']}")
    print(f"Verifier proxy score:  {results['verifier_proxy_score']:.3f}")
    print(f"Agents completed:      {list(results.keys())}")
