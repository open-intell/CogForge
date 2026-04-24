"""
CogForge: A Coding & Reasoning Transformer (~100M parameters)
=============================================================
Architecture:
  - Grouped-Query Attention (GQA) with Sliding Window
  - Rotary Positional Embeddings (RoPE)
  - Adaptive Computation Time (ACT) blocks
  - Latent Reasoning Tokens
  - Linear "Look-back" Attention for long-range context
  - Architect cross-attention module (repo-level context)
  - SwiGLU feed-forward networks
  - RMSNorm throughout
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CogForgeConfig:
    # Vocabulary & sequence
    vocab_size: int = 32000          # BPE vocab (code-tuned)
    max_seq_len: int = 4096
    pad_token_id: int = 0

    # Model dimensions — tuned to ~100M params
    d_model: int = 768               # hidden size
    n_heads: int = 12                # query heads
    n_kv_heads: int = 4              # key/value heads (GQA grouping)
    n_layers: int = 12               # transformer blocks
    d_ff_multiplier: float = 2.6667  # SwiGLU hidden = d_model * this ≈ 2048

    # Sliding window attention
    window_size: int = 512           # local attention window
    global_tokens: int = 64         # tokens that attend globally (BOS + specials)

    # Latent reasoning
    n_latent_tokens: int = 8        # "think" tokens prepended per example
    n_act_iterations: int = 3       # max ACT loops per block
    act_threshold: float = 0.99     # halting probability threshold

    # Architect (repo-level) cross-attention
    n_arch_layers: int = 2           # layers in the Architect encoder
    arch_d_model: int = 256          # Architect hidden size
    arch_n_heads: int = 4
    max_repo_chunks: int = 32        # max retrieved code chunks

    # Dropout & regularization
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0       # set >1 for extended context

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        # Round to nearest multiple of 256 for efficiency
        raw = int(self.d_model * self.d_ff_multiplier)
        return (raw // 256) * 256  # → 2048

    @property
    def kv_groups(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# Utility: RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Grouped-Query Attention with Sliding Window
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    GQA: n_heads query heads share n_kv_heads key/value heads.
    Supports sliding-window local attention with optional global tokens.
    """

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
        """Repeat KV heads to match query head count."""
        B, H, T, D = kv.shape
        kv = kv.unsqueeze(2).expand(B, H, self.kv_groups, T, D)
        return kv.reshape(B, H * self.kv_groups, T, D)

    def _sliding_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask restricted to window_size, except global_tokens attend everywhere.
        Returns additive mask of shape (T, T).
        """
        mask = torch.full((T, T), float("-inf"), device=device)
        # Each position attends to [max(0, i-window+1) .. i]
        for i in range(T):
            lo = max(0, i - self.window_size + 1)
            mask[i, lo:i + 1] = 0.0
        # Global tokens attend & are attended to everywhere (causal)
        for g in range(min(self.global_tokens, T)):
            mask[g, :g + 1] = 0.0   # global token attends back
            mask[:, g] = 0.0         # everyone attends to global tokens
        # Re-enforce causality: upper triangle = -inf
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

        # RoPE
        cos, sin = self.rotary(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        # KV-cache concat
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        # Expand KV for GQA
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        # Attention scores
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T_k)

        # Sliding window mask (only for full sequences, not generation step)
        T_k = k.shape[2]
        if T > 1:
            attn_mask = self._sliding_window_mask(T_k, x.device)
            # For generation, T < T_k, just apply causal
            if T == T_k:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                causal = torch.triu(
                    torch.ones(T, T_k, device=x.device) * float("-inf"), diagonal=1
                )
                scores = scores + causal.unsqueeze(0).unsqueeze(0)

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


# ---------------------------------------------------------------------------
# Linear "Look-back" Attention
# ---------------------------------------------------------------------------

class LinearLookbackAttention(nn.Module):
    """
    Efficient linear attention using kernel trick: φ(Q)·(φ(K)ᵀV).
    Allows O(T) retrieval of distant context (e.g., library definitions).
    Used as a parallel branch alongside the sliding-window GQA.
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_heads = config.n_kv_heads  # fewer heads — just context retrieval
        self.d_head = config.d_head
        dim = config.n_kv_heads * config.d_head

        self.q_proj = nn.Linear(config.d_model, dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, dim, bias=False)
        self.o_proj = nn.Linear(dim, config.d_model, bias=False)
        self.gate   = nn.Linear(config.d_model, config.d_model, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """ELU + 1 kernel approximation (standard linear attn trick)."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q, k = self._feature_map(q), self._feature_map(k)

        # Causal linear attention: S_t = S_{t-1} + k_t^T v_t
        # out_t = q_t S_t / (q_t z_t)  where z_t = cumsum(k_t)
        kv = torch.einsum("bhte,bhtd->bhed", k, v)  # (B, H, D, D)
        k_sum = k.sum(dim=2, keepdim=True)           # (B, H, 1, D)
        # full causal via cumsum over tokens
        kv_cum = torch.cumsum(
            torch.einsum("bhte,bhtd->bhtde", k, v), dim=2
        )  # (B, H, T, D, D) — expensive but correct; for large T use scan
        z_cum = torch.cumsum(k, dim=2)               # (B, H, T, D)

        out = torch.einsum("bhte,bhted->bhtd", q, kv_cum)
        denom = torch.einsum("bhte,bhte->bht", q, z_cum).unsqueeze(-1).clamp(min=1e-6)
        out = (out / denom).transpose(1, 2).contiguous().view(B, T, -1)

        gate = torch.sigmoid(self.gate(x))
        return self.o_proj(out) * gate


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: FFN(x) = (xW₁ ⊗ SiLU(xW_g)) W₂
    Two parallel projections, element-wise gated.
    """

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


# ---------------------------------------------------------------------------
# Adaptive Computation Time (ACT) Wrapper
# ---------------------------------------------------------------------------

class ACTBlock(nn.Module):
    """
    Wraps a transformer block with ACT: the model learns a halting probability
    h_t at each iteration. It "loops" the block until cumulative halting ≥ threshold
    or max_iterations is reached. Complex code tokens get more compute.

    Ponder cost is returned for regularization during training.
    """

    def __init__(self, block: nn.Module, d_model: int,
                 max_iter: int = 3, threshold: float = 0.99):
        super().__init__()
        self.block = block
        self.max_iter = max_iter
        self.threshold = threshold
        self.halting_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, **block_kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device = x.device

        halted = torch.zeros(B, T, 1, device=device)
        accum_output = torch.zeros_like(x)
        remainder = torch.ones(B, T, 1, device=device)
        ponder_cost = torch.zeros(B, T, device=device)

        state = x
        for step in range(self.max_iter):
            h = torch.sigmoid(self.halting_proj(state))  # (B, T, 1)
            still_running = (halted < self.threshold).float()

            # Last step: use remainder
            is_last = (step == self.max_iter - 1)
            if is_last:
                used_h = remainder * still_running
            else:
                used_h = h * still_running
                # Clamp so we don't exceed 1
                will_exceed = ((halted + used_h) > self.threshold).float()
                used_h = used_h * (1 - will_exceed) + remainder * will_exceed

            halted = halted + used_h
            ponder_cost += (used_h.squeeze(-1) * still_running.squeeze(-1))

            state, _ = self.block(state, **block_kwargs)
            accum_output = accum_output + used_h * state
            remainder = remainder - used_h

            if (halted >= self.threshold).all():
                break

        return accum_output, ponder_cost


# ---------------------------------------------------------------------------
# Base Transformer Block
# ---------------------------------------------------------------------------

class CogForgeBlock(nn.Module):
    """
    Single transformer block with:
      - Pre-norm GQA (sliding window)
      - Pre-norm Linear look-back attention (additive branch)
      - Pre-norm SwiGLU FFN
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.norm3 = RMSNorm(config.d_model)

        self.gqa        = GroupedQueryAttention(config)
        self.lookback   = LinearLookbackAttention(config)
        self.ffn        = SwiGLUFFN(config)
        self.drop       = nn.Dropout(config.dropout)

        # Learnable mixing of local GQA and global look-back
        self.mix_alpha  = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                past_kv: Optional[Tuple] = None
                ) -> Tuple[torch.Tensor, Tuple]:
        # Local attention
        h, new_kv = self.gqa(self.norm1(x), past_kv)
        # Linear look-back (global context)
        lb = self.lookback(self.norm2(x))
        # Combine: mostly local, blended with global
        x = x + self.drop(h) + self.drop(lb) * torch.sigmoid(self.mix_alpha)
        # FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, new_kv


# ---------------------------------------------------------------------------
# Architect Encoder (Repo-Level Context)
# ---------------------------------------------------------------------------

class ArchitectEncoder(nn.Module):
    """
    A lightweight transformer encoder that processes retrieved code chunks
    (up to max_repo_chunks) and produces a compressed "repo map" that is
    fed into the main decoder via cross-attention.

    Input: list of chunk embeddings [B, N_chunks, L_chunk, d_model_embed]
             → mean-pooled per chunk → [B, N_chunks, arch_d_model]
    Output: [B, N_chunks, d_model]  (projected to main model dim)
    """

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
        """
        chunk_embeds: (B, N_chunks, d_model) — mean-pooled chunk representations
        chunk_mask:   (B, N_chunks) — True = padding
        Returns:      (B, N_chunks, d_model)
        """
        h = self.chunk_proj(chunk_embeds)          # (B, N, arch_d)
        h = self.encoder(h, src_key_padding_mask=chunk_mask)
        h = self.out_proj(h)
        return self.norm(h)


class ArchitectCrossAttention(nn.Module):
    """Cross-attention from main sequence into architect repo context."""

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.n_heads = config.n_kv_heads  # light — fewer heads
        self.d_head  = config.d_head
        dim = self.n_heads * self.d_head

        self.q_proj  = nn.Linear(config.d_model, dim, bias=False)
        self.k_proj  = nn.Linear(config.d_model, dim, bias=False)
        self.v_proj  = nn.Linear(config.d_model, dim, bias=False)
        self.o_proj  = nn.Linear(dim, config.d_model, bias=False)
        self.norm_q  = RMSNorm(config.d_model)
        self.norm_c  = RMSNorm(config.d_model)

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
            # context_mask: (B, N) True=pad → -inf
            scores = scores.masked_fill(
                context_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        return x + self.o_proj(out)


# ---------------------------------------------------------------------------
# Value Function / Verifier Head (V(R))
# ---------------------------------------------------------------------------

class VerifierHead(nn.Module):
    """
    Learned value function that predicts correctness of a reasoning path.
    Used both during training (auxiliary loss) and inference (beam re-ranking).
    Score ∈ (0, 1): probability the generated code is logically correct.
    """

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
        """
        hidden: (B, T, d_model)
        mask:   (B, T) 1=real token, 0=pad
        Returns: (B,) scalar correctness score per sequence
        """
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        pooled = F.gelu(self.pool(pooled))
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Full CogForge Model
# ---------------------------------------------------------------------------

class CogForge(nn.Module):
    """
    CogForge: A transformer designed for coding and reasoning.

    ~100M parameter breakdown (d=768, L=12, h=12, kv=4, ff=2048, vocab=32000):
      Token embed:   32000 × 768           =  24.58M
      Blocks × 12:
        GQA Q proj:    768 × 768           =   0.59M
        GQA KV proj:   768 × 256 × 2       =   0.39M
        GQA O proj:    768 × 768           =   0.59M
        LinearLookback (4 projs, smaller)  =   0.55M
        SwiGLU (gate+up+down)              =   3.15M
        Norms + misc                       =   0.01M
        ACT halting proj                   =   0.001M
        Mix alpha (scalar)                 =   0.000M
        ≈ per block                        =   5.28M  × 12 = 63.36M
      Architect encoder (2 layers):        =   0.80M
      Architect cross-attn × 6 layers:     =   1.00M
      Verifier head:                       =   0.59M
      Latent token embeddings:             =   0.006M
      LM head (tied):                      =   0.00M (tied to embed)
      TOTAL                               ≈ 90–100M
    """

    def __init__(self, config: CogForgeConfig):
        super().__init__()
        self.config = config

        # Token embedding (shared with LM head)
        self.embed = nn.Embedding(config.vocab_size, config.d_model,
                                  padding_idx=config.pad_token_id)

        # Learnable latent reasoning tokens (prepended at forward time)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, config.d_model) * 0.02
        )

        # Transformer blocks (half use ACT, half are standard for efficiency)
        self.blocks = nn.ModuleList()
        self.act_blocks: list[int] = []          # track which layers use ACT
        for i in range(config.n_layers):
            base = CogForgeBlock(config)
            if i >= config.n_layers // 2:        # ACT on deeper layers only
                self.blocks.append(ACTBlock(base, config.d_model,
                                            config.n_act_iterations,
                                            config.act_threshold))
                self.act_blocks.append(i)
            else:
                self.blocks.append(base)

        # Architect module (repo-level context) — injected at layers 4 and 8
        self.architect_encoder = ArchitectEncoder(config)
        self.arch_cross_attn = nn.ModuleDict({
            "4": ArchitectCrossAttention(config),
            "8": ArchitectCrossAttention(config),
        })

        # Final norm + LM head
        self.norm_out = RMSNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        # Verifier (value function)
        self.verifier = VerifierHead(config)

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
        input_ids: torch.Tensor,                        # (B, T)
        repo_chunks: Optional[torch.Tensor] = None,     # (B, N_chunks, d_model)
        repo_mask: Optional[torch.Tensor] = None,       # (B, N_chunks) True=pad
        past_kvs: Optional[list] = None,
        return_verifier: bool = False,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device

        # ── Embed tokens ──────────────────────────────────────────────────
        x = self.drop(self.embed(input_ids))

        # ── Prepend latent reasoning tokens ───────────────────────────────
        latents = self.latent_tokens.expand(B, -1, -1)
        x = torch.cat([latents, x], dim=1)  # (B, n_latent + T, D)

        # ── Architect context ──────────────────────────────────────────────
        arch_ctx = None
        if repo_chunks is not None:
            arch_ctx = self.architect_encoder(repo_chunks, repo_mask)

        # ── Transformer blocks ─────────────────────────────────────────────
        new_kvs = []
        total_ponder = torch.zeros(B, device=device)
        block_past_kvs = past_kvs or [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            # Inject architect cross-attention at layers 4 & 8
            layer_key = str(i)
            if arch_ctx is not None and layer_key in self.arch_cross_attn:
                x = self.arch_cross_attn[layer_key](x, arch_ctx, repo_mask)

            if i in self.act_blocks:
                x, ponder = block(x, past_kv=block_past_kvs[i])
                total_ponder = total_ponder + ponder.mean(dim=1)
                new_kvs.append(None)  # ACT blocks manage state internally
            else:
                x, kv = block(x, past_kv=block_past_kvs[i])
                new_kvs.append(kv)

        # ── Remove latent tokens before output ────────────────────────────
        x = x[:, self.config.n_latent_tokens:, :]  # (B, T, D)
        x = self.norm_out(x)

        # ── LM head ───────────────────────────────────────────────────────
        logits = self.lm_head(x)  # (B, T, vocab_size)

        out = {
            "logits": logits,
            "ponder_cost": total_ponder,
            "past_kvs": new_kvs,
        }

        # ── Verifier ──────────────────────────────────────────────────────
        if return_verifier:
            attn_mask = (input_ids != self.config.pad_token_id).float()
            out["verifier_score"] = self.verifier(x, attn_mask)

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
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p nucleus sampling."""
        self.eval()
        generated = input_ids.clone()
        past_kvs = None

        for _ in range(max_new_tokens):
            out = self.forward(
                generated if past_kvs is None else generated[:, -1:],
                repo_chunks=repo_chunks,
                repo_mask=repo_mask,
                past_kvs=past_kvs,
            )
            logits = out["logits"][:, -1, :]  # (B, vocab)
            past_kvs = out["past_kvs"]

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))

            # Top-p (nucleus)
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
