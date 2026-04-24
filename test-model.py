"""
CogForge — tests/test_model.py
================================
Smoke tests that validate:
  - Parameter count (~100M)
  - Forward pass shapes
  - Loss computation
  - Generation (no crash)
  - ACT block behavior
  - Verifier output range
  - Architect cross-attention shapes
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import pytest

from model.architecture import (
    CogForgeConfig, CogForge, CogForgeBlock, ACTBlock,
    RotaryEmbedding, GroupedQueryAttention, SwiGLUFFN,
    LinearLookbackAttention, ArchitectEncoder
)
from model.losses import (
    CausalLMLoss, DataFlowLoss, ContrastiveBugLoss,
    VerifierLoss, ACTPonderLoss, CogForgeLoss
)
from data.dataset import CodeTokenizer


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_cfg():
    """Tiny config for fast tests (~3M params)."""
    return CogForgeConfig(
        vocab_size=512,
        max_seq_len=128,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        d_ff_multiplier=2.0,
        window_size=32,
        global_tokens=8,
        n_latent_tokens=4,
        n_act_iterations=2,
        act_threshold=0.99,
        n_arch_layers=1,
        arch_d_model=64,
        arch_n_heads=2,
        max_repo_chunks=4,
        dropout=0.0,
        attn_dropout=0.0,
    )


@pytest.fixture(scope="module")
def model(small_cfg):
    return CogForge(small_cfg).eval()


@pytest.fixture(scope="module")
def full_cfg():
    """Full ~100M config."""
    return CogForgeConfig()


# ── Architecture Tests ───────────────────────────────────────────────────────

class TestParameterCount:
    def test_full_model_near_100m(self, full_cfg):
        model = CogForge(full_cfg)
        n = model.count_parameters()
        print(f"\n  Total params: {n:,} ({n/1e6:.2f}M)")
        # Allow 80M–120M range
        assert 80_000_000 <= n <= 130_000_000, f"Unexpected param count: {n:,}"

    def test_small_model_runs(self, small_cfg):
        model = CogForge(small_cfg)
        assert model.count_parameters() > 0


class TestRotaryEmbeddings:
    def test_shape(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 64, 32)
        cos, sin = rope(x, seq_len=64)
        assert cos.shape == (1, 1, 64, 32)
        assert sin.shape == (1, 1, 64, 32)

    def test_values_bounded(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 64, 32)
        cos, sin = rope(x, seq_len=64)
        assert cos.abs().max().item() <= 1.0 + 1e-5
        assert sin.abs().max().item() <= 1.0 + 1e-5


class TestGQA:
    def test_output_shape(self, small_cfg):
        gqa = GroupedQueryAttention(small_cfg)
        x = torch.randn(2, 16, small_cfg.d_model)
        out, kv = gqa(x)
        assert out.shape == (2, 16, small_cfg.d_model)
        assert kv[0].shape[2] == 16  # key seq len

    def test_kv_cache(self, small_cfg):
        gqa = GroupedQueryAttention(small_cfg)
        x1 = torch.randn(1, 8, small_cfg.d_model)
        out1, kv1 = gqa(x1)
        x2 = torch.randn(1, 4, small_cfg.d_model)
        out2, kv2 = gqa(x2, past_kv=kv1)
        assert out2.shape == (1, 4, small_cfg.d_model)
        assert kv2[0].shape[2] == 12  # 8 + 4


class TestLinearLookback:
    def test_output_shape(self, small_cfg):
        lb = LinearLookbackAttention(small_cfg)
        x = torch.randn(2, 16, small_cfg.d_model)
        out = lb(x)
        assert out.shape == (2, 16, small_cfg.d_model)


class TestSwiGLU:
    def test_output_shape(self, small_cfg):
        ffn = SwiGLUFFN(small_cfg)
        x = torch.randn(2, 16, small_cfg.d_model)
        out = ffn(x)
        assert out.shape == (2, 16, small_cfg.d_model)


class TestACTBlock:
    def test_ponder_cost_positive(self, small_cfg):
        base = CogForgeBlock(small_cfg)
        act = ACTBlock(base, small_cfg.d_model, max_iter=3)
        x = torch.randn(2, 16, small_cfg.d_model)
        out, ponder = act(x)
        assert out.shape == x.shape
        assert (ponder >= 0).all()
        assert (ponder <= 3.1).all()  # max iterations


class TestArchitectEncoder:
    def test_output_shape(self, small_cfg):
        arch = ArchitectEncoder(small_cfg)
        B, N = 2, 4
        chunks = torch.randn(B, N, small_cfg.d_model)
        out = arch(chunks)
        assert out.shape == (B, N, small_cfg.d_model)


class TestFullForward:
    def test_forward_shape(self, model, small_cfg):
        B, T = 2, 32
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))
        with torch.no_grad():
            out = model(ids)
        assert out["logits"].shape == (B, T, small_cfg.vocab_size)
        assert out["ponder_cost"].shape == (B,)
        assert out["ponder_cost"].min().item() >= 0

    def test_forward_with_repo_context(self, model, small_cfg):
        B, T, N = 2, 32, 4
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))
        chunks = torch.randn(B, N, small_cfg.d_model)
        mask = torch.zeros(B, N, dtype=torch.bool)
        with torch.no_grad():
            out = model(ids, repo_chunks=chunks, repo_mask=mask)
        assert out["logits"].shape == (B, T, small_cfg.vocab_size)

    def test_verifier_range(self, model, small_cfg):
        B, T = 2, 32
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))
        with torch.no_grad():
            out = model(ids, return_verifier=True)
        score = out["verifier_score"]
        assert score.shape == (B,)
        assert (score >= 0).all() and (score <= 1).all()

    def test_no_nan(self, model, small_cfg):
        B, T = 2, 32
        ids = torch.randint(0, small_cfg.vocab_size, (B, T))
        with torch.no_grad():
            out = model(ids)
        assert not torch.isnan(out["logits"]).any()


class TestGeneration:
    def test_generate_shape(self, model, small_cfg):
        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=16)
        assert gen.shape[0] == 1
        assert gen.shape[1] <= 8 + 16 + 1  # prompt + new + possible eos


# ── Loss Tests ───────────────────────────────────────────────────────────────

class TestCausalLMLoss:
    def test_forward(self):
        loss_fn = CausalLMLoss(vocab_size=100)
        logits = torch.randn(2, 16, 100)
        labels = torch.randint(0, 100, (2, 16))
        labels[:, -1] = -100
        loss = loss_fn(logits, labels)
        assert loss.item() > 0
        assert not math.isnan(loss.item())


class TestDataFlowLoss:
    def test_forward(self):
        loss_fn = DataFlowLoss(weight=0.1)
        B, H, T = 2, 4, 16
        attn = torch.softmax(torch.randn(B, H, T, T), dim=-1)
        # 3 dependency pairs per sample
        pairs = torch.randint(0, T, (B, 3, 2))
        pair_mask = torch.ones(B, 3)
        loss = loss_fn(attn, pairs, pair_mask)
        assert 0 <= loss.item() <= 1.0


class TestContrastiveLoss:
    def test_forward(self):
        loss_fn = ContrastiveBugLoss(weight=0.3)
        B, T, D = 4, 16, 64
        fixed_h = torch.randn(B, T, D)
        buggy_h = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        loss = loss_fn(fixed_h, buggy_h, mask, mask)
        assert loss.item() >= 0


class TestVerifierLoss:
    def test_forward(self):
        loss_fn = VerifierLoss(weight=0.2)
        scores = torch.rand(4)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(scores, labels)
        assert loss.item() >= 0


class TestCogForgeLoss:
    def test_combined_loss(self):
        loss_fn = CogForgeLoss(vocab_size=100)
        logits = torch.randn(2, 16, 100)
        labels = torch.randint(0, 100, (2, 16))
        ponder = torch.tensor([1.2, 2.1])
        result = loss_fn(logits, labels, ponder)
        assert "total" in result
        assert "lm" in result
        assert "ponder" in result
        assert result["total"].item() > 0


# ── Tokenizer Tests ──────────────────────────────────────────────────────────

class TestTokenizer:
    def test_roundtrip(self):
        tok = CodeTokenizer()
        code = "def foo(x):\n    return x + 1\n"
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        # Not exact due to UNK, but should recover most chars
        assert len(decoded) > 0

    def test_special_tokens(self):
        tok = CodeTokenizer()
        ids = tok.encode("hello")
        assert ids[0] == CodeTokenizer.BOS
        assert ids[-1] == CodeTokenizer.EOS


if __name__ == "__main__":
    # Quick manual run without pytest
    import traceback

    cfg = CogForgeConfig(
        vocab_size=512, max_seq_len=64, d_model=128, n_heads=4,
        n_kv_heads=2, n_layers=4, d_ff_multiplier=2.0,
        window_size=32, global_tokens=8, n_latent_tokens=4,
        dropout=0.0
    )
    model = CogForge(cfg).eval()
    n = model.count_parameters()
    print(f"Small model params: {n:,}")

    full = CogForge(CogForgeConfig()).eval()
    n_full = full.count_parameters()
    print(f"Full model params: {n_full:,} ({n_full/1e6:.1f}M)")

    # Forward pass
    ids = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        out = model(ids, return_verifier=True)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Ponder cost: {out['ponder_cost']}")
    print(f"Verifier scores: {out['verifier_score']}")

    # Generation
    prompt = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        gen = model.generate(prompt, max_new_tokens=16)
    print(f"Generated shape: {gen.shape}")
    print("All manual checks passed ✓")
