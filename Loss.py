"""
CogForge Loss Functions
=======================
Beyond standard cross-entropy:
  1. CausalLMLoss          — standard next-token prediction
  2. DataFlowLoss          — penalize broken variable dependency chains
  3. ContrastiveBugLoss    — buggy vs. fixed code contrastive learning
  4. VerifierLoss          — train the value function on binary correctness
  5. ACTPonderLoss         — regularize adaptive computation time
  6. CogForgeLoss          — orchestrates all losses with weighted sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Standard Causal Language Modeling Loss
# ---------------------------------------------------------------------------

class CausalLMLoss(nn.Module):
    """
    Cross-entropy on shifted labels (next-token prediction).
    Ignores padding tokens.
    """

    def __init__(self, vocab_size: int, ignore_index: int = -100,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, vocab_size)
        labels: (B, T)  — shifted input_ids; pad positions = -100
        """
        # Shift: predict token t+1 from hidden state at t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return self.loss_fn(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )


# ---------------------------------------------------------------------------
# 2. Data Flow Loss
# ---------------------------------------------------------------------------

class DataFlowLoss(nn.Module):
    """
    Penalizes the model when it generates tokens at positions that should
    reference a previously-defined variable but instead attend to undefined
    positions.

    In practice, we receive a 'dependency_mask' tensor from static analysis
    (or a synthetic AST parser) indicating which token positions in the label
    are "use" sites and which positions they should attend to (def sites).

    We compute an auxiliary attention consistency loss over those pairs.
    If the model's attention weights are high on the correct def-site,
    the loss is low; otherwise it is penalized.

    This is computed over the GQA attention weights of the last layer.
    For efficiency the calling code passes the attention weights explicitly.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        attn_weights: torch.Tensor,      # (B, H, T, T) last-layer attention
        dependency_pairs: torch.Tensor,  # (B, N_pairs, 2): [(use_pos, def_pos), ...]
        pair_mask: torch.Tensor,         # (B, N_pairs) 1=valid, 0=padding
    ) -> torch.Tensor:
        """
        For each dependency pair (use_i, def_j), we want the model's
        attention at use_i to be high at def_j across all heads.
        """
        if dependency_pairs.numel() == 0:
            return torch.tensor(0.0, device=attn_weights.device)

        B, H, T, _ = attn_weights.shape
        B, N, _ = dependency_pairs.shape

        use_pos = dependency_pairs[:, :, 0]  # (B, N)
        def_pos = dependency_pairs[:, :, 1]  # (B, N)

        # Gather attention at use_pos rows, def_pos columns
        # attn_weights[:, :, use_pos, def_pos]
        use_pos_exp = use_pos.unsqueeze(1).expand(B, H, N)  # (B, H, N)
        def_pos_exp = def_pos.unsqueeze(1).expand(B, H, N)

        # attn_at_def[b, h, n] = attention from use_pos[b,n] to def_pos[b,n]
        attn_at_def = torch.gather(
            attn_weights.view(B, H, T * T),
            dim=2,
            index=(use_pos_exp * T + def_pos_exp).clamp(0, T * T - 1)
        )  # (B, H, N)

        # We want this to be high — loss = 1 - mean(attn_at_def)
        # Masked mean over valid pairs
        valid = pair_mask.unsqueeze(1).float()  # (B, 1, N)
        loss = 1.0 - (attn_at_def * valid).sum(dim=(1, 2)) / \
               (valid.sum(dim=(1, 2)) * H).clamp(min=1)
        return self.weight * loss.mean()


# ---------------------------------------------------------------------------
# 3. Contrastive Bug Loss
# ---------------------------------------------------------------------------

class ContrastiveBugLoss(nn.Module):
    """
    Unit-test contrastive learning on (buggy_code, fixed_code) pairs.

    Given a batch that interleaves [fixed, buggy, fixed, buggy, ...],
    we:
      1. Produce mean-pooled representations for each sample.
      2. Apply contrastive loss so fixed and buggy of the SAME problem
         are pulled apart, and different-problem fixed codes are orthogonal.
      3. Optionally: treat each (fixed, buggy) pair as an InfoNCE pair
         where the fixed version is the anchor and the buggy is a negative.

    This teaches the model the nuanced difference between correct and
    nearly-correct (buggy) code.
    """

    def __init__(self, temperature: float = 0.07, weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padding positions."""
        mask_f = mask.float().unsqueeze(-1)
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

    def forward(
        self,
        fixed_hidden: torch.Tensor,   # (B, T, D) — hidden states of fixed code
        buggy_hidden: torch.Tensor,   # (B, T, D) — hidden states of buggy code
        fixed_mask: torch.Tensor,     # (B, T)
        buggy_mask: torch.Tensor,     # (B, T)
    ) -> torch.Tensor:
        B = fixed_hidden.shape[0]

        z_fixed = F.normalize(self._mean_pool(fixed_hidden, fixed_mask), dim=-1)
        z_buggy = F.normalize(self._mean_pool(buggy_hidden, buggy_mask), dim=-1)

        # InfoNCE: fixed[i] should NOT be similar to buggy[i] (same problem)
        # but fixed[i] from different problems are neutral.
        # Treat (fixed[i], buggy[i]) as negative pair; all cross-problem
        # fixed-fixed pairs can be positives (they're all correct code).

        # Compute similarity matrix among all 2B samples
        z_all = torch.cat([z_fixed, z_buggy], dim=0)  # (2B, D)
        sim = torch.matmul(z_all, z_all.T) / self.temperature  # (2B, 2B)

        # Mask self-similarity
        sim.fill_diagonal_(float("-inf"))

        # Labels: fixed[i]'s positives are other fixed codes (0..B-1 except i)
        # We use a simple formulation: cross-entropy over fixed samples only
        # treating buggy[i] as the hardest negative.

        # For each fixed[i], the target is the average of other fixed vectors.
        # Simpler: use a symmetrised loss where the positive for fixed[i]
        # is the *average* fixed embedding (mean of others).

        # Practical: use NT-Xent style
        labels = torch.arange(B, device=fixed_hidden.device)
        # fixed[i] → positive: buggy[i] is the NEGATIVE, but we want to
        # CONTRAST fixed vs. buggy. Use buggy as the hard negative:
        # logits for fixed[i]: similarity to all buggy[j]
        logits_fb = torch.matmul(z_fixed, z_buggy.T) / self.temperature  # (B, B)
        # We want fixed[i] to be LEAST similar to buggy[i] specifically,
        # but NT-Xent needs a positive pair. Here we invert: treat the
        # diagonal as the "negative we want to push away" and off-diag as
        # the signal. Instead, use a margin-based approach:

        # Margin ranking: sim(fixed_i, fixed_j≠i) > sim(fixed_i, buggy_i) + margin
        sim_ff = torch.matmul(z_fixed, z_fixed.T) / self.temperature
        sim_fb_diag = (z_fixed * z_buggy).sum(-1)  # (B,) — same-problem similarity

        # Mean off-diagonal fixed-fixed similarity
        mask_offdiag = ~torch.eye(B, dtype=torch.bool, device=fixed_hidden.device)
        sim_ff_offdiag = sim_ff[mask_offdiag].view(B, B - 1).mean(dim=-1)

        # Want: sim_ff_offdiag > sim_fb_diag  (correct code more similar to
        # other correct code than to its buggy counterpart)
        margin = 0.2
        contrastive = F.relu(sim_fb_diag - sim_ff_offdiag + margin).mean()
        return self.weight * contrastive


# ---------------------------------------------------------------------------
# 4. Verifier (Value Function) Loss
# ---------------------------------------------------------------------------

class VerifierLoss(nn.Module):
    """
    Binary cross-entropy to train the verifier head.
    Labels: 1 = code passes tests, 0 = code fails tests.
    In training we use execution-based rewards (compiler/test-suite).
    """

    def __init__(self, weight: float = 0.2):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCELoss()

    def forward(self, scores: torch.Tensor,
                correctness_labels: torch.Tensor) -> torch.Tensor:
        """
        scores: (B,)  — output of VerifierHead, sigmoid-normalized
        correctness_labels: (B,) — float 0.0 or 1.0
        """
        return self.weight * self.bce(scores, correctness_labels.float())


# ---------------------------------------------------------------------------
# 5. ACT Ponder Cost Regularizer
# ---------------------------------------------------------------------------

class ACTPonderLoss(nn.Module):
    """
    Regularizes the Adaptive Computation Time (ACT) halting mechanism.
    Penalizes excessive computation via the ponder cost (expected number
    of steps). Balances accuracy against compute budget.

    L_ponder = λ · mean(ponder_cost)
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, ponder_cost: torch.Tensor) -> torch.Tensor:
        """ponder_cost: (B,) accumulated halting steps."""
        return self.weight * ponder_cost.mean()


# ---------------------------------------------------------------------------
# 6. Combined CogForge Loss
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    lm: float = 1.0
    dataflow: float = 0.1
    contrastive: float = 0.3
    verifier: float = 0.2
    ponder: float = 0.01


class CogForgeLoss(nn.Module):
    """
    Orchestrates all CogForge losses.

    Score = P(R|P) · V(R)          (reasoning path × verifier)
    Total = L_LM + L_flow + L_contrast + L_verifier + L_ponder
    """

    def __init__(self, vocab_size: int, weights: Optional[LossWeights] = None):
        super().__init__()
        w = weights or LossWeights()
        self.lm_loss         = CausalLMLoss(vocab_size)
        self.dataflow_loss   = DataFlowLoss(weight=w.dataflow)
        self.contrastive_loss = ContrastiveBugLoss(weight=w.contrastive)
        self.verifier_loss   = VerifierLoss(weight=w.verifier)
        self.ponder_loss     = ACTPonderLoss(weight=w.ponder)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ponder_cost: torch.Tensor,
        # Optional advanced losses
        attn_weights: Optional[torch.Tensor] = None,
        dependency_pairs: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        fixed_hidden: Optional[torch.Tensor] = None,
        buggy_hidden: Optional[torch.Tensor] = None,
        fixed_mask: Optional[torch.Tensor] = None,
        buggy_mask: Optional[torch.Tensor] = None,
        verifier_scores: Optional[torch.Tensor] = None,
        correctness_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        losses = {}

        # ── 1. LM loss (always) ───────────────────────────────────────────
        losses["lm"] = self.lm_loss(logits, labels)
        total = losses["lm"]

        # ── 2. Data-flow loss ─────────────────────────────────────────────
        if attn_weights is not None and dependency_pairs is not None:
            losses["dataflow"] = self.dataflow_loss(
                attn_weights, dependency_pairs, pair_mask
            )
            total = total + losses["dataflow"]

        # ── 3. Contrastive bug loss ───────────────────────────────────────
        if fixed_hidden is not None and buggy_hidden is not None:
            losses["contrastive"] = self.contrastive_loss(
                fixed_hidden, buggy_hidden, fixed_mask, buggy_mask
            )
            total = total + losses["contrastive"]

        # ── 4. Verifier loss ──────────────────────────────────────────────
        if verifier_scores is not None and correctness_labels is not None:
            losses["verifier"] = self.verifier_loss(verifier_scores, correctness_labels)
            total = total + losses["verifier"]

        # ── 5. ACT ponder cost ────────────────────────────────────────────
        losses["ponder"] = self.ponder_loss(ponder_cost)
        total = total + losses["ponder"]

        losses["total"] = total
        return losses
