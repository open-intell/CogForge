"""
CogForge Training Pipeline
==========================
Stages:
  1. Pretraining         — causal LM on code corpora
  2. Supervised Fine-Tuning (SFT) — instruction-following on code tasks
  3. RLHF / Execution-Based RL — compiler-as-judge reward signal

Includes:
  - Gradient accumulation & mixed precision (bfloat16)
  - Cosine LR schedule with warmup
  - Gradient clipping + weight decay
  - Checkpoint saving / resumption
  - WandB logging (optional)
  - Multi-GPU via torch.nn.parallel.DistributedDataParallel (DDP)
"""

import math
import os
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from Architecture import CogForge, CogForgeConfig
from Loss import CogForgeLoss, LossWeights


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Paths
    output_dir: str = "./checkpoints"
    run_name: str = "cogforge_100m"

    # Training duration
    max_steps: int = 100_000
    warmup_steps: int = 2_000
    eval_interval: int = 500
    save_interval: int = 1_000
    log_interval: int = 10

    # Batch & sequence
    batch_size: int = 8             # per-GPU micro-batch
    grad_accum_steps: int = 8       # effective batch = 64 sequences
    max_seq_len: int = 4096

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eps: float = 1e-8

    # Mixed precision
    use_amp: bool = True
    dtype: str = "bfloat16"        # "float16" or "bfloat16"

    # Regularization
    act_ponder_weight: float = 0.01

    # RLHF stage
    rlhf_enabled: bool = False
    rl_kl_coeff: float = 0.1       # KL divergence penalty vs. reference model
    rl_reward_scale: float = 1.0
    rl_steps: int = 10_000

    # Loss weights
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # WandB
    use_wandb: bool = False
    wandb_project: str = "cogforge"


# ---------------------------------------------------------------------------
# Learning Rate Schedule: Cosine with Warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainingConfig) -> float:
    """
    Linear warmup then cosine decay to min_lr.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Execution-Based Reward (compiler/interpreter judge)
# ---------------------------------------------------------------------------

class ExecutionReward:
    """
    Simulates a compiler/test-runner reward signal for RLHF.
    In production, spawn subprocess running Python / a sandbox.

    Returns scalar reward ∈ [−1, +1]:
       +1.0  code compiles and passes all unit tests
        0.0  code compiles but fails tests
       -1.0  syntax error / runtime error
    """

    @staticmethod
    def score(code_str: str, test_cases: List[Dict]) -> float:
        """
        Attempt to compile and exec code_str, then run test_cases.
        test_cases: [{"input": ..., "expected_output": ...}]
        """
        try:
            bytecode = compile(code_str, "<cogforge>", "exec")
        except SyntaxError:
            return -1.0

        namespace: Dict = {}
        try:
            exec(bytecode, namespace)
        except Exception:
            return -1.0

        if not test_cases:
            return 0.0

        passed = 0
        for tc in test_cases:
            try:
                fn_name = tc.get("function", "solution")
                if fn_name not in namespace:
                    continue
                result = namespace[fn_name](*tc.get("args", []))
                if result == tc["expected"]:
                    passed += 1
            except Exception:
                pass

        return (2.0 * passed / len(test_cases)) - 1.0  # scale to [-1, 1]

    @staticmethod
    def batch_score(
        generated_ids: torch.Tensor,   # (B, T)
        tokenizer,
        test_cases_batch: List[List[Dict]],
    ) -> torch.Tensor:
        """Decode and score a batch. Returns (B,) reward tensor."""
        rewards = []
        for i in range(generated_ids.shape[0]):
            code = tokenizer.decode(generated_ids[i].tolist())
            r = ExecutionReward.score(code, test_cases_batch[i] if test_cases_batch else [])
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# RLHF / PPO-lite: Policy Gradient with KL Penalty
# ---------------------------------------------------------------------------

class RLHFTrainer:
    """
    Simplified REINFORCE with KL-regularization against a frozen reference model.
    The "reward" comes from the execution-based judge.

    Policy gradient objective:
      L_RL = -E[r(y) · log P_θ(y|x)] + β · KL(P_θ || P_ref)

    We use a "leave-one-out" baseline to reduce variance:
      advantage_i = r_i - mean(r_{j≠i})
    """

    def __init__(
        self,
        policy_model: CogForge,
        ref_model: CogForge,          # frozen reference (SFT checkpoint)
        tokenizer,
        cfg: TrainingConfig,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.cfg = cfg

        for p in self.ref.parameters():
            p.requires_grad_(False)

    def compute_pg_loss(
        self,
        input_ids: torch.Tensor,        # (B, T_prompt)
        generated_ids: torch.Tensor,    # (B, T_gen) — sampled from policy
        rewards: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        B, T_gen = generated_ids.shape
        device = input_ids.device

        # Full sequence: [prompt | generation]
        full_ids = torch.cat([input_ids, generated_ids], dim=1)

        # Policy log probs over generated tokens
        policy_out = self.policy(full_ids)
        policy_logits = policy_out["logits"][:, input_ids.shape[1] - 1: -1, :]
        policy_logprob = F.log_softmax(policy_logits, dim=-1)
        gen_logprob = policy_logprob.gather(
            -1, generated_ids.unsqueeze(-1)
        ).squeeze(-1).sum(-1)  # (B,)

        # Reference log probs (for KL)
        with torch.no_grad():
            ref_out = self.ref(full_ids)
            ref_logits = ref_out["logits"][:, input_ids.shape[1] - 1: -1, :]
            ref_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_gen_logprob = ref_logprob.gather(
                -1, generated_ids.unsqueeze(-1)
            ).squeeze(-1).sum(-1)

        # LOO baseline
        r = rewards.to(device)
        advantage = r - (r.sum() - r) / max(1, B - 1)

        # Scaled rewards
        advantage = advantage * self.cfg.rl_reward_scale

        # Policy gradient loss (negative because we ascend reward)
        pg_loss = -(advantage.detach() * gen_logprob).mean()

        # KL penalty: KL(policy || ref) ≈ logprob_policy - logprob_ref
        kl = (gen_logprob - ref_gen_logprob).mean()
        kl_loss = self.cfg.rl_kl_coeff * kl

        return pg_loss + kl_loss


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, output_dir: str, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved: List[Path] = []

    def save(self, step: int, model: CogForge, optimizer: torch.optim.Optimizer,
             scaler: Optional[GradScaler], metrics: Dict):
        path = self.output_dir / f"ckpt_step{step:07d}.pt"
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "config": asdict(model.config),
        }, path)
        self._saved.append(path)
        # Remove old checkpoints
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()
        # Save latest pointer
        with open(self.output_dir / "latest.txt", "w") as f:
            f.write(str(path))
        print(f"[ckpt] Saved {path}")

    def load_latest(self, model: CogForge, optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None) -> int:
        pointer = self.output_dir / "latest.txt"
        if not pointer.exists():
            return 0
        path = Path(pointer.read_text().strip())
        if not path.exists():
            return 0
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if optimizer and ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        step = ckpt["step"]
        print(f"[ckpt] Resumed from {path} at step {step}")
        return step


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    def __init__(self):
        self._data: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._data.setdefault(k, []).append(float(v))

    def average(self, reset: bool = True) -> Dict[str, float]:
        avg = {k: sum(v) / len(v) for k, v in self._data.items() if v}
        if reset:
            self._data.clear()
        return avg


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class CogForgeTrainer:
    """
    Orchestrates pretraining → SFT → RLHF.
    """

    def __init__(
        self,
        model_config: CogForgeConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        tokenizer=None,
    ):
        self.model_cfg = model_config
        self.train_cfg = train_config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer

        torch.manual_seed(train_config.seed)
        self.device = torch.device(train_config.device)

        # ── Model ─────────────────────────────────────────────────────────
        self.model = CogForge(model_config).to(self.device)
        n_params = self.model.count_parameters()
        print(f"[model] Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

        # ── Optimizer ─────────────────────────────────────────────────────
        decay_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and p.dim() < 2]
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            eps=train_config.eps,
            fused=torch.cuda.is_available(),  # fused AdamW on CUDA
        )

        # ── Mixed precision ────────────────────────────────────────────────
        self.amp_dtype = torch.bfloat16 if train_config.dtype == "bfloat16" \
                         else torch.float16
        self.scaler = GradScaler(enabled=(train_config.use_amp and
                                          self.amp_dtype == torch.float16))

        # ── Loss ──────────────────────────────────────────────────────────
        self.loss_fn = CogForgeLoss(model_config.vocab_size, train_config.loss_weights)

        # ── Checkpointing ─────────────────────────────────────────────────
        self.ckpt_manager = CheckpointManager(train_config.output_dir)
        self.metrics = MetricTracker()
        self.global_step = 0

        # ── WandB ─────────────────────────────────────────────────────────
        self.wandb = None
        if train_config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=train_config.wandb_project,
                    name=train_config.run_name,
                    config={**asdict(model_config), **asdict(train_config)},
                )
            except ImportError:
                print("[warn] wandb not installed, skipping")

    @contextmanager
    def _autocast(self):
        if self.train_cfg.use_amp:
            with autocast(dtype=self.amp_dtype):
                yield
        else:
            yield

    # ── Single Training Step ───────────────────────────────────────────────

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        input_ids  = batch["input_ids"].to(self.device)
        labels     = batch["labels"].to(self.device)
        repo_chunks = batch.get("repo_chunks")
        repo_mask   = batch.get("repo_mask")
        if repo_chunks is not None:
            repo_chunks = repo_chunks.to(self.device)
            repo_mask   = repo_mask.to(self.device)

            # Project chunk token ids to embeddings for Architect
            with torch.no_grad():
                B, N, L = repo_chunks.shape
                flat = repo_chunks.view(B * N, L)
                chunk_embeds = self.model.embed(flat)
                # Mean pool per chunk
                chunk_mask_float = (flat != 0).float().unsqueeze(-1)
                chunk_embeds = (chunk_embeds * chunk_mask_float).sum(1) / \
                               chunk_mask_float.sum(1).clamp(min=1)
                chunk_embeds = chunk_embeds.view(B, N, -1)

        with self._autocast():
            model_out = self.model(
                input_ids,
                repo_chunks=chunk_embeds if repo_chunks is not None else None,
                repo_mask=repo_mask,
                return_verifier=False,
            )

            loss_dict = self.loss_fn(
                logits=model_out["logits"],
                labels=labels,
                ponder_cost=model_out["ponder_cost"],
            )
            loss = loss_dict["total"] / self.train_cfg.grad_accum_steps

        # Backward
        self.scaler.scale(loss).backward()

        return {k: v.item() for k, v in loss_dict.items()}

    # ── Optimizer Step ─────────────────────────────────────────────────────

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.train_cfg.grad_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # Update LR
        lr = get_lr(self.global_step, self.train_cfg)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}
        self.model.eval()
        eval_metrics = MetricTracker()
        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)
            with self._autocast():
                out = self.model(input_ids)
                losses = self.loss_fn(
                    logits=out["logits"],
                    labels=labels,
                    ponder_cost=out["ponder_cost"],
                )
            eval_metrics.update({f"eval/{k}": v.item() for k, v in losses.items()})
            # Perplexity
            eval_metrics.update({"eval/ppl": math.exp(min(losses["lm"].item(), 10))})

        return eval_metrics.average()

    # ── Main Train Loop ───────────────────────────────────────────────────

    def train(self, resume: bool = True):
        if resume:
            self.global_step = self.ckpt_manager.load_latest(
                self.model, self.optimizer, self.scaler
            )

        self.optimizer.zero_grad(set_to_none=True)
        data_iter = iter(self.train_loader)
        accum_count = 0
        t0 = time.time()

        print(f"[train] Starting from step {self.global_step}")

        while self.global_step < self.train_cfg.max_steps:
            # ── Get batch ─────────────────────────────────────────────────
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # ── Forward + backward ────────────────────────────────────────
            step_losses = self._train_step(batch)
            self.metrics.update({f"train/{k}": v for k, v in step_losses.items()})
            accum_count += 1

            # ── Optimizer step (every grad_accum_steps micro-batches) ─────
            if accum_count == self.train_cfg.grad_accum_steps:
                self._optimizer_step()
                self.global_step += 1
                accum_count = 0

                # ── Logging ───────────────────────────────────────────────
                if self.global_step % self.train_cfg.log_interval == 0:
                    elapsed = time.time() - t0
                    avg = self.metrics.average()
                    lr = self.optimizer.param_groups[0]["lr"]
                    tokens_per_sec = (
                        self.train_cfg.batch_size *
                        self.train_cfg.grad_accum_steps *
                        self.train_cfg.max_seq_len *
                        self.train_cfg.log_interval / elapsed
                    )
                    avg["train/lr"] = lr
                    avg["train/tokens_per_sec"] = tokens_per_sec
                    avg["train/step"] = self.global_step
                    avg["train/ppl"] = math.exp(min(avg.get("train/lm", 10), 10))

                    print(
                        f"step {self.global_step:6d} | "
                        f"loss {avg.get('train/total', 0):.4f} | "
                        f"ppl {avg.get('train/ppl', 0):.2f} | "
                        f"lr {lr:.2e} | "
                        f"{tokens_per_sec/1e3:.1f}K tok/s"
                    )

                    if self.wandb:
                        self.wandb.log(avg, step=self.global_step)
                    t0 = time.time()

                # ── Evaluation ────────────────────────────────────────────
                if self.global_step % self.train_cfg.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        print(
                            f"[eval ] step {self.global_step} | "
                            f"loss {eval_metrics.get('eval/total', 0):.4f} | "
                            f"ppl {eval_metrics.get('eval/ppl', 0):.2f}"
                        )
                    if self.wandb and eval_metrics:
                        self.wandb.log(eval_metrics, step=self.global_step)

                # ── Checkpointing ─────────────────────────────────────────
                if self.global_step % self.train_cfg.save_interval == 0:
                    self.ckpt_manager.save(
                        self.global_step, self.model,
                        self.optimizer, self.scaler,
                        {"step": self.global_step}
                    )

        # Final checkpoint
        self.ckpt_manager.save(
            self.global_step, self.model, self.optimizer, self.scaler,
            {"step": self.global_step, "final": True}
        )
        print("[train] Done.")

    # ── RLHF Fine-tuning ──────────────────────────────────────────────────

    def run_rlhf(
        self,
        prompt_loader: DataLoader,
        test_cases_fn,          # callable(batch) → List[List[Dict]]
        ref_model: Optional[CogForge] = None,
    ):
        """
        RLHF loop: sample from policy, score with execution reward,
        update with policy gradient + KL penalty.
        """
        if ref_model is None:
            # Freeze a copy of the current model as reference
            import copy
            ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        rl_trainer = RLHFTrainer(
            self.model, ref_model, self.tokenizer, self.train_cfg
        )
        rl_optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate * 0.1,  # lower LR for RL
            weight_decay=self.train_cfg.weight_decay,
        )

        print("[rlhf] Starting RLHF stage")
        for rl_step, batch in enumerate(prompt_loader):
            if rl_step >= self.train_cfg.rl_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            test_cases = test_cases_fn(batch)

            # Sample from policy
            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=0.9,
                )
            # Only the generated portion
            gen_only = gen_ids[:, input_ids.shape[1]:]

            # Execution reward
            rewards = ExecutionReward.batch_score(
                gen_only, self.tokenizer, test_cases
            ).to(self.device)

            # Policy gradient update
            self.model.train()
            with self._autocast():
                pg_loss = rl_trainer.compute_pg_loss(input_ids, gen_only, rewards)

            rl_optimizer.zero_grad()
            self.scaler.scale(pg_loss).backward()
            self.scaler.unscale_(rl_optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(rl_optimizer)
            self.scaler.update()

            if rl_step % 10 == 0:
                print(
                    f"[rlhf ] step {rl_step:5d} | "
                    f"pg_loss {pg_loss.item():.4f} | "
                    f"avg_reward {rewards.mean().item():.3f}"
                )

        print("[rlhf] RLHF stage complete")
