"""
CogForge — train.py
===================
Entry point for training. Run:

  # Pretraining
  python train.py --stage pretrain --data_dir ./data/code_jsonl

  # SFT
  python train.py --stage sft --data_dir ./data/instruct_jsonl \
                  --resume --checkpoint ./checkpoints

  # RLHF
  python train.py --stage rlhf --data_dir ./data/prompt_jsonl \
                  --resume --checkpoint ./checkpoints
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from model.architecture import CogForgeConfig
from training.trainer import CogForgeTrainer, TrainingConfig
from data.dataset import (
    CodeDataset, CodeTokenizer, CogForgeCollator, make_dataloader,
    BugContrastiveDataset
)


def parse_args():
    p = argparse.ArgumentParser(description="Train CogForge")
    p.add_argument("--stage", choices=["pretrain", "sft", "rlhf"], default="pretrain")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--run_name", type=str, default="cogforge_100m")
    return p.parse_args()


def build_model_config() -> CogForgeConfig:
    """
    ~100M parameter configuration.
    """
    return CogForgeConfig(
        vocab_size=32000,
        max_seq_len=4096,
        d_model=768,
        n_heads=12,
        n_kv_heads=4,
        n_layers=12,
        d_ff_multiplier=2.6667,  # → d_ff = 2048
        window_size=512,
        global_tokens=64,
        n_latent_tokens=8,
        n_act_iterations=3,
        act_threshold=0.99,
        n_arch_layers=2,
        arch_d_model=256,
        arch_n_heads=4,
        max_repo_chunks=8,
        dropout=0.1,
        attn_dropout=0.0,
        rope_theta=10000.0,
    )


def build_train_config(args) -> TrainingConfig:
    return TrainingConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        use_wandb=args.use_wandb,
    )


def make_synthetic_data_if_needed(data_dir: str, tokenizer: CodeTokenizer):
    """
    Generate a tiny synthetic JSONL file for quick smoke-test
    when no real data is present.
    """
    import json, os
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_path):
        return

    samples = [
        {"code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"},
        {"code": "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n"},
        {"code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n"},
        {"code": "import heapq\ndef dijkstra(graph, start):\n    dist = {n: float('inf') for n in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist\n"},
    ] * 50  # repeat for enough tokens

    with open(train_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[data] Wrote synthetic training data to {train_path}")


def main():
    args = parse_args()

    model_cfg = build_model_config()
    train_cfg = build_train_config(args)

    tokenizer = CodeTokenizer(vocab_size=model_cfg.vocab_size)

    # ── Data ──────────────────────────────────────────────────────────────
    make_synthetic_data_if_needed(args.data_dir, tokenizer)

    train_files = [
        str(p) for p in __import__("pathlib").Path(args.data_dir).glob("*.jsonl")
        if "eval" not in p.name
    ]
    eval_files = [
        str(p) for p in __import__("pathlib").Path(args.data_dir).glob("*.jsonl")
        if "eval" in p.name
    ]

    train_dataset = CodeDataset(
        train_files or [str(__import__("pathlib").Path(args.data_dir) / "train.jsonl")],
        tokenizer=tokenizer,
        max_seq_len=train_cfg.max_seq_len,
    )
    collator = CogForgeCollator(pad_token_id=model_cfg.pad_token_id,
                                 max_seq_len=train_cfg.max_seq_len)
    train_loader = make_dataloader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        num_workers=0,  # set to 4+ in production
        collator=collator,
    )

    eval_loader = None
    if eval_files:
        eval_dataset = CodeDataset(eval_files, tokenizer=tokenizer,
                                   max_seq_len=train_cfg.max_seq_len)
        eval_loader = make_dataloader(
            eval_dataset, batch_size=train_cfg.batch_size, collator=collator
        )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = CogForgeTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
    )

    if args.stage in ("pretrain", "sft"):
        trainer.train(resume=args.resume)
    elif args.stage == "rlhf":
        trainer.run_rlhf(
            prompt_loader=train_loader,
            test_cases_fn=lambda b: [[] for _ in range(b["input_ids"].shape[0])],
        )


if __name__ == "__main__":
    main()
