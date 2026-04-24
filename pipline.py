"""
CogForge Data Pipeline
======================
Handles:
  1. CodeDataset          — streaming tokenized code from JSONL files
  2. BugContrastiveDataset— paired (buggy, fixed) code samples
  3. RepoContextDataset   — wraps CodeDataset with RAG chunk retrieval
  4. DataCollator         — batching with dynamic padding + dependency masks
  5. Tokenizer            — BPE wrapper (tiktoken-compatible interface)
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Simple BPE Tokenizer Interface
# ---------------------------------------------------------------------------

class CodeTokenizer:
    """
    Thin wrapper around a BPE vocabulary.
    For production: swap in tiktoken (cl100k_base) or SentencePiece.
    Here we implement a character-level fallback so the code runs
    without external dependencies.
    """

    # Special tokens
    PAD   = 0
    BOS   = 1
    EOS   = 2
    UNK   = 3
    THINK = 4   # latent reasoning boundary

    def __init__(self, vocab_size: int = 32000, model_path: Optional[str] = None):
        self.vocab_size = vocab_size

        if model_path and Path(model_path).exists():
            self._load(model_path)
        else:
            # Minimal character-level vocabulary for standalone use
            self._build_char_vocab()

    def _build_char_vocab(self):
        """Build a 256-char + special-token vocab for zero-dependency usage."""
        specials = ["<pad>", "<bos>", "<eos>", "<unk>", "<think>"]
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        # Common code tokens
        keywords = [
            "def ", "class ", "return ", "import ", "from ", "if ", "else:",
            "elif ", "for ", "while ", "with ", "try:", "except ", "raise ",
            "lambda ", "yield ", "async ", "await ", "None", "True", "False",
            "self.", "    ", "        ",  # 4/8-space indents
            "->", "**", "//", "!=", "==", "<=", ">=", ":=",
        ]
        vocab = specials + keywords + chars
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Greedy longest-match tokenization."""
        ids = []
        if add_special_tokens:
            ids.append(self.BOS)
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(16, len(text) - i), 0, -1):
                sub = text[i:i + length]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                ids.append(self.UNK)
                i += 1
        if add_special_tokens:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        specials = {self.PAD, self.BOS, self.EOS, self.UNK, self.THINK}
        out = []
        for i in ids:
            if skip_special and i in specials:
                continue
            out.append(self.id_to_token.get(i, "?"))
        return "".join(out)

    def __len__(self) -> int:
        return len(self.token_to_id)


# ---------------------------------------------------------------------------
# 1. Streaming Code Dataset (JSONL)
# ---------------------------------------------------------------------------

class CodeDataset(IterableDataset):
    """
    Streams code samples from a JSONL file where each line is:
    {"code": "...", "language": "python", "repo": "user/repo", "path": "foo.py"}

    Yields tokenized windows of max_seq_len tokens with stride.
    Suitable for large-scale pretraining on code corpora.
    """

    def __init__(
        self,
        file_paths: List[str],
        tokenizer: CodeTokenizer,
        max_seq_len: int = 4096,
        stride: int = 2048,
        shuffle_files: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.shuffle_files = shuffle_files
        self.world_size = world_size
        self.rank = rank

    def _iter_file(self, path: Path) -> Iterator[Dict]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _tokenize_and_chunk(self, text: str) -> Iterator[List[int]]:
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        for start in range(0, max(1, len(ids) - self.max_seq_len + 1), self.stride):
            chunk = ids[start: start + self.max_seq_len]
            if len(chunk) < 16:  # skip tiny chunks
                continue
            yield chunk

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        paths = list(self.file_paths)
        if self.shuffle_files:
            random.shuffle(paths)

        # Distributed shard assignment
        paths = [p for i, p in enumerate(paths) if i % self.world_size == self.rank]

        for path in paths:
            for sample in self._iter_file(path):
                code = sample.get("code", sample.get("content", ""))
                if not code or len(code) < 32:
                    continue
                for chunk in self._tokenize_and_chunk(code):
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = input_ids.clone()
                    labels[:-1] = input_ids[1:]    # shift left for next-token pred
                    labels[-1] = -100              # last has no target
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "length": len(chunk),
                    }


# ---------------------------------------------------------------------------
# 2. Buggy/Fixed Contrastive Dataset
# ---------------------------------------------------------------------------

class BugContrastiveDataset(Dataset):
    """
    Loads pairs of (buggy_code, fixed_code) from a JSONL file:
    {"buggy": "...", "fixed": "...", "bug_type": "off_by_one"}

    Used for contrastive loss training.
    """

    def __init__(self, file_path: str, tokenizer: CodeTokenizer,
                 max_seq_len: int = 1024):
        self.pairs: List[Tuple[List[int], List[int]]] = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                buggy_ids = tokenizer.encode(sample["buggy"])[:max_seq_len]
                fixed_ids = tokenizer.encode(sample["fixed"])[:max_seq_len]
                self.pairs.append((buggy_ids, fixed_ids))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        buggy_ids, fixed_ids = self.pairs[idx]
        return {
            "buggy_ids":  torch.tensor(buggy_ids, dtype=torch.long),
            "fixed_ids":  torch.tensor(fixed_ids, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3. Repo-Level Context Dataset (RAG-augmented)
# ---------------------------------------------------------------------------

class RepoContextDataset(IterableDataset):
    """
    Wraps CodeDataset with a simple in-memory chunk index.
    For each sample, retrieves up to `n_chunks` related code snippets
    from the same repo based on token overlap (BM25-lite).

    Production would replace this with a vector DB (FAISS/pgvector).
    """

    def __init__(
        self,
        base_dataset: CodeDataset,
        chunk_store: Dict[str, List[List[int]]],  # repo → list of chunk token lists
        n_chunks: int = 8,
        chunk_len: int = 128,
        d_model: int = 768,
    ):
        self.base = base_dataset
        self.chunk_store = chunk_store
        self.n_chunks = n_chunks
        self.chunk_len = chunk_len
        self.d_model = d_model

    def _retrieve(self, query_ids: List[int], repo: str) -> torch.Tensor:
        """
        Retrieve n_chunks most relevant chunks from the repo's chunk store.
        Uses simple bag-of-tokens overlap score.
        Returns a padded tensor of shape (n_chunks, chunk_len).
        """
        candidates = self.chunk_store.get(repo, [])
        if not candidates:
            return torch.zeros(self.n_chunks, self.chunk_len, dtype=torch.long)

        query_set = set(query_ids)
        scored = []
        for chunk in candidates:
            overlap = len(query_set & set(chunk))
            scored.append((overlap, chunk))
        scored.sort(key=lambda x: -x[0])
        top = [c for _, c in scored[:self.n_chunks]]

        # Pad to (n_chunks, chunk_len)
        result = torch.zeros(self.n_chunks, self.chunk_len, dtype=torch.long)
        for i, chunk in enumerate(top):
            length = min(len(chunk), self.chunk_len)
            result[i, :length] = torch.tensor(chunk[:length], dtype=torch.long)
        return result

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for sample in self.base:
            # In real usage, we'd track which repo each sample came from.
            # Here we yield a dummy chunk context.
            repo_chunks = torch.zeros(
                self.n_chunks, self.chunk_len, dtype=torch.long
            )
            sample["repo_chunks"] = repo_chunks
            sample["repo_mask"] = (repo_chunks.sum(-1) == 0)  # True=pad
            yield sample


# ---------------------------------------------------------------------------
# 4. Data Collator
# ---------------------------------------------------------------------------

class CogForgeCollator:
    """
    Collates variable-length samples into padded batches.
    Handles:
      - Dynamic padding to longest sequence in batch
      - Label masking for padding positions
      - Chunk context stacking
    """

    def __init__(self, pad_token_id: int = 0, max_seq_len: int = 4096):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # ── Pad input_ids and labels ──────────────────────────────────────
        max_len = min(max(s["input_ids"].shape[0] for s in samples), self.max_seq_len)

        input_ids_list, labels_list, attn_masks = [], [], []
        for s in samples:
            ids = s["input_ids"][:max_len]
            lbl = s["labels"][:max_len]
            pad_len = max_len - ids.shape[0]

            ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100)])
            mask = torch.cat([
                torch.ones(max_len - pad_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])

            input_ids_list.append(ids)
            labels_list.append(lbl)
            attn_masks.append(mask)

        batch = {
            "input_ids":      torch.stack(input_ids_list),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(attn_masks),
        }

        # ── Repo chunks (if present) ──────────────────────────────────────
        if "repo_chunks" in samples[0]:
            batch["repo_chunks"] = torch.stack([s["repo_chunks"] for s in samples])
            batch["repo_mask"]   = torch.stack([s["repo_mask"]   for s in samples])

        # ── Buggy/fixed pairs (if present) ───────────────────────────────
        if "buggy_ids" in samples[0]:
            bug_max = max(s["buggy_ids"].shape[0] for s in samples)
            fix_max = max(s["fixed_ids"].shape[0] for s in samples)
            buggy_list, fixed_list = [], []
            buggy_masks, fixed_masks = [], []

            for s in samples:
                for key, lst, msks, mlen in [
                    ("buggy_ids", buggy_list, buggy_masks, bug_max),
                    ("fixed_ids", fixed_list, fixed_masks, fix_max),
                ]:
                    t = s[key]
                    pad = mlen - t.shape[0]
                    lst.append(torch.cat([t, torch.zeros(pad, dtype=torch.long)]))
                    msks.append(torch.cat([
                        torch.ones(t.shape[0]), torch.zeros(pad)
                    ]).long())

            batch["buggy_ids"]   = torch.stack(buggy_list)
            batch["buggy_mask"]  = torch.stack(buggy_masks)
            batch["fixed_ids"]   = torch.stack(fixed_list)
            batch["fixed_mask"]  = torch.stack(fixed_masks)

        return batch


# ---------------------------------------------------------------------------
# 5. DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    collator: Optional[CogForgeCollator] = None,
    **kwargs,
) -> DataLoader:
    if collator is None:
        collator = CogForgeCollator()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
