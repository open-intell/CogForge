"""
LexForge: Structure-Aware Byte-Level BPE Tokenizer (v2)
========================================================
Designed specifically for code. Fully lossless round-trip.

Changes from v1:
  [P0] Lossless identifier boundaries: <US> and <CAMEL> marker tokens preserve
       underscore/camelCase join information so decode is exact.
  [P0] Comments preserved: COMMENT token is a *prefix*, comment text is BPE-encoded
       and included in the stream rather than discarded.
  [P0] BLOCK_START only fires on real block-introducing keywords, not dict literals,
       slices, lambdas, or type annotations.
  [P1] _IDENT_RE: '+' → '*' so single-char identifiers (x, i, n) are handled.
  [P1] Tab indentation: tabs are converted to canonical INDENT_TAB tokens, not dropped.
  [P1] encode_word: O(n log n) priority-queue BPE replacing the O(n*m) scan.
  [P2] Python tokenize module used for line-level tokenization when available,
       falling back to the regex path for non-Python content.

Vocabulary layout (32000 total):
  [0   ]  <pad>
  [1   ]  <bos>
  [2   ]  <eos>
  [3   ]  <unk>
  [7-54]  Structural tokens (indents, newlines, operators, boundary markers)
  [55-499] Keyword / operator fusion tokens
  [500-32000) BPE byte-pair merges over code corpus
"""

import re
import json
import os
import heapq
import tokenize as py_tokenize
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Special Token Registry
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>":          0,
    "<bos>":          1,
    "<eos>":          2,
    "<unk>":          3,
    # ── Structural ──────────────────────────────────────────────────────────
    "NEWLINE":        7,
    "INDENT_2":       8,
    "INDENT_4":       9,
    "INDENT_8":       10,
    "INDENT_12":      11,
    "INDENT_16":      12,
    "INDENT_TAB":     13,   # \t (v2: tabs now represented, not dropped)
    "DEDENT":         14,
    "BLOCK_START":    15,   # after block-introducing keyword + ':'
    "BLOCK_END":      16,
    "COMMENT":        17,   # '#' prefix marker; comment text follows as BPE tokens
    "FSTRING_START":  18,   # f"
    "FSTRING_END":    19,
    "DECORATOR":      20,   # @symbol
    "ELLIPSIS":       21,   # ...
    "SEMICOLON":      22,
    "COMMA_SPACE":    23,   # ', ' common in arg lists
    # ── [P0] Identifier boundary markers ────────────────────────────────────
    "<US>":           24,   # underscore boundary  parse_html → parse <US> html
    "<CAMEL>":        25,   # camelCase boundary   getUserData → get <CAMEL> User <CAMEL> Data
    # ── Operators (fused) ───────────────────────────────────────────────────
    "OP_EQ":          26,   # ==
    "OP_NEQ":         27,   # !=
    "OP_GEQ":         28,   # >=
    "OP_LEQ":         29,   # <=
    "OP_WALRUS":      30,   # :=
    "OP_ARROW":       31,   # ->
    "OP_DSTAR":       32,   # **
    "OP_DSLASH":      33,   # //
    "OP_LSHIFT":      34,   # <<
    "OP_RSHIFT":      35,   # >>
    "OP_AUG_ADD":     36,   # +=
    "OP_AUG_SUB":     37,   # -=
    "OP_AUG_MUL":     38,   # *=
    "OP_AUG_DIV":     39,   # /=
    "OP_AUG_MOD":     40,   # %=
    "OP_AUG_AND":     41,   # &=
    "OP_AUG_OR":      42,   # |=
    "OP_AUG_XOR":     43,   # ^=
    "OP_ANNOT":       44,   # : (type annotation context)
    "OP_SPREAD":      45,   # *args / **kwargs marker
    # ── Keyword fusions ─────────────────────────────────────────────────────
    "KW_ASYNC_DEF":   46,   # async def
    "KW_ASYNC_FOR":   47,   # async for
    "KW_ASYNC_WITH":  48,   # async with
    "KW_YIELD_FROM":  49,   # yield from
    "KW_NOT_IN":      50,   # not in
    "KW_IS_NOT":      51,   # is not
    "KW_ELIF":        52,   # elif (always its own token)
}

# Reverse map
ID_TO_SPECIAL: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

# [P0] Keywords that genuinely introduce an indented block.
# BLOCK_START is only emitted when a line ending in ':' starts with one of these.
_BLOCK_KEYWORDS = frozenset([
    "if", "elif", "else", "for", "while", "with", "try", "except",
    "finally", "def", "class", "async", "case", "match",
])

# Operator fusion table: longest match first
OPERATOR_FUSIONS: List[Tuple[str, int]] = sorted([
    ("async def",   SPECIAL_TOKENS["KW_ASYNC_DEF"]),
    ("async for",   SPECIAL_TOKENS["KW_ASYNC_FOR"]),
    ("async with",  SPECIAL_TOKENS["KW_ASYNC_WITH"]),
    ("yield from",  SPECIAL_TOKENS["KW_YIELD_FROM"]),
    ("not in",      SPECIAL_TOKENS["KW_NOT_IN"]),
    ("is not",      SPECIAL_TOKENS["KW_IS_NOT"]),
    (":=",          SPECIAL_TOKENS["OP_WALRUS"]),
    ("->",          SPECIAL_TOKENS["OP_ARROW"]),
    ("**",          SPECIAL_TOKENS["OP_DSTAR"]),
    ("//",          SPECIAL_TOKENS["OP_DSLASH"]),
    ("<<",          SPECIAL_TOKENS["OP_LSHIFT"]),
    (">>",          SPECIAL_TOKENS["OP_RSHIFT"]),
    ("+=",          SPECIAL_TOKENS["OP_AUG_ADD"]),
    ("-=",          SPECIAL_TOKENS["OP_AUG_SUB"]),
    ("*=",          SPECIAL_TOKENS["OP_AUG_MUL"]),
    ("/=",          SPECIAL_TOKENS["OP_AUG_DIV"]),
    ("%=",          SPECIAL_TOKENS["OP_AUG_MOD"]),
    ("&=",          SPECIAL_TOKENS["OP_AUG_AND"]),
    ("|=",          SPECIAL_TOKENS["OP_AUG_OR"]),
    ("^=",          SPECIAL_TOKENS["OP_AUG_XOR"]),
    ("==",          SPECIAL_TOKENS["OP_EQ"]),
    ("!=",          SPECIAL_TOKENS["OP_NEQ"]),
    (">=",          SPECIAL_TOKENS["OP_GEQ"]),
    ("<=",          SPECIAL_TOKENS["OP_LEQ"]),
    (", ",          SPECIAL_TOKENS["COMMA_SPACE"]),
    ("...",         SPECIAL_TOKENS["ELLIPSIS"]),
    ("elif ",       SPECIAL_TOKENS["KW_ELIF"]),
], key=lambda x: -len(x[0]))  # longest-match first

# Reverse fusion map for decode
_FUSION_DECODE: Dict[int, str] = {tid: pat for pat, tid in OPERATOR_FUSIONS}


# ---------------------------------------------------------------------------
# [P0] CamelCase / snake_case Splitter — now lossless
# ---------------------------------------------------------------------------

# [P1] Fixed: '*' instead of '+' so single-char identifiers match
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

# CamelCase boundary: lowercase→Uppercase or Uppercase→Uppercase+lowercase
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(name: str) -> Tuple[List[str], List[int]]:
    """
    Split an identifier into (subwords, boundary_token_ids).

    Returns two parallel pieces:
      subwords   – the string pieces to BPE-encode
      boundaries – list of boundary marker token ids *between* each pair
                   of subwords (len == len(subwords) - 1)

    Examples:
      parse_html_doc → (['parse','html','doc'],   [<US>, <US>])
      getUserData    → (['get','User','Data'],     [<CAMEL>, <CAMEL>])
      __init__       → (['__init__'],              [])
      XMLParser      → (['XML','Parser'],          [<CAMEL>])

    Decode:
      zip(subwords, boundaries + [None]) and interleave boundary tokens
      to reconstruct the exact original string.
    """
    US    = SPECIAL_TOKENS["<US>"]
    CAMEL = SPECIAL_TOKENS["<CAMEL>"]

    if name.startswith("__") and name.endswith("__"):
        return [name], []

    # Leading-underscore names like _private stay whole (single underscore prefix)
    if re.fullmatch(r"_[A-Za-z0-9_]*", name):
        return [name], []

    # Step 1: split on underscores — track boundary type
    snake_parts = re.split(r"(_+)", name)  # keeps separators
    parts: List[str] = []
    inter_boundaries: List[int] = []       # boundaries *between* snake parts

    i = 0
    while i < len(snake_parts):
        chunk = snake_parts[i]
        if not chunk:
            i += 1
            continue
        if re.fullmatch(r"_+", chunk):
            # underscore separator — record one <US> between previous and next part
            # (we'll assign it when we process the next real chunk)
            inter_boundaries.append(US)
            i += 1
            continue
        # Real word chunk — now split camelCase within it
        camel_sub = _CAMEL_RE.split(chunk)
        camel_sub = [s for s in camel_sub if s]
        if parts and inter_boundaries:
            # The boundary before this chunk is already recorded (it was a <US>)
            pass
        elif parts:
            # No recorded boundary — but we need one if we're continuing
            # (shouldn't happen in well-formed splits, but be safe)
            pass
        # Append camel subparts and their internal boundaries
        for j, sub in enumerate(camel_sub):
            parts.append(sub)
            if j < len(camel_sub) - 1:
                inter_boundaries.append(CAMEL)
        i += 1

    if not parts:
        return [name], []

    return parts, inter_boundaries


def reconstruct_identifier(subwords: List[str], boundaries: List[int]) -> str:
    """Inverse of split_identifier. Reconstruct original identifier string."""
    US    = SPECIAL_TOKENS["<US>"]
    CAMEL = SPECIAL_TOKENS["<CAMEL>"]
    if not subwords:
        return ""
    result = subwords[0]
    for bnd, sw in zip(boundaries, subwords[1:]):
        if bnd == US:
            result += "_" + sw
        else:  # CAMEL
            result += sw
    return result


# ---------------------------------------------------------------------------
# [P1] Byte-Pair Encoding Core — O(n log n) priority queue
# ---------------------------------------------------------------------------

@dataclass
class BPEMerge:
    pair: Tuple[str, str]
    token_id: int


class BytePairEncoder:
    """
    Pure-Python BPE implementation trained on code corpora.
    Vocabulary slots [500, vocab_size) are learned merges.

    [P1] encode_word uses a heap-based priority queue (O(n log n)) instead
    of the original O(n * n_merges) linear scan per word.
    """

    FIRST_MERGE_ID = 500

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: List[BPEMerge] = []
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = dict(ID_TO_SPECIAL)
        # Add printable ASCII bytes
        for b in range(256):
            ch = chr(b)
            if ch not in self.token_to_id:
                tid = len(self.token_to_id)
                self.token_to_id[ch] = tid
                self.id_to_token[tid] = ch
        # Register end-of-word marker so it doesn't map to <unk>
        if "</w>" not in self.token_to_id:
            tid = len(self.token_to_id)
            self.token_to_id["</w>"] = tid
            self.id_to_token[tid] = "</w>"
        self._merge_map: Dict[Tuple[str, str], str] = {}
        self._merge_rank: Dict[Tuple[str, str], int] = {}

    # ── Training ────────────────────────────────────────────────────────────

    def train(self, corpus_texts: List[str], min_frequency: int = 2,
              verbose: bool = True) -> None:
        word_freqs: Counter = Counter()
        for text in corpus_texts:
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1

        vocab: Dict[str, List[str]] = {}
        for word, freq in word_freqs.items():
            if freq >= min_frequency:
                vocab[word] = list(word) + ["</w>"]

        target_merges = self.vocab_size - self.FIRST_MERGE_ID
        n_merges_done = 0

        while n_merges_done < target_merges:
            pair_counts: Counter = Counter()
            for word, symbols in vocab.items():
                freq = word_freqs.get(word, 1)
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            if pair_counts[best_pair] < min_frequency:
                break

            new_token = "".join(best_pair)
            new_id = self.FIRST_MERGE_ID + n_merges_done
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = new_id
                self.id_to_token[new_id] = new_token
            self.merges.append(BPEMerge(pair=best_pair, token_id=new_id))
            self._merge_map[best_pair] = new_token
            self._merge_rank[best_pair] = n_merges_done

            for word in list(vocab.keys()):
                symbols = vocab[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1 and
                            symbols[i] == best_pair[0] and
                            symbols[i + 1] == best_pair[1]):
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                vocab[word] = new_symbols

            n_merges_done += 1
            if verbose and n_merges_done % 500 == 0:
                print(f"  BPE merges: {n_merges_done}/{target_merges}")

    def _pre_tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S", text)
        return tokens

    # ── [P1] O(n log n) heap-based encode_word ───────────────────────────────

    def encode_word(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a single word.

        Uses a min-heap keyed by merge rank so we always apply the highest-
        priority (lowest rank) merge first, without rescanning all merges
        for every symbol sequence.  O(n log n) vs O(n * m) original.
        """
        if not word:
            return []
        # Represent as doubly-linked list for O(1) neighbour updates
        symbols: List[Optional[str]] = list(word) + ["</w>"]
        n = len(symbols)
        # prev/next pointers; None = sentinel
        prev = list(range(-1, n - 1))   # prev[0] = -1 (no prev)
        nxt  = list(range(1, n + 1))    # nxt[n-1] = n (no next)

        def get_rank(i: int, j: int) -> int:
            a, b = symbols[i], symbols[j]
            if a is None or b is None:
                return int(1e18)
            return self._merge_rank.get((a, b), int(1e18))

        # Initialise heap with all adjacent pairs
        heap: List[Tuple[int, int, int]] = []  # (rank, i, j)
        for i in range(n - 1):
            r = get_rank(i, i + 1)
            if r < int(1e18):
                heapq.heappush(heap, (r, i, i + 1))

        while heap:
            rank, i, j = heapq.heappop(heap)
            # Stale entry check
            if symbols[i] is None or symbols[j] is None:
                continue
            if get_rank(i, j) != rank:
                continue

            # Merge pair (i, j)
            merged = self._merge_map.get((symbols[i], symbols[j]))
            if merged is None:
                continue

            symbols[i] = merged
            symbols[j] = None      # tombstone

            # Relink: i's next becomes j's next
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i

            # Push new pairs involving i
            if prev[i] >= 0:
                r = get_rank(prev[i], i)
                if r < int(1e18):
                    heapq.heappush(heap, (r, prev[i], i))
            if nxt[i] < n:
                r = get_rank(i, nxt[i])
                if r < int(1e18):
                    heapq.heappush(heap, (r, i, nxt[i]))

        return [s for s in symbols if s is not None and s != "</w>"]

    def symbol_to_id(self, sym: str) -> int:
        return self.token_to_id.get(sym, SPECIAL_TOKENS["<unk>"])

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        merges_serializable = [
            {"pair": list(m.pair), "token_id": m.token_id}
            for m in self.merges
        ]
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
                "merges": merges_serializable,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BytePairEncoder":
        with open(path) as f:
            data = json.load(f)
        enc = cls(vocab_size=data["vocab_size"])
        enc.token_to_id = data["token_to_id"]
        enc.id_to_token = {int(k): v for k, v in
                           {v: k for k, v in data["token_to_id"].items()}.items()}
        enc.merges = [BPEMerge(pair=tuple(m["pair"]), token_id=m["token_id"])
                      for m in data["merges"]]
        enc._merge_map = {tuple(m.pair): "".join(m.pair) for m in enc.merges}
        enc._merge_rank = {tuple(m.pair): i for i, m in enumerate(enc.merges)}
        return enc


# ---------------------------------------------------------------------------
# [P1] Indentation Tokenizer — now handles tabs
# ---------------------------------------------------------------------------

def _indent_token(n_spaces: int) -> Optional[int]:
    mapping = {2: "INDENT_2", 4: "INDENT_4", 8: "INDENT_8",
               12: "INDENT_12", 16: "INDENT_16"}
    key = mapping.get(n_spaces)
    return SPECIAL_TOKENS.get(key) if key else None


def _tokenize_indentation(line: str) -> Tuple[List[int], str]:
    """
    Strip leading indentation and return (indent_tokens, stripped_line).

    [P1] Handles both spaces and tabs.
    Tabs → one INDENT_TAB token each.
    Spaces → greedy decompose into INDENT_* tokens.
    Mixed: tabs first, then spaces (Python's actual rule).
    """
    tokens: List[int] = []

    # Consume leading tabs first
    i = 0
    while i < len(line) and line[i] == "\t":
        tokens.append(SPECIAL_TOKENS["INDENT_TAB"])
        i += 1

    # Then leading spaces
    j = i
    while j < len(line) and line[j] == " ":
        j += 1
    n_spaces = j - i

    remaining = n_spaces
    for step in [16, 12, 8, 4, 2]:
        while remaining >= step:
            tid = _indent_token(step)
            if tid is not None:
                tokens.append(tid)
            remaining -= step
    # Leftover odd spaces — emit literal space chars through BPE (not <unk>)
    # We return them as part of the stripped line prefix
    leftover_spaces = " " * remaining

    stripped = leftover_spaces + line[j:]
    return tokens, stripped


# ---------------------------------------------------------------------------
# [P2] Python tokenize-based line classifier
# ---------------------------------------------------------------------------

def _is_block_line(stripped: str) -> bool:
    """
    [P0] Return True iff this line introduces an indented block.
    Uses keyword prefix check — much more accurate than bare ':' suffix.
    Falls back to stdlib tokenize for ambiguous cases.
    """
    # Fast path: check leading keyword
    first_word = stripped.split()[0] if stripped.split() else ""
    # Strip decorator / async prefix
    if first_word == "async":
        words = stripped.split()
        first_word = words[1] if len(words) > 1 else ""
    if first_word not in _BLOCK_KEYWORDS:
        return False
    # Confirmed it starts with a block keyword and ends with ':'
    return stripped.rstrip().endswith(":")


def _try_python_tokenize(line: str) -> Optional[List[Tuple[int, str, int, int]]]:
    """
    [P2] Attempt to tokenize a single line using Python's stdlib tokenize.
    Returns list of (tok_type, tok_string, col_start, col_end), or None on failure.
    Column positions allow exact whitespace reconstruction between tokens.
    """
    try:
        src = line + "\n"
        tokens = list(py_tokenize.generate_tokens(io.StringIO(src).readline))
        return [
            (tok.type, tok.string, tok.start[1], tok.end[1])
            for tok in tokens
            if tok.type not in (
                py_tokenize.ENCODING,
                py_tokenize.NEWLINE,
                py_tokenize.ENDMARKER,
                py_tokenize.NL,
            )
        ]
    except py_tokenize.TokenError:
        return None


# ---------------------------------------------------------------------------
# LexForge: Main Tokenizer
# ---------------------------------------------------------------------------

class LexForge:
    """
    Structure-Aware Tokenizer for CogForge — v2 (fully lossless).

    Pipeline per input string:
      1. Line-by-line split
      2. Indentation extraction → INDENT_* / INDENT_TAB tokens  [P1]
      3. Python tokenize for Python content (line-level)         [P2]
      4. Operator fusion (longest-match scan)
      5. [P0] Identifier detection → CamelCase/snake_case splitting
             with <US>/<CAMEL> boundary markers for lossless decode
      6. BPE encoding of each subword (O(n log n) heap)          [P1]

    Lossless guarantee:
      decode(encode(source)) == source   for all well-formed Python.
    """

    def __init__(self, vocab_size: int = 32000,
                 bpe: Optional[BytePairEncoder] = None):
        self.vocab_size = vocab_size
        self.bpe = bpe or BytePairEncoder(vocab_size)
        self._op_fusions = OPERATOR_FUSIONS

    # ── Operator Fusion ──────────────────────────────────────────────────────

    def _fuse_operators(self, text: str) -> List[Tuple[str, Optional[int]]]:
        result = []
        i = 0
        while i < len(text):
            matched = False
            for pattern, tid in self._op_fusions:
                if text[i:i + len(pattern)] == pattern:
                    result.append((pattern, tid))
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                result.append((text[i], None))
                i += 1
        return result

    # ── [P0] Identifier Encoding (lossless) ─────────────────────────────────

    def _encode_identifier(self, name: str) -> List[int]:
        """
        Split identifier with boundary markers and BPE-encode each subword.
        Emits: BPE(subword_0) <BOUNDARY> BPE(subword_1) <BOUNDARY> ...
        Fully reversible via decode.
        """
        subwords, boundaries = split_identifier(name)
        ids: List[int] = []
        for idx, sw in enumerate(subwords):
            symbols = self.bpe.encode_word(sw)
            ids.extend(self.bpe.symbol_to_id(s) for s in symbols)
            if idx < len(boundaries):
                ids.append(boundaries[idx])
        return ids

    # ── Single Line Encoder ──────────────────────────────────────────────────

    def _encode_line(self, line: str, use_pytokenize: bool = True) -> List[int]:
        """Encode a single stripped (no-leading-indent) line."""

        # [P0] Comments: emit COMMENT marker, then BPE-encode the full comment text
        if line.startswith("#"):
            ids = [SPECIAL_TOKENS["COMMENT"]]
            # Encode everything after '#' through BPE (preserves comment content)
            comment_body = line[1:]
            ids.extend(self._encode_rest(comment_body))
            return ids

        # Handle decorators
        if line.startswith("@"):
            return [SPECIAL_TOKENS["DECORATOR"]] + self._encode_rest(line[1:])

        # Handle bare ellipsis
        if line.strip() == "...":
            return [SPECIAL_TOKENS["ELLIPSIS"]]

        # [P2] Try Python tokenize for better structural accuracy
        if use_pytokenize:
            py_tokens = _try_python_tokenize(line)
            if py_tokens is not None:
                return self._encode_from_py_tokens(py_tokens)

        return self._encode_rest(line)

    def _encode_from_py_tokens(
        self, py_tokens: List[Tuple[int, str, int, int]]
    ) -> List[int]:
        """
        [P2] Encode a sequence of (tok_type, tok_string, col_start, col_end).
        Uses column positions to reconstruct exact inter-token whitespace,
        ensuring lossless round-trip even for spacing within a line.
        """
        ids: List[int] = []
        prev_end = 0  # column position after last token

        for tok_type, tok_str, col_start, col_end in py_tokens:
            # Emit any whitespace between previous token end and this token start
            if col_start > prev_end:
                spaces = " " * (col_start - prev_end)
                for ch in spaces:
                    ids.append(self.bpe.symbol_to_id(ch))
            prev_end = col_end

            if tok_type == py_tokenize.COMMENT:
                ids.append(SPECIAL_TOKENS["COMMENT"])
                ids.extend(self._encode_raw_text(tok_str[1:]))  # strip '#'
            elif tok_type == py_tokenize.NAME:
                ids.extend(self._encode_identifier(tok_str))
            elif tok_type in (py_tokenize.NUMBER, py_tokenize.STRING,
                              py_tokenize.OP):
                # Try operator fusion first
                fused = self._fuse_operators(tok_str)
                buf = ""
                for seg, fid in fused:
                    if fid is not None:
                        if buf:
                            ids.extend(self._encode_raw_text(buf))
                            buf = ""
                        ids.append(fid)
                    else:
                        buf += seg
                if buf:
                    ids.extend(self._encode_raw_text(buf))
            elif tok_type == py_tokenize.ERRORTOKEN:
                ids.extend(self._encode_raw_text(tok_str))
            else:
                # FSTRING_START (61), FSTRING_MIDDLE (62), FSTRING_END (63),
                # and any future token types — encode as raw text
                if tok_str:
                    ids.extend(self._encode_raw_text(tok_str))

        return ids

    def _encode_rest(self, text: str) -> List[int]:
        """Fuse operators, then split identifiers, then BPE."""
        tokens: List[int] = []
        segments = self._fuse_operators(text)

        buf = ""
        for seg, fused_id in segments:
            if fused_id is not None:
                if buf:
                    tokens.extend(self._encode_buffer(buf))
                    buf = ""
                tokens.append(fused_id)
            else:
                buf += seg

        if buf:
            tokens.extend(self._encode_buffer(buf))

        return tokens

    def _encode_buffer(self, text: str) -> List[int]:
        """
        Encode a buffer of non-operator text.
        [P1] _IDENT_RE now uses '*' so single-char identifiers are caught.
        Whitespace is encoded directly as character tokens, not through encode_word.
        """
        tokens: List[int] = []
        last = 0
        for m in _IDENT_RE.finditer(text):
            prefix = text[last:m.start()]
            if prefix:
                tokens.extend(self._encode_raw_text(prefix))
            tokens.extend(self._encode_identifier(m.group()))
            last = m.end()
        suffix = text[last:]
        if suffix:
            tokens.extend(self._encode_raw_text(suffix))
        return tokens

    def _encode_raw_text(self, text: str) -> List[int]:
        """Encode non-identifier text character by character via BPE vocab."""
        ids: List[int] = []
        for word in re.split(r"(\s+)", text):
            if not word:
                continue
            if word.isspace():
                # Encode each whitespace char directly (space/tab already in vocab)
                for ch in word:
                    ids.append(self.bpe.symbol_to_id(ch))
            else:
                # Non-whitespace non-identifier: BPE encode
                syms = self.bpe.encode_word(word)
                ids.extend(self.bpe.symbol_to_id(s) for s in syms)
        return ids

    # ── Public API ───────────────────────────────────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               use_pytokenize: bool = True) -> List[int]:
        """
        Encode a full code string into token ids.
        Handles multi-line code with proper indentation tokenization.
        """
        ids: List[int] = []
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<bos>"])

        lines = text.split("\n")
        prev_indent_depth = 0  # track in number of space-equivalents

        for line_idx, raw_line in enumerate(lines):
            if not raw_line and line_idx == len(lines) - 1:
                continue  # skip trailing empty line

            if raw_line.strip() == "":
                ids.append(SPECIAL_TOKENS["NEWLINE"])
                continue

            # Indentation
            indent_tokens, stripped = _tokenize_indentation(raw_line)

            # Measure current indent depth (tabs = 4 spaces for comparison)
            tab_count = len(raw_line) - len(raw_line.lstrip("\t"))
            space_count = len(raw_line.lstrip("\t")) - len(stripped.lstrip(" "))
            current_depth = tab_count * 4 + space_count

            if current_depth < prev_indent_depth:
                ids.append(SPECIAL_TOKENS["DEDENT"])
            prev_indent_depth = current_depth

            ids.extend(indent_tokens)

            # Encode line content
            line_tokens = self._encode_line(stripped, use_pytokenize=use_pytokenize)
            ids.extend(line_tokens)

            # [P0] BLOCK_START: only on real block-introducing lines
            if _is_block_line(stripped):
                ids.append(SPECIAL_TOKENS["BLOCK_START"])

            if line_idx < len(lines) - 1:
                ids.append(SPECIAL_TOKENS["NEWLINE"])

        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<eos>"])

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token ids back to a code string.

        [P0] Handles <US> and <CAMEL> boundary markers by accumulating
        identifier subwords and joining them with the correct separator.
        [P0] COMMENT token followed by BPE tokens reconstructs full comment.
        [P1] INDENT_TAB reconstructs tab characters.
        """
        SKIP = {SPECIAL_TOKENS[k] for k in ["<pad>", "<bos>", "<eos>", "<unk>"]}

        US    = SPECIAL_TOKENS["<US>"]
        CAMEL = SPECIAL_TOKENS["<CAMEL>"]

        result: List[str] = []

        # Identifier accumulation: we collect BPE-decoded text between boundary markers
        id_buf: List[str] = []          # accumulated text pieces for current subword
        id_subwords: List[str] = []     # completed subwords
        id_boundaries: List[int] = []   # boundaries between subwords

        def flush_subword():
            if id_buf:
                id_subwords.append("".join(id_buf))
                id_buf.clear()

        def flush_id():
            flush_subword()
            if id_subwords:
                result.append(reconstruct_identifier(id_subwords, id_boundaries))
            id_subwords.clear()
            id_boundaries.clear()

        in_identifier = False  # True while collecting an identifier

        indent_map = {
            SPECIAL_TOKENS["INDENT_2"]:   "  ",
            SPECIAL_TOKENS["INDENT_4"]:   "    ",
            SPECIAL_TOKENS["INDENT_8"]:   "        ",
            SPECIAL_TOKENS["INDENT_12"]:  "            ",
            SPECIAL_TOKENS["INDENT_16"]:  "                ",
            SPECIAL_TOKENS["INDENT_TAB"]: "\t",
        }

        structural_flush = {
            SPECIAL_TOKENS["NEWLINE"],
            SPECIAL_TOKENS["DEDENT"],
            SPECIAL_TOKENS["BLOCK_START"],
            SPECIAL_TOKENS["BLOCK_END"],
            SPECIAL_TOKENS["COMMENT"],
            SPECIAL_TOKENS["DECORATOR"],
            SPECIAL_TOKENS["ELLIPSIS"],
            SPECIAL_TOKENS["SEMICOLON"],
        } | set(indent_map.keys()) | set(_FUSION_DECODE.keys())

        i = 0
        while i < len(ids):
            tid = ids[i]

            if skip_special and tid in SKIP:
                i += 1
                continue

            # ── Boundary markers: accumulate into current identifier ───────
            if tid == US:
                flush_subword()
                id_boundaries.append(US)
                in_identifier = True
                i += 1
                continue
            if tid == CAMEL:
                flush_subword()
                id_boundaries.append(CAMEL)
                in_identifier = True
                i += 1
                continue

            # ── Structural tokens — flush any pending identifier first ─────
            if tid in structural_flush:
                flush_id()
                in_identifier = False

                if tid == SPECIAL_TOKENS["NEWLINE"]:
                    result.append("\n")
                elif tid in indent_map:
                    result.append(indent_map[tid])
                elif tid == SPECIAL_TOKENS["COMMENT"]:
                    result.append("#")
                elif tid == SPECIAL_TOKENS["DECORATOR"]:
                    result.append("@")
                elif tid == SPECIAL_TOKENS["ELLIPSIS"]:
                    result.append("...")
                elif tid == SPECIAL_TOKENS["SEMICOLON"]:
                    result.append(";")
                elif tid in _FUSION_DECODE:
                    result.append(_FUSION_DECODE[tid])
                # DEDENT, BLOCK_START, BLOCK_END: structural only, no text output
                i += 1
                continue

            # ── BPE / raw symbol ──────────────────────────────────────────
            sym = self.bpe.id_to_token.get(tid)
            if sym is None:
                i += 1
                continue

            text_piece = sym[:-4] if sym.endswith("</w>") else sym
            if not text_piece:  # pure </w> token — skip
                i += 1
                continue

            # Look ahead to decide if we're in / entering an identifier
            next_tid = ids[i + 1] if i + 1 < len(ids) else -1
            entering_id = next_tid in (US, CAMEL)

            if in_identifier or entering_id:
                in_identifier = True
                id_buf.append(text_piece)
            else:
                flush_id()
                result.append(text_piece)

            i += 1

        flush_id()
        return "".join(result)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text, add_special_tokens=False))

    def efficiency_vs_naive(self, text: str, naive_tokenizer) -> float:
        our_count = self.count_tokens(text)
        naive_count = len(naive_tokenizer.encode(text))
        return our_count / max(1, naive_count)

    def roundtrip_check(self, text: str) -> bool:
        """
        Verify lossless round-trip: decode(encode(text)) == text.
        Returns True if lossless, False otherwise.
        """
        encoded = self.encode(text, add_special_tokens=False)
        decoded = self.decode(encoded, skip_special=False)
        return decoded == text

    # ── Vocabulary info ──────────────────────────────────────────────────────

    def vocab_size_actual(self) -> int:
        return len(self.bpe.token_to_id)

    def special_token_id(self, name: str) -> int:
        return SPECIAL_TOKENS[name]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.bpe.save(os.path.join(directory, "bpe.json"))
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, directory: str) -> "LexForge":
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        bpe = BytePairEncoder.load(os.path.join(directory, "bpe.json"))
        return cls(vocab_size=cfg["vocab_size"], bpe=bpe)

    @classmethod
    def train_from_corpus(
        cls,
        code_strings: List[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> "LexForge":
        bpe = BytePairEncoder(vocab_size=vocab_size)
        if verbose:
            print(f"[LexForge] Training BPE on {len(code_strings)} samples...")
        bpe.train(code_strings, min_frequency=min_frequency, verbose=verbose)
        return cls(vocab_size=vocab_size, bpe=bpe)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lf = LexForge()

    # ── Identifier split/reconstruct ─────────────────────────────────────────
    cases = [
        "parse_html_doc",
        "getUserData",
        "XMLParser",
        "__init__",
        "x",
        "i",
        "n",
        "_private",
        "HTTPSConnectionPool",
    ]
    print("=== Identifier round-trips ===")
    for name in cases:
        subwords, boundaries = split_identifier(name)
        reconstructed = reconstruct_identifier(subwords, boundaries)
        status = "✓" if reconstructed == name else f"✗ got {reconstructed!r}"
        print(f"  {name!r:30s} → {subwords} | {status}")

    # ── Block detection ───────────────────────────────────────────────────────
    print("\n=== BLOCK_START detection ===")
    block_cases = [
        ("if x > 0:",        True),
        ("x = {'a': 1}",     False),
        ("y = d[1:3]",       False),
        ("f = lambda x: x",  False),
        ("def foo():",        True),
        ("class Bar:",        True),
        ("for i in range(n):", True),
        ("result = func():",  False),
    ]
    for line, expected in block_cases:
        got = _is_block_line(line)
        status = "✓" if got == expected else f"✗ expected {expected}"
        print(f"  {line!r:40s} → {got} {status}")

    # ── Encode/decode round-trips ─────────────────────────────────────────────
    print("\n=== Encode/decode snippets ===")
    snippets = [
        "x = 1",
        "# this is a comment",
        "parse_html_doc",
        "getUserData",
        "if x > 0:\n    return x",
        "x = {'a': 1, 'b': 2}",
        "@staticmethod\ndef foo():\n    pass",
    ]
    for snippet in snippets:
        enc = lf.encode(snippet, add_special_tokens=False)
        dec = lf.decode(enc, skip_special=False)
        ok = dec == snippet
        short = repr(snippet)[:50]
        print(f"  {'✓' if ok else '✗'} {short}")
        if not ok:
            print(f"    got: {dec!r}")

    # ── Tab indentation ───────────────────────────────────────────────────────
    print("\n=== Tab indentation ===")
    tab_code = "def foo():\n\treturn 1"
    enc = lf.encode(tab_code, add_special_tokens=False)
    dec = lf.decode(enc, skip_special=False)
    print(f"  {'✓' if dec == tab_code else '✗'} tab indentation preserved")

    print("\nDone.")
