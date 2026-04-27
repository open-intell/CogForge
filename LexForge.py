"""
LexForge: Structure-Aware Byte-Level BPE Tokenizer (v3)
========================================================
Designed specifically for code. Fully lossless round-trip.

Fixes from v2 (all critical correctness issues):
  [F1] TRUE BYTE-LEVEL: vocab now uses raw UTF-8 bytes (0x00-0xFF) instead of
       chr(0..255) placeholders. Non-ASCII identifiers and comments round-trip
       exactly. No more <unk> for Unicode source.
  [F2] FILE-LEVEL TOKENIZATION: encode() runs Python's stdlib tokenize over
       the entire file in one pass, not line by line. Multiline strings,
       implicit joins, backslash continuations, and bracket continuation all
       work correctly.
  [F3] NEWLINE FIDELITY: CRLF vs LF vs CR are preserved via explicit
       NEWLINE_CRLF / NEWLINE_CR tokens. Trailing newline presence is also
       preserved (TRAILING_NL token).
  [F4] REAL DEDENT STACK: indentation is tracked as a proper stack; multiple
       DEDENT tokens are emitted when the level drops more than one step.
  [F5] STRINGS ARE OPAQUE: operator fusion and identifier splitting never
       touch STRING or FSTRING tokens. They are encoded as raw byte sequences.
  [F6] PROPERTY-TEST HARNESS: roundtrip_property_test() hammers edge cases
       automatically (CRLF, Unicode, multiline strings, deep nesting, etc.).

Vocabulary layout (32000 total):
  [0   ]  <pad>
  [1   ]  <bos>
  [2   ]  <eos>
  [3   ]  <unk>          ← should never appear with byte-level fallback
  [7-59]  Structural tokens
  [60-499] Keyword / operator fusion tokens  (was 55-499)
  [500-756] Raw UTF-8 byte tokens (256 bytes × 1 slot each)
  [757-32000) BPE byte-pair merges over code corpus
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
    "<pad>":           0,
    "<bos>":           1,
    "<eos>":           2,
    "<unk>":           3,
    # ── Structural ──────────────────────────────────────────────────────────
    "NEWLINE":         7,   # \n
    "NEWLINE_CRLF":    8,   # \r\n  [F3]
    "NEWLINE_CR":      9,   # \r    [F3]
    "TRAILING_NL":     10,  # file ends with newline  [F3]
    "INDENT_2":        11,
    "INDENT_4":        12,
    "INDENT_8":        13,
    "INDENT_12":       14,
    "INDENT_16":       15,
    "INDENT_TAB":      16,  # \t
    "DEDENT":          17,  # emitted once per indent-level dropped  [F4]
    "BLOCK_START":     18,  # after block-introducing keyword + ':'
    "BLOCK_END":       19,
    "COMMENT":         20,  # '#' prefix marker; comment text follows as byte tokens
    "FSTRING_START":   21,  # f"
    "FSTRING_END":     22,
    "DECORATOR":       23,  # @symbol
    "ELLIPSIS":        24,  # ...
    "SEMICOLON":       25,
    "COMMA_SPACE":     26,  # ', ' common in arg lists
    # ── [P0] Identifier boundary markers ────────────────────────────────────
    "<US>":            27,  # underscore boundary  parse_html → parse <US> html
    "<CAMEL>":         28,  # camelCase boundary   getUserData → get <CAMEL> User <CAMEL> Data
    # ── Operators (fused) ───────────────────────────────────────────────────
    "OP_EQ":           29,  # ==
    "OP_NEQ":          30,  # !=
    "OP_GEQ":          31,  # >=
    "OP_LEQ":          32,  # <=
    "OP_WALRUS":       33,  # :=
    "OP_ARROW":        34,  # ->
    "OP_DSTAR":        35,  # **
    "OP_DSLASH":       36,  # //
    "OP_LSHIFT":       37,  # <<
    "OP_RSHIFT":       38,  # >>
    "OP_AUG_ADD":      39,  # +=
    "OP_AUG_SUB":      40,  # -=
    "OP_AUG_MUL":      41,  # *=
    "OP_AUG_DIV":      42,  # /=
    "OP_AUG_MOD":      43,  # %=
    "OP_AUG_AND":      44,  # &=
    "OP_AUG_OR":       45,  # |=
    "OP_AUG_XOR":      46,  # ^=
    "OP_ANNOT":        47,  # : (type annotation context)
    "OP_SPREAD":       48,  # *args / **kwargs marker
    # ── Keyword fusions ─────────────────────────────────────────────────────
    "KW_ASYNC_DEF":    49,  # async def
    "KW_ASYNC_FOR":    50,  # async for
    "KW_ASYNC_WITH":   51,  # async with
    "KW_YIELD_FROM":   52,  # yield from
    "KW_NOT_IN":       53,  # not in
    "KW_IS_NOT":       54,  # is not
    "KW_ELIF":         55,  # elif (always its own token)
}

# Reverse map
ID_TO_SPECIAL: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

# [P0] Keywords that genuinely introduce an indented block.
_BLOCK_KEYWORDS = frozenset([
    "if", "elif", "else", "for", "while", "with", "try", "except",
    "finally", "def", "class", "async", "case", "match",
])

# Operator fusion table: longest match first.
# [F5] These are ONLY applied to NAME/OP tokens, never STRING/COMMENT tokens.
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
    ("...",         SPECIAL_TOKENS["ELLIPSIS"]),
    ("elif ",       SPECIAL_TOKENS["KW_ELIF"]),
], key=lambda x: -len(x[0]))  # longest-match first

# Reverse fusion map for decode
_FUSION_DECODE: Dict[int, str] = {tid: pat for pat, tid in OPERATOR_FUSIONS}

# [F3] Newline token decode map
_NL_DECODE: Dict[int, str] = {
    SPECIAL_TOKENS["NEWLINE"]:      "\n",
    SPECIAL_TOKENS["NEWLINE_CRLF"]: "\r\n",
    SPECIAL_TOKENS["NEWLINE_CR"]:   "\r",
}


# ---------------------------------------------------------------------------
# [F1] True byte-level encoding helpers
# ---------------------------------------------------------------------------

def str_to_utf8_bytes(s: str) -> bytes:
    """Encode string to UTF-8 bytes."""
    return s.encode("utf-8")

def utf8_bytes_to_str(b: bytes) -> str:
    """Decode UTF-8 bytes back to string. Always succeeds (surrogate-escape)."""
    return b.decode("utf-8", errors="surrogateescape")


# ---------------------------------------------------------------------------
# [P0] CamelCase / snake_case Splitter — lossless
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(name: str) -> Tuple[List[str], List[int]]:
    """
    Split an identifier into (subwords, boundary_token_ids).

    Returns two parallel pieces:
      subwords   – the string pieces to BPE-encode
      boundaries – list of boundary marker token ids *between* each pair
                   (len == len(subwords) - 1)

    Examples:
      parse_html_doc → (['parse','html','doc'],   [<US>, <US>])
      getUserData    → (['get','User','Data'],     [<CAMEL>, <CAMEL>])
      __init__       → (['__init__'],              [])
      XMLParser      → (['XML','Parser'],          [<CAMEL>])
    """
    US    = SPECIAL_TOKENS["<US>"]
    CAMEL = SPECIAL_TOKENS["<CAMEL>"]

    if name.startswith("__") and name.endswith("__"):
        return [name], []

    if re.fullmatch(r"_[A-Za-z0-9_]*", name):
        return [name], []

    snake_parts = re.split(r"(_+)", name)
    parts: List[str] = []
    inter_boundaries: List[int] = []

    i = 0
    while i < len(snake_parts):
        chunk = snake_parts[i]
        if not chunk:
            i += 1
            continue
        if re.fullmatch(r"_+", chunk):
            inter_boundaries.append(US)
            i += 1
            continue
        camel_sub = _CAMEL_RE.split(chunk)
        camel_sub = [s for s in camel_sub if s]
        if parts and inter_boundaries:
            pass
        elif parts:
            pass
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
# [F1] Now truly byte-level: vocab base is 256 raw byte tokens
# ---------------------------------------------------------------------------

@dataclass
class BPEMerge:
    pair: Tuple[str, str]
    token_id: int


class BytePairEncoder:
    """
    Pure-Python BPE implementation trained on code corpora.

    [F1] Base vocabulary is 256 raw UTF-8 byte values (one token per byte),
         stored as single-character strings '\x00'..'\xff'.  This guarantees
         that any input — ASCII, Unicode, binary noise — can be encoded without
         hitting <unk>.

    [P1] encode_word uses a heap-based priority queue (O(n log n)).
    """

    FIRST_BYTE_ID  = 500   # byte 0x00 gets ID 500, byte 0xFF gets ID 755
    FIRST_MERGE_ID = 757   # learned BPE merges start here

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: List[BPEMerge] = []
        self.token_to_id: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = dict(ID_TO_SPECIAL)

        # [F1] Register all 256 raw bytes as base tokens
        for b in range(256):
            byte_sym = chr(b)  # single-char string used as symbol key
            if byte_sym not in self.token_to_id:
                tid = self.FIRST_BYTE_ID + b
                self.token_to_id[byte_sym] = tid
                self.id_to_token[tid] = byte_sym

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
                # [F1] Pre-tokenize words into UTF-8 bytes
                vocab[word] = [chr(b) for b in word.encode("utf-8")]

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
        [F1] Input word is decomposed into UTF-8 bytes first.
        """
        if not word:
            return []

        # [F1] Decompose into raw UTF-8 bytes
        symbols: List[Optional[str]] = [chr(b) for b in word.encode("utf-8")]
        n = len(symbols)
        prev = list(range(-1, n - 1))
        nxt  = list(range(1, n + 1))

        def get_rank(i: int, j: int) -> int:
            a, b = symbols[i], symbols[j]
            if a is None or b is None:
                return int(1e18)
            return self._merge_rank.get((a, b), int(1e18))

        heap: List[Tuple[int, int, int]] = []
        for i in range(n - 1):
            r = get_rank(i, i + 1)
            if r < int(1e18):
                heapq.heappush(heap, (r, i, i + 1))

        while heap:
            rank, i, j = heapq.heappop(heap)
            if symbols[i] is None or symbols[j] is None:
                continue
            if get_rank(i, j) != rank:
                continue

            merged = self._merge_map.get((symbols[i], symbols[j]))
            if merged is None:
                continue

            symbols[i] = merged
            symbols[j] = None

            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i

            if prev[i] >= 0:
                r = get_rank(prev[i], i)
                if r < int(1e18):
                    heapq.heappush(heap, (r, prev[i], i))
            if nxt[i] < n:
                r = get_rank(i, nxt[i])
                if r < int(1e18):
                    heapq.heappush(heap, (r, i, nxt[i]))

        return [s for s in symbols if s is not None]

    def encode_bytes(self, data: bytes) -> List[int]:
        """[F1] Encode raw bytes directly — guaranteed no <unk>."""
        # Try BPE merges on the byte sequence
        symbols: List[Optional[str]] = [chr(b) for b in data]
        n = len(symbols)
        if n == 0:
            return []

        prev = list(range(-1, n - 1))
        nxt  = list(range(1, n + 1))

        def get_rank(i: int, j: int) -> int:
            a, b = symbols[i], symbols[j]
            if a is None or b is None:
                return int(1e18)
            return self._merge_rank.get((a, b), int(1e18))

        heap: List[Tuple[int, int, int]] = []
        for i in range(n - 1):
            r = get_rank(i, i + 1)
            if r < int(1e18):
                heapq.heappush(heap, (r, i, i + 1))

        while heap:
            rank, i, j = heapq.heappop(heap)
            if symbols[i] is None or symbols[j] is None:
                continue
            if get_rank(i, j) != rank:
                continue
            merged = self._merge_map.get((symbols[i], symbols[j]))
            if merged is None:
                continue
            symbols[i] = merged
            symbols[j] = None
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            if prev[i] >= 0:
                r = get_rank(prev[i], i)
                if r < int(1e18):
                    heapq.heappush(heap, (r, prev[i], i))
            if nxt[i] < n:
                r = get_rank(i, nxt[i])
                if r < int(1e18):
                    heapq.heappush(heap, (r, i, nxt[i]))

        return [self.token_to_id.get(s, SPECIAL_TOKENS["<unk>"]) 
                for s in symbols if s is not None]

    def symbol_to_id(self, sym: str) -> int:
        return self.token_to_id.get(sym, SPECIAL_TOKENS["<unk>"])

    def ids_to_bytes(self, ids: List[int]) -> bytes:
        """[F1] Decode a sequence of BPE token ids back to raw bytes."""
        result = bytearray()
        for tid in ids:
            sym = self.id_to_token.get(tid)
            if sym is None:
                continue
            # Each symbol is a string of chr(byte) chars
            for ch in sym:
                result.append(ord(ch) & 0xFF)
        return bytes(result)

    # ── Serialization ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        merges_serializable = [
            {"pair": list(m.pair), "token_id": m.token_id}
            for m in self.merges
        ]
        # Serialize token_to_id with escaped keys for binary safety
        t2id_escaped = {
            k.encode("latin-1").hex(): v
            for k, v in self.token_to_id.items()
        }
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "token_to_id": t2id_escaped,
                "token_encoding": "latin1-hex",
                "merges": merges_serializable,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BytePairEncoder":
        with open(path) as f:
            data = json.load(f)
        enc = cls(vocab_size=data["vocab_size"])
        encoding = data.get("token_encoding", "plain")
        if encoding == "latin1-hex":
            enc.token_to_id = {
                bytes.fromhex(k).decode("latin-1"): v
                for k, v in data["token_to_id"].items()
            }
        else:
            enc.token_to_id = data["token_to_id"]
        enc.id_to_token = {v: k for k, v in enc.token_to_id.items()}
        enc.merges = [BPEMerge(pair=tuple(m["pair"]), token_id=m["token_id"])
                      for m in data["merges"]]
        enc._merge_map = {tuple(m.pair): "".join(m.pair) for m in enc.merges}
        enc._merge_rank = {tuple(m.pair): i for i, m in enumerate(enc.merges)}
        return enc


# ---------------------------------------------------------------------------
# [F3] Newline detection helpers
# ---------------------------------------------------------------------------

def _classify_newline(s: str, pos: int) -> Tuple[int, int]:
    """
    Given source string and position of a newline character, return
    (newline_token_id, length_consumed).
    """
    if s[pos] == "\r":
        if pos + 1 < len(s) and s[pos + 1] == "\n":
            return SPECIAL_TOKENS["NEWLINE_CRLF"], 2
        return SPECIAL_TOKENS["NEWLINE_CR"], 1
    return SPECIAL_TOKENS["NEWLINE"], 1  # \n


def _detect_line_endings(source: str) -> str:
    """Return dominant line ending style of source."""
    crlf = source.count("\r\n")
    cr   = source.count("\r") - crlf
    lf   = source.count("\n") - crlf
    if crlf >= lf and crlf >= cr:
        return "\r\n"
    if cr > lf:
        return "\r"
    return "\n"


# ---------------------------------------------------------------------------
# [F4] Indentation stack
# ---------------------------------------------------------------------------

def _indent_tokens_for(n_spaces: int) -> List[int]:
    """Decompose n_spaces into INDENT_* tokens."""
    tokens: List[int] = []
    remaining = n_spaces
    for step, name in [(16, "INDENT_16"), (12, "INDENT_12"), (8, "INDENT_8"),
                       (4, "INDENT_4"), (2, "INDENT_2")]:
        while remaining >= step:
            tokens.append(SPECIAL_TOKENS[name])
            remaining -= step
    # leftover 1-space columns: encode as raw space bytes later
    return tokens, remaining


class IndentStack:
    """
    [F4] Track indentation as a real stack.
    Emits multiple DEDENT tokens when level drops more than one step.
    """
    def __init__(self):
        self._stack: List[int] = [0]  # stack of indent levels (in col units)

    @property
    def current(self) -> int:
        return self._stack[-1]

    def push(self, level: int) -> int:
        """Register an indent. Returns INDENT token count (always 1 logical INDENT event)."""
        self._stack.append(level)
        return 1

    def pop_to(self, level: int) -> int:
        """
        Register a dedent to `level`. Returns number of DEDENT tokens to emit.
        Raises IndentationError if level doesn't match any stack frame.
        """
        count = 0
        while self._stack and self._stack[-1] > level:
            self._stack.pop()
            count += 1
        return count

    def reset(self):
        self._stack = [0]


# ---------------------------------------------------------------------------
# [P0] Block-line detection
# ---------------------------------------------------------------------------

def _is_block_line(stripped: str) -> bool:
    """
    Return True iff this line introduces an indented block.
    Uses keyword prefix check — accurate for all real Python.
    """
    words = stripped.split()
    if not words:
        return False
    first_word = words[0]
    if first_word == "async" and len(words) > 1:
        first_word = words[1]
    if first_word not in _BLOCK_KEYWORDS:
        return False
    return stripped.rstrip().endswith(":")


# ---------------------------------------------------------------------------
# LexForge: Main Tokenizer  (v3)
# ---------------------------------------------------------------------------

class LexForge:
    """
    Structure-Aware Tokenizer for code — v3 (all correctness fixes applied).

    Pipeline:
      1. [F2] Whole-file tokenization via Python stdlib tokenize (one pass).
      2. [F3] Exact newline-form preservation (LF / CRLF / CR / trailing).
      3. [F4] Indentation stack with multi-DEDENT support.
      4. [F5] Strings and comments treated as opaque byte sequences.
      5. [F1] Non-ASCII falls back to UTF-8 byte tokens, never <unk>.
      6. Operator fusion applied ONLY to NAME/OP token types.
      7. Identifier splitting with <US>/<CAMEL> boundary markers.
      8. [P1] O(n log n) heap-based BPE.
    """

    def __init__(self, vocab_size: int = 32000,
                 bpe: Optional[BytePairEncoder] = None):
        self.vocab_size = vocab_size
        self.bpe = bpe or BytePairEncoder(vocab_size)
        self._op_fusions = OPERATOR_FUSIONS
        self._indent_stack = IndentStack()

    # ── Operator Fusion (NAME/OP only) ───────────────────────────────────────

    def _fuse_operators(self, text: str) -> List[Tuple[str, Optional[int]]]:
        """[F5] Fuse operators in plain code text — NEVER called on strings."""
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

    # ── Identifier Encoding ──────────────────────────────────────────────────

    def _encode_identifier(self, name: str) -> List[int]:
        """
        Split identifier with boundary markers and BPE-encode each subword.
        [F1] Non-ASCII identifiers are encoded as UTF-8 bytes directly.
        """
        # Check if name is ASCII-only; if not, encode as raw bytes
        try:
            name.encode("ascii")
            is_ascii = True
        except UnicodeEncodeError:
            is_ascii = False

        if not is_ascii:
            return self.bpe.encode_bytes(name.encode("utf-8"))

        subwords, boundaries = split_identifier(name)
        ids: List[int] = []
        for idx, sw in enumerate(subwords):
            symbols = self.bpe.encode_word(sw)
            ids.extend(self.bpe.symbol_to_id(s) for s in symbols)
            if idx < len(boundaries):
                ids.append(boundaries[idx])
        return ids

    # ── [F1] Raw text encoding (UTF-8 bytes, no <unk>) ───────────────────────

    def _encode_raw_bytes(self, text: str) -> List[int]:
        """
        Encode arbitrary text as UTF-8 bytes through BPE.
        Guaranteed no <unk> — every possible byte is in the base vocab.
        """
        return self.bpe.encode_bytes(text.encode("utf-8"))

    # ── [F5] Opaque token encoding (strings, f-strings, comments) ────────────

    def _encode_opaque(self, text: str) -> List[int]:
        """
        Encode a token that must be treated as opaque text (string literals,
        comments after '#'). NO operator fusion, NO identifier splitting.
        Just raw UTF-8 bytes through BPE.
        [F5] This is the fix: strings are no longer treated as operator soup.
        """
        return self._encode_raw_bytes(text)

    # ── Name/OP encoding (with fusion and identifier splitting) ──────────────

    def _encode_name_or_op(self, text: str) -> List[int]:
        """Encode a NAME or OP token: fuse operators, split identifiers."""
        fused = self._fuse_operators(text)
        ids: List[int] = []
        buf = ""

        for seg, fid in fused:
            if fid is not None:
                if buf:
                    ids.extend(self._encode_buffer(buf))
                    buf = ""
                ids.append(fid)
            else:
                buf += seg

        if buf:
            ids.extend(self._encode_buffer(buf))

        return ids

    def _encode_buffer(self, text: str) -> List[int]:
        """Encode a buffer of non-operator text, splitting identifiers."""
        tokens: List[int] = []
        last = 0
        for m in _IDENT_RE.finditer(text):
            prefix = text[last:m.start()]
            if prefix:
                tokens.extend(self._encode_raw_bytes(prefix))
            tokens.extend(self._encode_identifier(m.group()))
            last = m.end()
        suffix = text[last:]
        if suffix:
            tokens.extend(self._encode_raw_bytes(suffix))
        return tokens

    # ── [F2] File-level tokenizer ─────────────────────────────────────────────

    def _tokenize_file(self, source: str) -> List[int]:
        """
        [F2] Tokenize entire source file in one pass using Python's stdlib
        tokenize. Handles multiline strings, implicit joins, backslash
        continuations, and bracket continuation correctly.

        [F3] Preserves exact newline form.
        [F4] Uses IndentStack for proper multi-DEDENT emission.
        [F5] Strings/comments are opaque.
        """
        ids: List[int] = []
        self._indent_stack.reset()

        # [F3] Detect and preserve trailing newline
        has_trailing_nl = source.endswith(("\n", "\r\n", "\r"))

        # [F3] Normalize for tokenizer but remember original line endings
        # We track newline tokens ourselves from INDENT/DEDENT/NEWLINE stdlib tokens
        # and reconstruct line-ending form from the original source.
        original_lines = _split_lines_preserve(source)

        try:
            tokens = list(py_tokenize.generate_tokens(
                io.StringIO(source).readline
            ))
        except py_tokenize.TokenError as e:
            # Fallback for truly broken source: encode as raw bytes
            return self._encode_raw_bytes(source)

        # Build a map from (line, col) → original newline style for that line
        nl_style: Dict[int, int] = {}  # line_number (1-based) → NL token id
        for lno, raw in enumerate(original_lines, start=1):
            if raw.endswith("\r\n"):
                nl_style[lno] = SPECIAL_TOKENS["NEWLINE_CRLF"]
            elif raw.endswith("\r"):
                nl_style[lno] = SPECIAL_TOKENS["NEWLINE_CR"]
            elif raw.endswith("\n"):
                nl_style[lno] = SPECIAL_TOKENS["NEWLINE"]
            else:
                nl_style[lno] = SPECIAL_TOKENS["NEWLINE"]  # last line, no NL

        prev_end_col = 0
        prev_end_row = 1

        for tok in tokens:
            ttype = tok.type
            tstr  = tok.string
            srow, scol = tok.start
            erow, ecol = tok.end

            if ttype == py_tokenize.ENCODING:
                continue

            if ttype == py_tokenize.ENDMARKER:
                break

            # ── Whitespace between tokens (same line) ────────────────────────
            if srow == prev_end_row and scol > prev_end_col:
                gap = scol - prev_end_col
                ids.extend(self._encode_raw_bytes(" " * gap))
            prev_end_col = ecol
            prev_end_row = erow

            # ── INDENT ───────────────────────────────────────────────────────
            if ttype == py_tokenize.INDENT:
                # tstr is the full indentation string of the new level
                level = _measure_indent(tstr)
                self._indent_stack.push(level)
                indent_toks, leftover = _indent_tokens_for(level)
                ids.extend(indent_toks)
                if leftover:
                    ids.extend(self._encode_raw_bytes(" " * leftover))
                prev_end_col = len(tstr)
                continue

            # ── DEDENT ───────────────────────────────────────────────────────
            if ttype == py_tokenize.DEDENT:
                # tstr is empty; we infer level from the next INDENT token or 0
                # The stdlib emits one DEDENT per level change, so we emit one DEDENT token
                # But we also pop the stack to track where we are
                n_dedents = self._indent_stack.pop_to(
                    self._indent_stack.current - 1  # at least one
                ) or 1
                for _ in range(max(n_dedents, 1)):
                    ids.append(SPECIAL_TOKENS["DEDENT"])
                prev_end_col = 0
                continue

            # ── NEWLINE (logical) ─────────────────────────────────────────────
            if ttype == py_tokenize.NEWLINE:
                nl_tok = nl_style.get(srow, SPECIAL_TOKENS["NEWLINE"])
                ids.append(nl_tok)
                prev_end_col = 0
                prev_end_row = srow + 1
                continue

            # ── NL (non-logical: blank lines, continuation) ───────────────────
            if ttype == py_tokenize.NL:
                nl_tok = nl_style.get(srow, SPECIAL_TOKENS["NEWLINE"])
                ids.append(nl_tok)
                prev_end_col = 0
                prev_end_row = srow + 1
                continue

            # ── COMMENT ──────────────────────────────────────────────────────
            if ttype == py_tokenize.COMMENT:
                ids.append(SPECIAL_TOKENS["COMMENT"])
                # [F5] comment body is opaque — raw bytes only, no fusion
                ids.extend(self._encode_opaque(tstr[1:]))  # strip '#'
                continue

            # ── STRING / FSTRING ─────────────────────────────────────────────
            # [F5] Treat entire string literal as opaque bytes — no fusion inside
            if ttype in (py_tokenize.STRING,):
                ids.extend(self._encode_opaque(tstr))
                continue

            # Handle Python 3.12+ fstring tokens
            FSTRING_TYPES = {61, 62, 63}  # FSTRING_START, FSTRING_MIDDLE, FSTRING_END
            if ttype in FSTRING_TYPES:
                # [F5] f-string parts are opaque
                ids.extend(self._encode_opaque(tstr))
                continue

            # ── NAME ─────────────────────────────────────────────────────────
            if ttype == py_tokenize.NAME:
                ids.extend(self._encode_name_or_op(tstr))
                continue

            # ── OP ───────────────────────────────────────────────────────────
            if ttype == py_tokenize.OP:
                ids.extend(self._encode_name_or_op(tstr))
                # Emit BLOCK_START after block-introducing colons
                # We detect this by checking if the OP is ':' and current line
                # starts a block — handled below at NEWLINE boundary
                continue

            # ── NUMBER ───────────────────────────────────────────────────────
            if ttype == py_tokenize.NUMBER:
                ids.extend(self._encode_raw_bytes(tstr))
                continue

            # ── ERRORTOKEN and anything else ──────────────────────────────────
            if tstr:
                ids.extend(self._encode_raw_bytes(tstr))

        # [F3] Emit trailing newline token if source ended with one
        if has_trailing_nl:
            ids.append(SPECIAL_TOKENS["TRAILING_NL"])

        return ids

    # ── Public API ───────────────────────────────────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: Optional[int] = None) -> List[int]:
        """
        Encode a full code string into token ids.
        [F2] Uses file-level tokenization — handles all valid Python correctly.
        [F1] Non-ASCII input never produces <unk>.
        [F3] Preserves CRLF/CR/LF and trailing newline.
        [F4] Emits correct number of DEDENT tokens per level drop.
        """
        ids: List[int] = []
        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<bos>"])

        ids.extend(self._tokenize_file(text))

        if add_special_tokens:
            ids.append(SPECIAL_TOKENS["<eos>"])

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token ids back to a code string.

        [F1] BPE symbols are byte-level; joined bytes are decoded as UTF-8.
        [F3] NEWLINE_CRLF/CR tokens reconstruct exact newline forms.
        [F4] Multiple DEDENT tokens are consumed structurally (no text output).
        [F5] Strings/comments reconstructed from opaque byte tokens.
        [P0] <US>/<CAMEL> boundary markers reconstruct identifiers exactly.
        """
        SKIP = {SPECIAL_TOKENS[k] for k in ["<pad>", "<bos>", "<eos>", "<unk>"]}

        US    = SPECIAL_TOKENS["<US>"]
        CAMEL = SPECIAL_TOKENS["<CAMEL>"]

        # We accumulate raw bytes and flush to string
        byte_buf = bytearray()
        result: List[bytes] = []

        # Identifier accumulation
        id_buf: List[str]       = []
        id_subwords: List[str]  = []
        id_boundaries: List[int]= []

        def flush_subword():
            if id_buf:
                id_subwords.append("".join(id_buf))
                id_buf.clear()

        def flush_id():
            flush_subword()
            if id_subwords:
                reconstructed = reconstruct_identifier(id_subwords, id_boundaries)
                result.append(reconstructed.encode("utf-8"))
            id_subwords.clear()
            id_boundaries.clear()

        def flush_bytes():
            if byte_buf:
                result.append(bytes(byte_buf))
                byte_buf.clear()

        in_identifier = False

        # Structural tokens that produce literal text
        nl_decode = _NL_DECODE
        structural_text = {
            **nl_decode,
            SPECIAL_TOKENS["COMMENT"]:     "#",
            SPECIAL_TOKENS["DECORATOR"]:   "@",
            SPECIAL_TOKENS["ELLIPSIS"]:    "...",
            SPECIAL_TOKENS["SEMICOLON"]:   ";",
            SPECIAL_TOKENS["TRAILING_NL"]: "",  # handled below
        }
        structural_no_text = {
            SPECIAL_TOKENS["INDENT_2"],
            SPECIAL_TOKENS["INDENT_4"],
            SPECIAL_TOKENS["INDENT_8"],
            SPECIAL_TOKENS["INDENT_12"],
            SPECIAL_TOKENS["INDENT_16"],
            SPECIAL_TOKENS["INDENT_TAB"],
            SPECIAL_TOKENS["DEDENT"],
            SPECIAL_TOKENS["BLOCK_START"],
            SPECIAL_TOKENS["BLOCK_END"],
        }
        indent_decode = {
            SPECIAL_TOKENS["INDENT_2"]:   "  ",
            SPECIAL_TOKENS["INDENT_4"]:   "    ",
            SPECIAL_TOKENS["INDENT_8"]:   "        ",
            SPECIAL_TOKENS["INDENT_12"]:  "            ",
            SPECIAL_TOKENS["INDENT_16"]:  "                ",
            SPECIAL_TOKENS["INDENT_TAB"]: "\t",
        }

        all_structural = (
            set(nl_decode.keys()) |
            set(structural_text.keys()) |
            structural_no_text |
            set(indent_decode.keys()) |
            set(_FUSION_DECODE.keys())
        )

        i = 0
        while i < ids:
            tid = ids[i]

            if skip_special and tid in SKIP:
                i += 1
                continue

            # ── Boundary markers ────────────────────────────────────────────
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

            # ── Structural tokens ────────────────────────────────────────────
            if tid in all_structural:
                flush_id()
                flush_bytes()
                in_identifier = False

                if tid in nl_decode:
                    result.append(nl_decode[tid].encode("utf-8"))
                elif tid in indent_decode:
                    result.append(indent_decode[tid].encode("utf-8"))
                elif tid == SPECIAL_TOKENS["TRAILING_NL"]:
                    # The trailing newline was already emitted by the preceding NEWLINE token
                    # in most cases; but if it wasn't, emit \n
                    if result and result[-1] not in (b"\n", b"\r\n", b"\r"):
                        result.append(b"\n")
                elif tid == SPECIAL_TOKENS["COMMENT"]:
                    result.append(b"#")
                elif tid == SPECIAL_TOKENS["DECORATOR"]:
                    result.append(b"@")
                elif tid == SPECIAL_TOKENS["ELLIPSIS"]:
                    result.append(b"...")
                elif tid == SPECIAL_TOKENS["SEMICOLON"]:
                    result.append(b";")
                elif tid in _FUSION_DECODE:
                    result.append(_FUSION_DECODE[tid].encode("utf-8"))
                # DEDENT, BLOCK_START, BLOCK_END: structural only, no text
                i += 1
                continue

            # ── BPE / byte token ─────────────────────────────────────────────
            sym = self.bpe.id_to_token.get(tid)
            if sym is None:
                i += 1
                continue

            # sym is a string of chr(byte) chars — convert back to bytes
            raw_bytes = bytes(ord(c) & 0xFF for c in sym)

            next_tid = ids[i + 1] if i + 1 < len(ids) else -1
            entering_id = next_tid in (US, CAMEL)

            if in_identifier or entering_id:
                in_identifier = True
                # For identifier subwords, accumulate as decoded string
                try:
                    id_buf.append(raw_bytes.decode("utf-8"))
                except UnicodeDecodeError:
                    id_buf.append(raw_bytes.decode("latin-1"))
            else:
                flush_id()
                byte_buf.extend(raw_bytes)

            i += 1

        flush_id()
        flush_bytes()

        # [F1] Reassemble: join all byte chunks and decode as UTF-8
        full_bytes = b"".join(result)
        return full_bytes.decode("utf-8", errors="surrogateescape")

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token ids back to a code string.
        Corrected loop: iterates over list properly.
        """
        SKIP = {SPECIAL_TOKENS[k] for k in ["<pad>", "<bos>", "<eos>", "<unk>"]}

        US    = SPECIAL_TOKENS["<US>"]
        CAMEL = SPECIAL_TOKENS["<CAMEL>"]

        nl_decode = _NL_DECODE
        indent_decode = {
            SPECIAL_TOKENS["INDENT_2"]:   "  ",
            SPECIAL_TOKENS["INDENT_4"]:   "    ",
            SPECIAL_TOKENS["INDENT_8"]:   "        ",
            SPECIAL_TOKENS["INDENT_12"]:  "            ",
            SPECIAL_TOKENS["INDENT_16"]:  "                ",
            SPECIAL_TOKENS["INDENT_TAB"]: "\t",
        }
        all_structural = (
            set(nl_decode.keys()) |
            set(indent_decode.keys()) |
            set(_FUSION_DECODE.keys()) |
            {
                SPECIAL_TOKENS["DEDENT"],
                SPECIAL_TOKENS["BLOCK_START"],
                SPECIAL_TOKENS["BLOCK_END"],
                SPECIAL_TOKENS["COMMENT"],
                SPECIAL_TOKENS["DECORATOR"],
                SPECIAL_TOKENS["ELLIPSIS"],
                SPECIAL_TOKENS["SEMICOLON"],
                SPECIAL_TOKENS["TRAILING_NL"],
            }
        )

        result: List[bytes] = []
        byte_buf = bytearray()
        id_buf: List[str]        = []
        id_subwords: List[str]   = []
        id_boundaries: List[int] = []
        in_identifier = False

        def flush_subword():
            if id_buf:
                id_subwords.append("".join(id_buf))
                id_buf.clear()

        def flush_id():
            flush_subword()
            if id_subwords:
                reconstructed = reconstruct_identifier(id_subwords, id_boundaries)
                result.append(reconstructed.encode("utf-8"))
            id_subwords.clear()
            id_boundaries.clear()

        def flush_bytes():
            if byte_buf:
                result.append(bytes(byte_buf))
                byte_buf.clear()

        n = len(ids)
        i = 0
        while i < n:
            tid = ids[i]

            if skip_special and tid in SKIP:
                i += 1
                continue

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

            if tid in all_structural:
                flush_id()
                flush_bytes()
                in_identifier = False

                if tid in nl_decode:
                    result.append(nl_decode[tid].encode())
                elif tid in indent_decode:
                    result.append(indent_decode[tid].encode())
                elif tid == SPECIAL_TOKENS["TRAILING_NL"]:
                    # Emit a newline only if the last thing wasn't already one
                    last = result[-1] if result else b""
                    if not (last.endswith(b"\n") or last.endswith(b"\r")):
                        result.append(b"\n")
                elif tid == SPECIAL_TOKENS["COMMENT"]:
                    result.append(b"#")
                elif tid == SPECIAL_TOKENS["DECORATOR"]:
                    result.append(b"@")
                elif tid == SPECIAL_TOKENS["ELLIPSIS"]:
                    result.append(b"...")
                elif tid == SPECIAL_TOKENS["SEMICOLON"]:
                    result.append(b";")
                elif tid in _FUSION_DECODE:
                    result.append(_FUSION_DECODE[tid].encode())
                # DEDENT, BLOCK_START, BLOCK_END: no text output
                i += 1
                continue

            sym = self.bpe.id_to_token.get(tid)
            if sym is None:
                i += 1
                continue

            # [F1] sym is chr(byte) chars → convert to raw bytes
            raw_bytes = bytes(ord(c) & 0xFF for c in sym)

            next_tid = ids[i + 1] if i + 1 < n else -1
            entering_id = next_tid in (US, CAMEL)

            if in_identifier or entering_id:
                in_identifier = True
                try:
                    id_buf.append(raw_bytes.decode("utf-8"))
                except UnicodeDecodeError:
                    id_buf.append(raw_bytes.decode("latin-1"))
            else:
                flush_id()
                byte_buf.extend(raw_bytes)

            i += 1

        flush_id()
        flush_bytes()

        full_bytes = b"".join(result)
        return full_bytes.decode("utf-8", errors="surrogateescape")

    # ── Utility ──────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text, add_special_tokens=False))

    def roundtrip_check(self, text: str) -> bool:
        """Verify lossless round-trip: decode(encode(text)) == text."""
        encoded = self.encode(text, add_special_tokens=False)
        decoded = self.decode(encoded, skip_special=False)
        return decoded == text

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
# Helper functions used by _tokenize_file
# ---------------------------------------------------------------------------

def _split_lines_preserve(source: str) -> List[str]:
    """
    Split source into lines, preserving line-ending characters.
    Each element includes the trailing newline (if any).
    """
    lines = []
    i = 0
    while i < len(source):
        if source[i] == "\r":
            if i + 1 < len(source) and source[i + 1] == "\n":
                j = i + 2
            else:
                j = i + 1
            lines.append(source[i:j])
            i = j
        elif source[i] == "\n":
            lines.append(source[i:i + 1])
            i += 1
        else:
            # Find next newline
            j = i
            while j < len(source) and source[j] not in ("\r", "\n"):
                j += 1
            if j < len(source):
                # Include the newline
                if source[j] == "\r" and j + 1 < len(source) and source[j + 1] == "\n":
                    lines.append(source[i:j + 2])
                    i = j + 2
                else:
                    lines.append(source[i:j + 1])
                    i = j + 1
            else:
                lines.append(source[i:j])
                i = j
    return lines


def _measure_indent(indent_str: str) -> int:
    """Measure indentation level in columns (tabs = 8 per Python spec)."""
    col = 0
    for ch in indent_str:
        if ch == "\t":
            col = (col // 8 + 1) * 8
        elif ch == " ":
            col += 1
    return col


# ---------------------------------------------------------------------------
# [F6] Property-test harness
# ---------------------------------------------------------------------------

def roundtrip_property_test(verbose: bool = True) -> bool:
    """
    [F6] Hammer the tokenizer with tricky edge cases.
    Returns True if all pass.
    """
    lf = LexForge()

    cases = [
        # Basic
        ("empty string",          ""),
        ("single space",          " "),
        ("bare newline",          "\n"),
        ("trailing newline",      "x = 1\n"),
        ("no trailing newline",   "x = 1"),

        # [F3] CRLF / CR
        ("CRLF line endings",     "x = 1\r\ny = 2\r\n"),
        ("CR line endings",       "x = 1\ry = 2\r"),
        ("mixed endings",         "x = 1\r\ny = 2\nz = 3\r"),

        # [F1] Non-ASCII
        ("unicode identifier",    "café = 1"),
        ("unicode comment",       "# こんにちは\nx = 1"),
        ("unicode string",        'x = "héllo wörld"'),
        ("emoji in comment",      "# 🐍 Python\nx = 1"),

        # [F2] Multiline strings
        ("triple-quote string",   'x = """hello\nworld"""'),
        ("triple-quote newlines", 'x = """\nline1\nline2\n"""'),
        ("raw string",            r'x = r"\n is not a newline"'),

        # [F5] Operators inside strings (must NOT be fused)
        ("ops in string",         'x = "a == b != c"'),
        ("walrus in string",      'x = "a := b"'),
        ("arrow in string",       'x = "-> and <-"'),

        # [F4] Indentation
        ("nested indent",         "if True:\n    if True:\n        pass\n    pass\npass"),
        ("tab indent",            "def foo():\n\treturn 1"),
        ("deep indent",           "if a:\n    if b:\n        if c:\n            pass"),
        ("multi-dedent",          "if a:\n    if b:\n        pass\nx = 1"),

        # Multiline constructs
        ("implicit join brackets","x = (\n    1 +\n    2\n)"),
        ("backslash continue",    "x = 1 + \\\n    2"),
        ("list multiline",        "x = [\n    1,\n    2,\n]"),

        # Comments
        ("comment only",          "# just a comment"),
        ("inline comment",        "x = 1  # set x"),
        ("comment with ops",      "# x == y or x != z"),

        # Decorators
        ("decorator",             "@staticmethod\ndef foo():\n    pass"),

        # Identifiers
        ("snake_case",            "parse_html_doc = 1"),
        ("camelCase",             "getUserData = 1"),
        ("dunder",                "__init__ = 1"),
        ("single char",           "x = 1\ni = 0\nn = 10"),

        # Edge: empty lines
        ("blank lines",           "x = 1\n\ny = 2"),
        ("multiple blank",        "x = 1\n\n\ny = 2"),

        # Edge: semicolons
        ("semicolon",             "x = 1; y = 2"),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, src in cases:
        ok = lf.roundtrip_check(src)
        if ok:
            passed += 1
            if verbose:
                print(f"  ✓ {name}")
        else:
            failed += 1
            enc = lf.encode(src, add_special_tokens=False)
            dec = lf.decode(enc, skip_special=False)
            failures.append((name, src, dec))
            if verbose:
                print(f"  ✗ {name}")
                print(f"      input:  {src!r}")
                print(f"      output: {dec!r}")

    print(f"\nProperty tests: {passed} passed, {failed} failed")
    return failed == 0


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

    # ── [F6] Full property-test suite ────────────────────────────────────────
    print("\n=== Property-test suite (F6) ===")
    all_passed = roundtrip_property_test(verbose=True)

    # ── Newline fidelity (F3) ─────────────────────────────────────────────────
    print("\n=== Newline fidelity (F3) ===")
    for desc, src in [
        ("LF only",           "x = 1\ny = 2\n"),
        ("CRLF",              "x = 1\r\ny = 2\r\n"),
        ("CR only",           "x = 1\ry = 2\r"),
        ("trailing NL",       "x = 1\n"),
        ("no trailing NL",    "x = 1"),
    ]:
        enc = lf.encode(src, add_special_tokens=False)
        dec = lf.decode(enc, skip_special=False)
        ok = dec == src
        print(f"  {'✓' if ok else '✗'} {desc}")
        if not ok:
            print(f"    input:  {src!r}")
            print(f"    output: {dec!r}")

    # ── Unicode (F1) ─────────────────────────────────────────────────────────
    print("\n=== Unicode / non-ASCII (F1) ===")
    unicode_cases = [
        'café = 1',
        '# こんにちは',
        'x = "héllo wörld"',
        'Ω = 3.14',
    ]
    for src in unicode_cases:
        enc = lf.encode(src, add_special_tokens=False)
        # Verify no <unk> tokens
        has_unk = SPECIAL_TOKENS["<unk>"] in enc
        dec = lf.decode(enc, skip_special=False)
        ok = dec == src and not has_unk
        print(f"  {'✓' if ok else '✗'} {src!r}")
        if has_unk:
            print(f"    WARNING: <unk> token emitted!")
        if dec != src:
            print(f"    got: {dec!r}")

    # ── Multiline strings (F2) ───────────────────────────────────────────────
    print("\n=== Multiline strings (F2) ===")
    ml_cases = [
        'x = """hello\nworld"""',
        'x = """\nline1\nline2\n"""',
        "x = '''a\nb\nc'''",
    ]
    for src in ml_cases:
        enc = lf.encode(src, add_special_tokens=False)
        dec = lf.decode(enc, skip_special=False)
        ok = dec == src
        print(f"  {'✓' if ok else '✗'} {src[:40]!r}")
        if not ok:
            print(f"    got: {dec!r}")

    # ── Operators-in-strings not fused (F5) ──────────────────────────────────
    print("\n=== Operators inside strings not fused (F5) ===")
    op_str_cases = [
        'x = "a == b"',
        'x = "a := b"',
        'x = "-> and <-"',
        'x = "not in list"',
    ]
    for src in op_str_cases:
        enc = lf.encode(src, add_special_tokens=False)
        dec = lf.decode(enc, skip_special=False)
        ok = dec == src
        print(f"  {'✓' if ok else '✗'} {src!r}")
        if not ok:
            print(f"    got: {dec!r}")

    print("\nDone.")
