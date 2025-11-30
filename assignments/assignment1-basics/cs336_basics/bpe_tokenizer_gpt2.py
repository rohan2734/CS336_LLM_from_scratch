from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path
import regex as re

# GPT-2 style tokenization regex (the standard one used for byte-BPE)
GPT2_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    re.UNICODE,
)


def _bytes_to_unicode() -> Dict[int, str]:
    """
    GPT-2 reversible bytes->unicode mapping.
    Returns mapping int(byte) -> single-character unicode string.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train byte-level BPE compatible with the assignment reference.

    Returns:
        vocab: dict[int, bytes]  (0..255 raw bytes, then special tokens, then merged tokens)
        merges: list[tuple(bytes, bytes]]  (order of creation)
    """
    if special_tokens is None:
        special_tokens = []

    # 1) Build initial vocab: raw bytes 0..255 then special tokens
    vocab: Dict[int, bytes] = {}
    next_id = 0
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    initial_vocab_size = next_id
    merges_allowed = vocab_size - initial_vocab_size
    if merges_allowed <= 0:
        return vocab, []

    # 2) bytes <-> unicode mapping (GPT-2)
    b2u = _bytes_to_unicode()            # int -> single-char string
    u2b: Dict[str, bytes] = {v: bytes([k]) for k, v in b2u.items()}

    # ensure special tokens map to their bytes and are atomic
    special_set = set(special_tokens)
    for tok in special_tokens:
        u2b[tok] = tok.encode("utf-8")

    # 3) Read file (raw bytes -> decode for tokenization)
    raw = Path(input_path).read_bytes()
    text = raw.decode("utf-8", errors="ignore")

    # 4) Remove special tokens BEFORE pretokenization: split on special tokens so they don't get tokenized
    #    This ensures special tokens remain atomic and don't pollute surrounding tokens.
    if special_tokens:
        escaped = [re.escape(s) for s in special_tokens]
        splitter = re.compile("|".join(escaped))
        chunks = splitter.split(text)
        # Note: we intentionally discard the special tokens from chunks; they are represented separately below.
        # However, we still want to include them in the vocabulary (done above).
    else:
        chunks = [text]

    # 5) Pretokenize each chunk using GPT2_PAT and build token frequency table
    #    Each token (string) -> convert to bytes, then map each byte to a single unicode char token
    token_freq: Counter = Counter()
    for chunk in chunks:
        for m in GPT2_PAT.finditer(chunk):
            tok = m.group(0)
            if not tok:
                continue
            bseq = tok.encode("utf-8")
            # map bytes -> unicode chars (one char per byte)
            tup = tuple(b2u[b] for b in bseq)
            token_freq[tup] += 1

    # Also ensure special tokens are present as single-token tuples (they should not be split)
    for tok in special_tokens:
        token_freq[(tok,)] = token_freq.get((tok,), 0)

    # 6) BPE training loop (per-token counts; tokens are opaque unicode-strings)
    merges: List[Tuple[bytes, bytes]] = []
    for _ in range(merges_allowed):
        # Count adjacent pairs across all token tuples (weighted by frequency)
        pair_counts: Dict[Tuple[str, str], int] = {}
        for tup, freq in token_freq.items():
            if len(tup) < 2:
                continue
            # skip single-token special tokens (they will have len==1)
            for a, b in zip(tup, tup[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq

        if not pair_counts:
            break

        # Choose best pair: highest frequency, tie-break lexicographically (pair as tuple of strings)
        best_pair, _ = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        a_sym, b_sym = best_pair

        # Map the two symbols to raw bytes for the merges output
        a_bytes = u2b[a_sym]
        b_bytes = u2b[b_sym]
        merges.append((a_bytes, b_bytes))

        # Create merged symbol (opaque) and map its bytes
        merged_sym = a_sym + b_sym
        u2b[merged_sym] = a_bytes + b_bytes

        # Replace occurrences of the best_pair in all token tuples
        new_token_freq: Counter = Counter()
        for tup, freq in token_freq.items():
            # keep single-token special tokens as-is
            if len(tup) == 1 and tup[0] in special_set:
                new_token_freq[tup] += freq
                continue

            out = []
            i = 0
            L = len(tup)
            while i < L:
                if i + 1 < L and tup[i] == a_sym and tup[i + 1] == b_sym:
                    out.append(merged_sym)
                    i += 2
                else:
                    out.append(tup[i])
                    i += 1
            new_token_freq[tuple(out)] += freq
        token_freq = new_token_freq

        # Add merged token bytes to vocab in creation order
        vocab[next_id] = u2b[merged_sym]
        next_id += 1

    return vocab, merges
