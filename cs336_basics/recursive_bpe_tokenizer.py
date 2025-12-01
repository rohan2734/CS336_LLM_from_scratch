import regex as re
from cs336_basics.pretokenization_multiprocsesing_4 import  PretokenizerMP

class RecursiveBPETokenizer:
    def __init__(self, merges, vocab):
        self.vocab = vocab

        # Reverse lookup for merge priority
        self.merge_rank = {
            (A, B): i for i, (A, B) in enumerate(merges)
        }

        # GPT-2 regex pattern used earlier
        self.pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def _merge_sequence(self, seq):
        """Apply BPE merges recursively to a byte tuple sequence."""
        seq = list(seq)

        while True:
            if len(seq) <= 1:
                break

            # Find all mergeable pairs
            pairs = [(i, (seq[i], seq[i+1])) for i in range(len(seq)-1)]

            # Among mergeable pairs, pick the one with smallest merge rank
            merge_candidates = [(i, pair) for i, pair in pairs if pair in self.merge_rank]

            if not merge_candidates:
                break

            # Pick best pair
            i, best_pair = min(merge_candidates, key=lambda x: self.merge_rank[x[1]])

            # Merge A,B into (A+B)
            A, B = best_pair
            new_token = A + B

            # Rebuild sequence with merge applied
            seq = seq[:i] + [new_token] + seq[i+2:]

        return tuple(seq)

    def encode(self, text):
        """Encode a string into a list of token IDs using BPE merges."""
        tokens = []

        # GPT-2 regex splitting
        for match in self.pat.finditer(text):
            tok = match.group(0)
            bt = PretokenizerMP.word_to_byte_tuple(tok)

            # apply recursive BPE merges to this byte-tuple
            merged = self._merge_sequence(bt)

            # map each final merged token to an ID
            for t in merged:
                tokens.append(self.vocab[t])

        return tokens