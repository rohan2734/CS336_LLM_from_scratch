from collections import defaultdict

class BPETrainer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges = []       # list of merges (pair A,B)
        self.vocab = {}        # token → id

    def _get_pairs(self, token_seq):
        """Return all adjacent pairs inside a token sequence."""
        return [(token_seq[i], token_seq[i+1]) for i in range(len(token_seq)-1)]

    def _count_pairs(self, token_freqs):
        """Count adjacent pairs across all sequences, weighted by frequency."""
        pair_counts = defaultdict(int)
        for seq, freq in token_freqs.items():
            pairs = self._get_pairs(seq)
            for p in pairs:
                pair_counts[p] += freq
        return pair_counts

    def _merge_pair_in_sequence(self, seq, pair_to_merge):
        """
        Replace every (A,B) in seq with merged token A+B.
        Example:
        seq = [(b't',), (b'h',), (b'e',)]
        pair_to_merge = ((b'h',), (b'e',))
        → [(b't',), (b'h', b'e')]
        """
        A, B = pair_to_merge
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == A and seq[i+1] == B:
                new_seq.append(A + B)
                i += 2      # skip next token
            else:
                new_seq.append(seq[i])
                i += 1
        return tuple(new_seq)

    def train(self, token_freqs):
        """
        token_freqs: Dict[Tuple[bytes,...], int]
        This is exactly what PretokenizerMP.run() returns.
        """
        # Extract initial vocab from byte tuples
        vocab = set()
        for seq in token_freqs.keys():
            for token in seq:
                vocab.add(token)

        # Continue merging pairs until vocab_size reached
        while len(vocab) < self.vocab_size:
            # print(f"Merging step {len(self.merges)}...")

            # Count all adjacent pairs
            pair_counts = self._count_pairs(token_freqs)

            if not pair_counts:
                break   # nothing to merge

            # Find best pair to merge
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges.append(best_pair)

            # Update vocab: add merged token
            merged_token = best_pair[0] + best_pair[1]
            vocab.add(merged_token)

            # Update all sequences in token_freqs
            new_token_freqs = {}
            for seq, freq in token_freqs.items():
                new_seq = self._merge_pair_in_sequence(seq, best_pair)
                new_token_freqs[new_seq] = new_token_freqs.get(new_seq, 0) + freq

            token_freqs = new_token_freqs

        # Build final vocab ID mapping
        self._build_vocab(vocab)
        return self.merges, self.vocab

    def _build_vocab(self, vocab):
        """
        Assign integer IDs to all tokens.
        Initial single-byte tokens get lower IDs;
        merged tokens get higher IDs in merge order.
        """
        # Start with sorted byte tokens (optional)
        byte_tokens = sorted([v for v in vocab if len(v) == 1])

        idx = 0
        for tok in byte_tokens:
            self.vocab[tok] = idx
            idx += 1

        # Add merged tokens in merge order
        for A, B in self.merges:
            tok = A + B
            if tok not in self.vocab:
                self.vocab[tok] = idx
                idx += 1