import heapq
import pickle
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # id -> token is given; build token -> id
        self._byte_to_id = {v: k for k, v in vocab.items()}

        # merge ranks: (tokenA, tokenB) -> priority
        self._merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # -------------------------------------------------------
    # INTERNAL RULE: only merge if merged token exists in vocab
    # -------------------------------------------------------
    def _valid_merge(self, a: bytes, b: bytes) -> bool:
        merged = a + b
        # only merge pairs that are actually in the vocab
        return merged in self._byte_to_id

    def encode(self, text: str) -> list[int]:
        # Start with raw UTF-8 bytes
        tokens = [bytes([b]) for b in text.encode("utf-8")]

        if len(tokens) <= 1:
            return [self._byte_to_id[t] for t in tokens]

        # Helper: get merge rank if allowed
        def get_rank(i):
            if i < 0 or i >= len(tokens) - 1:
                return None
            pair = (tokens[i], tokens[i+1])
            rank = self._merge_ranks.get(pair)
            if rank is None:
                return None

            # reject merges not present in vocab
            if not self._valid_merge(pair[0], pair[1]):
                return None

            return rank

        # Build heap of merge candidates
        heap = []
        for i in range(len(tokens) - 1):
            r = get_rank(i)
            if r is not None:
                heapq.heappush(heap, (r, i))

        # Apply merges
        while heap:
            rank, i = heapq.heappop(heap)

            if i >= len(tokens) - 1:
                continue

            current_rank = get_rank(i)
            if current_rank != rank:
                continue

            # merge
            A, B = tokens[i], tokens[i+1]
            merged = A + B

            # extra guard
            if merged not in self._byte_to_id:
                continue

            tokens[i:i+2] = [merged]

            # neighbors may form new merge opportunities
            for j in (i - 1, i):
                r = get_rank(j)
                if r is not None:
                    heapq.heappush(heap, (r, j))

        # Map to IDs
        return [self._byte_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids: list[int]) -> str:
        byte_pieces = [self.vocab[token_id] for token_id in ids]
        full_bytes = b"".join(byte_pieces)
        return full_bytes.decode("utf-8", errors="replace")

    """
        # Example: streaming tokens from a text file line by line
        with open("big_file.txt", "r", encoding="utf-8") as f:
            for token_id in tokenizer.encode_iterable(f):
                process(token_id)   # You can write to disk, push to model, etc.
        """

    """
    def encode(self,text: str) -> list[int]:
        #encode input text into a sequence of tokenIds

        tokens = [bytes([b]) for b in text.encode("utf-8")]

        heap=[]
        for first,second in self.merges:
            i=0
            while i < len(tokens)-1:
                if tokens[i] == first and tokens[i+1]==second:
                    merged = first + second
                    tokens[i:i+2] = [merged]
                    continue
                i+=1

        ids=[ self._byte_to_id[token] for token in tokens]
        return ids
    """
