import os
import time
from typing import Tuple, Dict, BinaryIO,List
from multiprocessing import Pool
from collections import defaultdict


class PretokenizerMP:
    def __init__(self, path: str, num_processes: int = 4, split_special_token: bytes = b"<|endoftext|>"):
        self.path = path
        self.num_processes = num_processes
        self.split_special_token = split_special_token
        self.global_counts = defaultdict(int)

    @staticmethod
    def word_to_byte_tuple(word: str) -> Tuple[bytes, ...]:
        return tuple(bytes([b]) for b in word.encode("utf-8"))

    @staticmethod
    def pretokenization(text_chunk: str) -> Dict[Tuple[bytes, ...], int]:
        word_frequencies = {}
        for w in text_chunk.split():
            word_frequencies[w] = word_frequencies.get(w, 0) + 1

        return {
            PretokenizerMP.word_to_byte_tuple(word): count
            for word, count in word_frequencies.items()
        }

    @staticmethod
    def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes):
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            pos = chunk_boundaries[bi]
            file.seek(pos)
            while True:
                mini = file.read(mini_chunk_size)
                if mini == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found = mini.find(split_special_token)
                if found != -1:
                    chunk_boundaries[bi] = pos + found
                    break
                pos += mini_chunk_size

        return sorted(set(chunk_boundaries))

    @staticmethod
    def worker(path: str, start: int, end: int):
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            return PretokenizerMP.pretokenization(chunk)

    def _merge(self, local_counts):
        for token, count in local_counts.items():
            self.global_counts[token] += count

    def run(self):
        with open(self.path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes, self.split_special_token)

        tasks = [(self.path, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

        t0 = time.perf_counter()
        with Pool(self.num_processes) as pool:
            for local_counts in pool.starmap(self.worker, tasks):
                self._merge(local_counts)
        total = time.perf_counter() - t0

        return self.global_counts, total

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    path = os.path.join(BASE_DIR,"TinyStories-valid.txt")
    print(f"path is {path}")
    pretokenizer = PretokenizerMP(path,num_processes=4)
    counts,elapsed = pretokenizer.run()

    print("Total time:",elapsed)
    print("Unique pre-tokens:",len(counts))