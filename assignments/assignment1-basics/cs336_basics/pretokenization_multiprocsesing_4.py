import os
from typing import BinaryIO, Dict,Tuple
import multiprocessing as Pool


def word_to_byte_tuple(word:str) -> Tuple[bytes,...]:
    return tuple( bytes([byte_val]) for byte_val in word.encode("utf-8") )

def pretokenization(
    text_chunk: str,

):
    word_frequencies : Dict[str,int] = {}

    for word in text_chunk.split():
        word_frequencies[word] = word_frequencies.get(word,0)+1

    byte_word_frequencies: Dict[tuple(bytes),int] = {
       word_to_byte_tuple(word)  :count for word,count  in word_frequencies.items()
    }

    return byte_word_frequencies


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

from collections import defaultdict
global_counts = defaultdict(int)

def aggregate_word_frequencies(local_counts):
    for token,count in local_counts.items():
        global_counts[token]  = global_counts.get(token,0) + count
    
## Usage

## sequential
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.

#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
#         local_counts= pretokenization(chunk)

#         aggregate_word_frequencies(local_counts)

def worker(path:str,start:int,end:int):
    with open(path,"rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        local_counts= pretokenization(chunk)
        
        return local_counts



path="./tinyStories.txt"
with open(path, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    tasks= [(path,start,end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    num_tasks =len(tasks)

    with Pool(num_processes) as pool:
        results = pool.starmap(worker, tasks)

        for local_counts in results:
            aggregate_word_frequencies(local_counts)


def bpe_tokenizer_train(input_path: str, vocab_size: int, special_tokens: List[str] = None):
    """Train BPE tokenizer."""