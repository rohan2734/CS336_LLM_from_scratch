from typing import List, Dict, Tuple


def word_to_byte_tuple(word: str) -> Tuple[bytes, ...]:
    """Convert a word string to a tuple of individual bytes.
    
    Args:
        word: A string to convert to bytes
        
    Returns:
        A tuple where each element is a single-byte bytes object
    """
    return tuple(bytes([byte_value]) for byte_value in word.encode("utf-8"))


corpus = """ 
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
"""

# Expected word frequencies: {low: 5, lower: 2, widest: 3, newest: 6}

vocab: Dict[str, int] = {}


def bpe_tokenizer_train(corpus: str, vocab: Dict[str, int],vocab_size=500, initial_vocab_size=256) -> List[Tuple[bytes, bytes]]:
    """Train BPE tokenizer on a corpus.
    
    Args:
        corpus: Input text corpus as a string
        vocab: Vocabulary dictionary 
        
    Returns:
        List of merges, where each merge is a tuple of (bytes, bytes)
    """
    # Pretokenization: split on whitespace

    print("bpe_tokenizer_train called")
    word_frequencies: Dict[str, int] = {}
    
    for word in corpus.split():
        word_frequencies[word] = word_frequencies.get(word, 0) + 1
    
    print(f"word_frequencies: {word_frequencies}")
    
    # Convert words to byte tuples for BPE processing
    byte_word_frequencies: Dict[Tuple[bytes, ...], int] = {
        word_to_byte_tuple(word): count 
        for word, count in word_frequencies.items()
    }
    
    print(f"byte_word_frequencies: {byte_word_frequencies}")
    
    # merges
    merges=[]
    while len(merges) < vocab_size - initial_vocab_size:
        # merge the pairs based on how they appear in the pretokenized byte word frequencies
        
        #1. count pairs
        pair_counts={}
        for byte_tuple,word_frequency in byte_word_frequencies.items():
            consecutive_pairs=zip(byte_tuple,byte_tuple[1:])

            for consecutive_pair in consecutive_pairs:
                pair_counts[consecutive_pair] = pair_counts.get(consecutive_pair,0)+ word_frequency 

        #2. find the best pair
        # Check if there are any pairs left to merge
        if not pair_counts:
            print("No more pairs to merge. Stopping.")
            break
            
        # When frequencies are equal, pick lexicographically greater pair
        # key=(frequency, pair) means: sort by frequency first, then by pair
        best_pair, best_pair_count = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        print(f"Best pair to merge: {best_pair} (appears {best_pair_count} times)")

        #3. merge the pair
        merged_token = best_pair[0] + best_pair[1]

        #4. update byte_word_frequencies
        new_byte_word_frequencies={}
        for byte_tuple,word_frequency in byte_word_frequencies.items():
            # build new tuple
            new_list=[]
            i=0
            while i < len(byte_tuple):
                if i +1 < len(byte_tuple) and (byte_tuple[i],byte_tuple[i+1])==best_pair:
                    new_list.append(merged_token)
                    i+=2
                else:
                    new_list.append(byte_tuple[i])
                    i+=1
            new_tuple=tuple(new_list)

            new_byte_word_frequencies[new_tuple] = word_frequency



        byte_word_frequencies = new_byte_word_frequencies

        #5. add to merges
        merges.append(best_pair)

    
    

    # Return the list of merges
    return merges


# merges=bpe_tokenizer_train(corpus, vocab)
# print(f"merges is {merges}")

def bpe_tokenize(word: str, merges: List[Tuple[bytes, bytes]]):
    tokens = list(word_to_byte_tuple(word))  # start from raw bytes

    for merge_pair in merges:
        i = 0
        merged = []
        while i < len(tokens):
            if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == merge_pair:
                merged.append(tokens[i] + tokens[i+1]) # combined token
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    
    return tokens

# tokens = bpe_tokenize("newest", merges)
# print(tokens)

def train_bpe(input_path:str, vocab_size:int , special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    """Train BPE tokenizer on a corpus

    Args:
        input_path: Input text corpus from input path
        vocab_size: maximum vocabulary size
        special_tokens: List of special tokens to ignore

    Returns:
        vocab : vocabulary dictionary with tokenId, bytes
        merges: List of merges, where each merge is a tuple of (bytes, bytes)
    """

    """ GPT-2’s training code (OpenAI version) does this:
        - Read file in raw bytes
        - Decode using UTF-8 but ignore errors
        - Split into words
        - Convert each word back into bytes
        - Run byte-pair merges
        - This ensures:
        - perfect byte reproduction
        - correct Unicode handling
        - correct word segmentation
        - merges operate on raw byte sequences
    """
    # Pretokenization: split on whitespace
    print("train_bpe called")

    if special_tokens is None:
        special_tokens = []

        # ---------------------------
        # 1. Initialize vocabulary
        # ---------------------------
    vocab: Dict[int, bytes] = {}
    next_id = 0

    # Add raw byte tokens (0–255)
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1

    # Add special tokens
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    initial_vocab_size = next_id
    num_merges_allowed = vocab_size - initial_vocab_size

    # 2. Read data
    with open(input_path, "rb") as f:
        data = f.read()

    text = data.decode("utf-8", errors="ignore")

    # 3. Word frequencies
    word_frequencies: Dict[str, int] = {}
    for word in text.split():
        word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Convert words to byte tuples
    byte_word_frequencies: Dict[Tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in word.encode("utf-8")): count
        for word, count in word_frequencies.items()
    }

    # 4. BPE merge loop
    merges: List[Tuple[bytes, bytes]] = []

    while len(merges) < num_merges_allowed:
        # Count all byte pairs
        pair_counts = {}
        for byte_tuple, freq in byte_word_frequencies.items():
            for a, b in zip(byte_tuple, byte_tuple[1:]):
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if not pair_counts:
            break

        # Best pair = highest freq, tie-break lexicographically
        best_pair, _ = max(pair_counts.items(), key=lambda x: (x[1], x[0]))

        # New merged token
        merged_token = best_pair[0] + best_pair[1]

        # Add to vocab
        vocab[next_id] = merged_token
        next_id += 1

        # Replace occurrences of best_pair
        new_byte_word_frequencies = {}
        for byte_tuple, freq in byte_word_frequencies.items():
            new_list = []
            i = 0
            while i < len(byte_tuple):
                if i + 1 < len(byte_tuple) and (byte_tuple[i], byte_tuple[i + 1]) == best_pair:
                    new_list.append(merged_token)
                    i += 2
                else:
                    new_list.append(byte_tuple[i])
                    i += 1

            new_byte_word_frequencies[tuple(new_list)] = freq

        byte_word_frequencies = new_byte_word_frequencies

        merges.append(best_pair)

    return vocab, merges


