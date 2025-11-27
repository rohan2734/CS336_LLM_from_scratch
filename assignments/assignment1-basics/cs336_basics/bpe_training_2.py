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


def bpe_train(corpus: str, vocab: Dict[str, int],vocab_size=500, initial_vocab_size=256) -> List[Tuple[bytes, bytes]]:
    """Train BPE tokenizer on a corpus.
    
    Args:
        corpus: Input text corpus as a string
        vocab: Vocabulary dictionary 
        
    Returns:
        List of merges, where each merge is a tuple of (bytes, bytes)
    """
    # Pretokenization: split on whitespace
   
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


bpe_train(corpus, vocab)




