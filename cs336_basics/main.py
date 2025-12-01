from cs336_basics.pretokenization_multiprocsesing_4 import PretokenizerMP
from cs336_basics.bpe_trainer import BPETrainer
from cs336_basics.recursive_bpe_tokenizer import RecursiveBPETokenizer
from cs336_basics.tokenizer import Tokenizer
import os
import time


def main():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "TinyStories-train.txt")

    print(f"\nUsing dataset: {path}\n")

    # Step 1–3: Pretokenization (MULTIPROCESS)
    t0 = time.perf_counter()
    pre = PretokenizerMP(path, num_processes=16)
    token_freqs, pretime = pre.run()
    t1 = time.perf_counter()

    print(f"Pretokenization time (reported by class): {pretime:.2f} sec")
    print(f"Total pretoke step time (wall clock): {t1 - t0:.2f} sec")
    print(f"Unique pre-tokens: {len(token_freqs)}\n")

    # Optional: show sample
    print("--- Sample pre-tokens ---")
    for k, v in list(token_freqs.items())[:10]:
        print(k, v)

    # Step 4: BPE Training
    t2 = time.perf_counter()
    trainer = BPETrainer(vocab_size=5000)
    merges, vocab = trainer.train(token_freqs)
    t3 = time.perf_counter()

    print("\nBPE Training complete!")
    print(f"Number of merges: {len(merges)}")
    print(f"Final vocab size: {len(vocab)}")
    print(f"BPE training time: {t3 - t2:.2f} sec")

    # Step 5: Recursive BPE Tokenizer
    # tokenizer = RecursiveBPETokenizer(merges, vocab)
    #
    # example = "Spot saw the shiny car."
    #
    # t4 = time.perf_counter()
    # encoded = tokenizer.encode(example)
    # t5 = time.perf_counter()

    # Convert vocab to expected format {id: bytes}
    # BPETrainer gives you {bytes: id}, so invert it
    id_to_bytes = {tid: tok for tok, tid in vocab.items()}

    # Initialize your tokenizer
    byte_tokenizer = Tokenizer(vocab=id_to_bytes, merges=merges)

    example = "Spot saw the shiny car."

    t4 = time.perf_counter()
    encoded = byte_tokenizer.encode(example)
    t5 = time.perf_counter()

    print("\nExample sentence:", example)
    print("Encoded token IDs:", encoded)
    print(f"Encoding time: {t5 - t4:.6f} sec")

    # Test decode
    decoded = byte_tokenizer.decode(encoded)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()