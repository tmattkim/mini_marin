import numpy as np
from datasets import load_dataset
from collections import Counter
import json
import os

def build_vocab(texts, vocab_size=10000):
    counter = Counter()
    for text in texts:
        tokens = text.strip().split()
        counter.update(tokens)
    most_common = counter.most_common(vocab_size - 2)  # reserve 2 for <unk> and <pad>
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab

def encode_text(text, vocab):
    tokens = text.strip().split()
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

def decode_tokens(tokens, inv_vocab):
    return " ".join([inv_vocab.get(t, "<unk>") for t in tokens])

def load_and_tokenize_dataset(path, block_size, vocab=None, build_vocab_flag=False, vocab_size=10000):
    raw_dataset = load_dataset("text", data_files=path, split="train")
    texts = [ex["text"] for ex in raw_dataset]

    if build_vocab_flag:
        vocab, inv_vocab = build_vocab(texts, vocab_size=vocab_size)
    else:
        inv_vocab = {v: k for k, v in vocab.items()}

    tokenized = []
    for text in texts:
        ids = encode_text(text, vocab)
        for i in range(0, len(ids) - block_size, block_size):
            block = ids[i:i + block_size + 1]
            tokenized.append(np.array(block))

    return tokenized, vocab, inv_vocab

def save_vocab(vocab, path):
    with open(path, "w") as f:
        json.dump(vocab, f)

def load_vocab(path):
    import json
    with open(path, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab
