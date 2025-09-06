import torch

def load_dataset(filename="data.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    vocab_size = len(chars)
    
    return data, stoi, itos, vocab_size
