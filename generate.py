import torch
from model import MiniGPT
from dataset import load_dataset

# Load dataset (vocab, encoders, decoders)
data, stoi, itos, vocab_size, encode, decode = load_dataset("data.txt")

device = "cpu"

# Load trained model
model = MiniGPT(vocab_size).to(device)
model.load_state_dict(torch.load("minigpt.pth", map_location=device))
model.eval()

# Starting token (empty context)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate text
out = model.generate(context, max_new_tokens=100)[0].tolist()
print("Generated text:\n", decode(out))
