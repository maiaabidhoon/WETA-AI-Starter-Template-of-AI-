import torch
from dataset import load_dataset
from model import MiniGPT

# Load dataset
data, stoi, itos, vocab_size = load_dataset("data.txt")
device = "cpu"

# Load final trained model
model = MiniGPT(vocab_size, block_size=16, device=device).to(device)
model.load_state_dict(torch.load("minigpt_final.pth", map_location=device))
model.eval()

def generate_text(prompt, max_new_tokens=100):
    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return ''.join([itos[i] for i in out])

print("🤖 MiniGPT Chat (type 'exit' to quit)")
while True:
    prompt = input("You: ")
    if prompt.lower() == "exit":
        break
    response = generate_text(prompt, max_new_tokens=80)
    print("AI:", response)
