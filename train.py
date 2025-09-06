import torch
from dataset import load_dataset
from model import MiniGPT
import os

# Load dataset
data, stoi, itos, vocab_size = load_dataset("data.txt")
device = "cpu"

# Train/Val split
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 16
batch_size = 4

# Ensure stepdata folder exists
os.makedirs("stepdata", exist_ok=True)

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model init
model = MiniGPT(vocab_size, block_size=block_size, device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("🚀 Training started... (Press CTRL+C to stop)\n")

step = 0
try:
    while True:
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        if step % 500 == 0 and step > 0:
            torch.save(model.state_dict(), f"stepdata/minigpt_step{step}.pth")
            print(f"💾 Model saved at step {step}")

        step += 1

except KeyboardInterrupt:
    print("\n🛑 Training stopped by user.")
    torch.save(model.state_dict(), "minigpt_final.pth")
    print("✅ Final model saved as minigpt_final.pth")

    # Generate sample after training stop
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=100)[0].tolist()
    print("\nGenerated text:\n", ''.join([itos[i] for i in out]))
