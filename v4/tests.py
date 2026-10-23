import torch

from model import MiniK3



device = "cuda" if torch.cuda.is_available() else "cpu"

ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
print(ids.shape)

model = MiniK3().to(device)

print("created model")

out, cache = model(ids)

print(out.shape)

print("\n\n")
print(torch.argmax(out[0][4]))

print(cache)