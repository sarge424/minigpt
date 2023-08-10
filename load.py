import torch
import torch.nn
from main import GPT, decode

max_tokens = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT()
model.load_state_dict(torch.load('models/model64-256-384-6-6-5000-3E-04.pt'))
m = model.to(device)

m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)

output = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
with open('output.txt', 'w') as file:
    file.write(output)

