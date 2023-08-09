import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparams
batch_size = 4
block_size = 32
max_iters = 15000
eval_iters = 300
learning_rate = 1e-2

torch.manual_seed(1337)

#set input data
text = open('input.txt', 'r').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# helper functions
def eval_model():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


class BigramLangModel(nn.Module):
    def __init__(self, v_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(v_size, v_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # B,T,C

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T). add T+1 into the T dimension
        for _ in range(max_new_tokens):
            logit, loss = self(idx)
            logit = logit[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


xb, yb = get_batch('train')
m = BigramLangModel(vocab_size)
logits, loss = m(xb, yb)

print('loss:', eval_model())
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for i in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print('loss:', eval_model())
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=300)[0].tolist()))
