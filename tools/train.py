"""
Train the reference NamesGPT on the public makemore names corpus (data/names.txt)
and save the float weights to tools/weights.npz. Our own training script.

Vocabulary: index 0 is the delimiter '.', indices 1..26 are 'a'..'z'. A name is
modeled as  . n a m e .  so the net learns to start after '.' and to emit '.' to end.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from model import ModelConfig, NamesGPT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = 1337


def build_vocab():
    # fixed, explicit: '.' + a..z
    chars = ["."] + [chr(ord("a") + i) for i in range(26)]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def load_dataset(cfg, stoi):
    names = [w.strip() for w in open(os.path.join(ROOT, "data", "names.txt")) if w.strip()]
    xs, ys = [], []
    for w in names:
        toks = [0] + [stoi[c] for c in w] + [0]      # .name.
        for i in range(1, len(toks)):
            ctx = toks[max(0, i - cfg.block_size):i]
            ctx = [0] * (cfg.block_size - len(ctx)) + ctx
            xs.append(ctx)
            ys.append(toks[i])
    return torch.tensor(xs), torch.tensor(ys)


def main():
    torch.manual_seed(SEED)
    cfg = ModelConfig()
    stoi, itos = build_vocab()
    X, Y = load_dataset(cfg, stoi)
    print(f"dataset: {X.shape[0]} examples, block_size={cfg.block_size}")

    model = NamesGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    n, bs, steps = X.shape[0], 512, 6000
    for step in range(steps):
        ix = torch.randint(0, n, (bs,))
        logits = model(X[ix])[:, -1, :]          # predict next from last position
        loss = F.cross_entropy(logits, Y[ix])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # save weights as plain numpy (our portable format)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(os.path.join(HERE, "weights.npz"), **sd)
    print("saved tools/weights.npz")

    # quick float sample to sanity-check name quality
    model.eval()
    g = torch.Generator().manual_seed(7)
    for _ in range(12):
        ctx = [0] * cfg.block_size
        out = []
        for _ in range(cfg.block_size):
            x = torch.tensor([ctx])
            logits = model(x)[0, -1, :]
            p = F.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(p, 1, generator=g).item()
            if nxt == 0:
                break
            out.append(itos[nxt]); ctx = ctx[1:] + [nxt]
        print("  ", "".join(out))


if __name__ == "__main__":
    main()
