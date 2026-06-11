"""
Train the reference NamesGPT on the public makemore names corpus (data/names.txt)
and save the float weights to tools/weights.npz.

Standard causal training with ABSOLUTE positions (token i always at position i), so
inference can use an incremental KV cache (a token's K/V never change as the sequence
grows). A name is modeled as the sequence  . n a m e .  ; at every position the net
predicts the next token. Vocabulary: 0='.', 1..26='a'..'z'.
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
    chars = ["."] + [chr(ord("a") + i) for i in range(26)]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


def load_dataset(cfg, stoi):
    """Full-sequence examples (absolute positions), right-padded + masked."""
    names = [w.strip() for w in open(os.path.join(ROOT, "data", "names.txt")) if w.strip()]
    B = cfg.block_size
    X, Y, M = [], [], []
    for w in names:
        toks = ([0] + [stoi[c] for c in w] + [0])[: B + 1]   # .name.  (capped)
        x, y = toks[:-1], toks[1:]
        L = len(x)
        X.append(x + [0] * (B - L))
        Y.append(y + [0] * (B - L))
        M.append([1.0] * L + [0.0] * (B - L))
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


def main():
    torch.manual_seed(SEED)
    cfg = ModelConfig()
    stoi, itos = build_vocab()
    X, Y, M = load_dataset(cfg, stoi)
    V = cfg.vocab_size
    print(f"dataset: {X.shape[0]} examples, block_size={cfg.block_size}")

    model = NamesGPT(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    n, bs, steps = X.shape[0], 512, 6000
    for step in range(steps):
        ix = torch.randint(0, n, (bs,))
        logits = model(X[ix])                                  # [bs, T, V]
        loss = F.cross_entropy(logits.reshape(-1, V), Y[ix].reshape(-1), reduction="none")
        loss = (loss * M[ix].reshape(-1)).sum() / M[ix].sum()  # mask padding
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(os.path.join(HERE, "weights.npz"), **sd)
    print("saved tools/weights.npz")

    # incremental float sampling (absolute positions) to sanity-check name quality
    model.eval()
    g = torch.Generator().manual_seed(7)
    for _ in range(12):
        seq, out = [0], []
        for _ in range(cfg.block_size):
            logits = model(torch.tensor([seq]))[0, -1, :]
            p = F.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(p, 1, generator=g).item()
            if nxt == 0:
                break
            out.append(itos[nxt]); seq.append(nxt)
        print("  ", "".join(out))


if __name__ == "__main__":
    main()
