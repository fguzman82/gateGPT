"""Dump an attention unit-test: q, K cache, V cache, expected attn_out."""
import os
import numpy as np
from model import ModelConfig
from fixedpoint import QModel

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(os.path.dirname(HERE), "generated")


def wr(name, arr):
    with open(os.path.join(GEN, name), "w") as f:
        for v in np.asarray(arr).reshape(-1):
            f.write(f"{int(v) & 0xFFFF:04x}\n")


def main():
    cfg = ModelConfig()
    sd = dict(np.load(os.path.join(HERE, "weights.npz")))
    m = QModel(sd, cfg)
    # a realistic context: ".saa" left-padded
    seq = [0, 19, 1, 1]
    ctx = [0] * (cfg.block_size - len(seq)) + seq
    q, k, v, out = m.attn_debug(ctx)
    wr("test_attn_q.hex", q)              # 24
    wr("test_attn_k.hex", k)              # 16*24 row-major (s*24+e)
    wr("test_attn_v.hex", v)              # 16*24
    wr("test_attn_out.hex", out)          # 24
    print("attn_scale:", m.attn_scale)
    print("expected attn_out:", [int(x) for x in out])


if __name__ == "__main__":
    main()
