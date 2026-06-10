"""Dump a matvec unit-test vector: input = ones(Q11), expected = wq @ input."""
import os
import numpy as np
from model import ModelConfig
from fixedpoint import QModel, matvec, rmsnorm, exp_neg_q11

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
    xin = np.full(cfg.n_embed, 2048, dtype=np.int64)   # all ones in Q11
    exp = matvec(m.wq, xin)
    wr("test_in.hex", xin)
    wr("test_wq.hex", exp)
    print("test_in (24):", list(xin[:4]), "...")
    print("expected wq@ones (24):", [int(v) for v in exp])

    # norm test: a realistic vector (tok_embed[5] + pos_embed[3]) through rmsnorm(g1)
    xn = np.array([int(m.tok[5][i]) + int(m.pos[3][i]) for i in range(cfg.n_embed)], dtype=np.int64)
    xn = np.array([max(-32768, min(32767, int(v))) for v in xn], dtype=np.int64)
    nexp = rmsnorm(xn, m.g1)
    wr("test_norm_in.hex", xn)
    wr("test_norm_out.hex", nexp)
    print("expected rmsnorm (24):", [int(v) for v in nexp])

    # exp sweep
    zs = list(range(0, -33000, -337)) + [-1, -2047, -2048, -2049, -32768]
    es = [exp_neg_q11(z) for z in zs]
    wr("test_exp_z.hex", np.array(zs, dtype=np.int64))
    wr("test_exp_e.hex", np.array(es, dtype=np.int64))
    print(f"exp sweep: {len(zs)} cases")


if __name__ == "__main__":
    main()
