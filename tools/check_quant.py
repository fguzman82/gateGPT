"""Sanity-check the fixed-point model: generate names, print a deterministic golden."""
import os
import numpy as np
from model import ModelConfig
from fixedpoint import QModel, generate, q

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = ModelConfig()
    sd = dict(np.load(os.path.join(HERE, "weights.npz")))
    model = QModel(sd, cfg)

    inv_temp = q(1.0 / 0.7)   # temperature 0.7 -> 1/0.7 in Q11
    print("== sampled (seed sweep, T=0.7) ==")
    for seed in range(1, 16):
        _, s = generate(model, seed, inv_temp)
        print(f"  seed={seed:2d}  {s}")

    print("== greedy (deterministic) ==")
    toks, s = generate(model, 0, inv_temp, greedy=True)
    print(f"  greedy -> {s}  tokens={toks}")

    # GOLDEN: fixed seed + temperature, the bit-exact sequence the RTL must match
    gseed, gtemp = 2, 0.7
    toks, s = generate(model, gseed, q(1.0 / gtemp))
    print(f"GOLDEN seed={gseed} T={gtemp}: tokens={toks} name='{s}'")


if __name__ == "__main__":
    main()
