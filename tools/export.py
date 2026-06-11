"""
Export the quantized model to RTL artifacts:
  - generated/*.hex     one row-major Q5.11 ROM per tensor (16-bit two's complement)
  - core/core_params.vh  dims, FRAC, attention scale, exp table, golden
All paths are inside this project; no external dependencies.
"""
import os
import numpy as np
from model import ModelConfig
from fixedpoint import QModel, generate, q, EXP_TAB, EXP_K, FRAC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN = os.path.join(ROOT, "generated")
COREP = os.path.join(ROOT, "core", "core_params.vh")


def write_hex(name, arr2d):
    """arr2d row-major -> hex file, one 4-digit (16-bit two's complement) word per line."""
    flat = np.asarray(arr2d).reshape(-1)
    with open(os.path.join(GEN, name), "w") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFFFF:04x}\n")
    return flat.size


def write_tiled_hex(name, W, lanes=24):
    """Wide weight ROM for the LANES-parallel, 2-columns/cycle matvec tile. W has shape
    (out_dim, in_dim). Output rows are split into tiles of LANES rows; each ROM word holds
    the LANES weights for TWO consecutive input columns (2j, 2j+1) across the tile's rows.
    A word is addressed by tile*(in_dim/2) + j; packing puts column 2j in the low LANES*16
    bits (lane 0 = LSB) and column 2j+1 in the high LANES*16 bits, so w_rdata[lane*16 +:16]
    is the lane's weight for col 2j and w_rdata[LANES*16 + lane*16 +:16] for col 2j+1.
    in_dim must be even; tiles past out_dim are zero-padded."""
    W = np.asarray(W)
    out_dim, in_dim = W.shape
    assert in_dim % 2 == 0, "2-column matvec needs even in_dim"
    tiles = (out_dim + lanes - 1) // lanes
    with open(os.path.join(GEN, name), "w") as f:
        for t in range(tiles):
            for j in range(in_dim // 2):
                word = ""
                # most-significant first: col 2j+1 (lane23..0), then col 2j (lane23..0)
                for col in (2 * j + 1, 2 * j):
                    for lane in reversed(range(lanes)):
                        o = t * lanes + lane
                        val = int(W[o, col]) if o < out_dim else 0
                        word += f"{val & 0xFFFF:04x}"
                f.write(word + "\n")
    return tiles * (in_dim // 2)


def main():
    os.makedirs(GEN, exist_ok=True)
    cfg = ModelConfig()
    sd = dict(np.load(os.path.join(HERE, "weights.npz")))
    m = QModel(sd, cfg)

    sizes = {}
    sizes["tok_embed"] = write_hex("tok_embed.hex", m.tok)     # 27 x 24 (embed reads this)
    sizes["pos_embed"] = write_hex("pos_embed.hex", m.pos)     # 16 x 24 (embed reads this)

    # tiled weight ROMs for the 24-lane, 2-columns/cycle matvec (one word = 48 weights).
    # These are the ONLY weight ROMs the RTL loads (wrom reads generated/*_t.hex). The
    # RMSNorm gains go into core/gains.vh as a combinational function (below), not a ROM.
    sizes["wq_t"] = write_tiled_hex("wq_t.hex", m.wq)          # 24 x 24
    sizes["wk_t"] = write_tiled_hex("wk_t.hex", m.wk)
    sizes["wv_t"] = write_tiled_hex("wv_t.hex", m.wv)
    sizes["wo_t"] = write_tiled_hex("wo_t.hex", m.wo)
    sizes["fc1_t"] = write_tiled_hex("fc1_t.hex", m.fc1)       # 96 x 24
    sizes["fc2_t"] = write_tiled_hex("fc2_t.hex", m.fc2)       # 24 x 96
    sizes["lm_t"] = write_tiled_hex("lm_t.hex", m.lm)          # 27 x 24
    sizes["exp"] = None
    with open(os.path.join(GEN, "exp_tab.hex"), "w") as f:
        for v in EXP_TAB:
            f.write(f"{int(v) & 0xFFFF:04x}\n")

    # gains as a combinational case (XST won't infer ROM for these tiny arrays and
    # ties $readmemh ones to zero -> emit explicit constants instead).
    with open(os.path.join(ROOT, "core", "gains.vh"), "w") as f:
        f.write("// Auto-generated RMSNorm gains (Q5.11). gsel 0=g1 1=g2 2=gf.\n")
        f.write("function signed [15:0] gain_lut;\n")
        f.write("    input [1:0] gsel; input [4:0] gidx;\n")
        f.write("    case ({gsel, gidx})\n")
        for si, gain in enumerate([m.g1, m.g2, m.gf]):
            for idx in range(cfg.n_embed):
                key = (si << 5) | idx
                f.write(f"        7'd{key}: gain_lut = 16'sh{int(gain[idx]) & 0xFFFF:04x};\n")
        f.write("        default: gain_lut = 16'sd0;\n    endcase\nendfunction\n")

    # golden for the testbench
    gseed, gtemp = 2, 0.7
    toks, s = generate(m, gseed, q(1.0 / gtemp))
    gtoks, gs = generate(m, 0, q(1.0 / gtemp), greedy=True)

    with open(COREP, "w") as f:
        f.write("// Auto-generated model parameters (Q5.11). Do not edit by hand.\n")
        f.write(f"localparam integer FRAC_BITS = {FRAC};\n")
        f.write(f"localparam integer VOCAB    = {cfg.vocab_size};\n")
        f.write(f"localparam integer N_EMBED  = {cfg.n_embed};\n")
        f.write(f"localparam integer N_HEAD   = {cfg.n_head};\n")
        f.write(f"localparam integer HEAD_DIM = {cfg.head_dim};\n")
        f.write(f"localparam integer MLP_HID  = {cfg.mlp_hidden};\n")
        f.write(f"localparam integer BLOCK    = {cfg.block_size};\n")
        f.write(f"localparam integer EXP_K    = {EXP_K};\n")
        f.write(f"localparam signed [15:0] ATTN_SCALE = 16'sd{m.attn_scale};\n")
        f.write(f"// golden sample seed={gseed} T={gtemp}: '{s}' tokens={toks}\n")
        f.write(f"// golden greedy: '{gs}' tokens={gtoks}\n")

    print("wrote ROMs:", {k: v for k, v in sizes.items() if v})
    print(f"GOLDEN sample seed={gseed} T={gtemp}: tokens={toks} '{s}'")
    print(f"GOLDEN greedy: tokens={gtoks} '{gs}'")


if __name__ == "__main__":
    main()
