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


def main():
    os.makedirs(GEN, exist_ok=True)
    cfg = ModelConfig()
    sd = dict(np.load(os.path.join(HERE, "weights.npz")))
    m = QModel(sd, cfg)

    sizes = {}
    sizes["tok_embed"] = write_hex("tok_embed.hex", m.tok)     # 27 x 24
    sizes["pos_embed"] = write_hex("pos_embed.hex", m.pos)     # 16 x 24
    sizes["wq"] = write_hex("wq.hex", m.wq)                     # 24 x 24
    sizes["wk"] = write_hex("wk.hex", m.wk)
    sizes["wv"] = write_hex("wv.hex", m.wv)
    sizes["wo"] = write_hex("wo.hex", m.wo)
    sizes["fc1"] = write_hex("fc1.hex", m.fc1)                  # 96 x 24
    sizes["fc2"] = write_hex("fc2.hex", m.fc2)                  # 24 x 96
    sizes["lm"] = write_hex("lm_head.hex", m.lm)               # 27 x 24
    sizes["g1"] = write_hex("gain1.hex", m.g1.reshape(1, -1))
    sizes["g2"] = write_hex("gain2.hex", m.g2.reshape(1, -1))
    sizes["gf"] = write_hex("gainf.hex", m.gf.reshape(1, -1))
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
