"""
Microassembler: emit the core's control program (microcode) as generated/ucode.hex
and the shared memory map / opcode constants as core/coremap.vh. The 16-position
embed/norm/K/V loop is unrolled. The sequencer fetches and executes one macro-op
per program word; the datapath actuators do the work.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- vmem memory map (word addresses, AW=10). KC/VC (0..767) are reused for the
#      post-attention vectors, which are computed after K/V are no longer needed. ----
MAP = dict(KC=0, VC=384, TMP=768, XN=792, QV=816, AO=840,
           WOT=0, X1=24, XN2=48, HID=72, H2T=168, X2=192, XF=216, LOG=240)
NE, BLOCK, MLP, VOCAB = 24, 16, 96, 27

# ---- opcodes ----
OP = dict(NOP=0, EMBED=1, NORM=2, MATV=3, ATTN=4, VADD=5, RELU=6, SAMPLE=7, HALT=8)
# matvec tensor select
WS = dict(WQ=0, WK=1, WV=2, WO=3, FC1=4, FC2=5, LM=6)
# gain select
GS = dict(G1=0, G2=1, GF=2)


def enc(op, wsel=0, in_dim=0, out_dim=0, descale=0, gsel=0, pos=0, a=0, b=0, d=0):
    w = (op & 0xF)
    w |= (wsel & 0xF) << 4
    w |= (in_dim & 0x7F) << 8
    w |= (out_dim & 0x7F) << 15
    w |= (descale & 0x1F) << 22
    w |= (gsel & 0x3) << 27
    w |= (pos & 0xF) << 29
    w |= (a & 0x7FF) << 33
    w |= (b & 0x7FF) << 44
    w |= (d & 0x7FF) << 55
    return w


def build():
    P = []
    M = MAP
    for p in range(BLOCK):
        P.append(enc(OP["EMBED"], pos=p, d=M["TMP"]))
        P.append(enc(OP["NORM"], a=M["TMP"], d=M["XN"], gsel=GS["G1"]))
        P.append(enc(OP["MATV"], wsel=WS["WK"], in_dim=NE, out_dim=NE,
                     descale=11, a=M["XN"], d=M["KC"] + p * NE))
        P.append(enc(OP["MATV"], wsel=WS["WV"], in_dim=NE, out_dim=NE,
                     descale=11, a=M["XN"], d=M["VC"] + p * NE))
    # last position's TMP/XN persist
    P.append(enc(OP["MATV"], wsel=WS["WQ"], in_dim=NE, out_dim=NE, descale=11,
                 a=M["XN"], d=M["QV"]))
    P.append(enc(OP["ATTN"]))                                  # fixed QV/KC/VC/AO
    P.append(enc(OP["MATV"], wsel=WS["WO"], in_dim=NE, out_dim=NE, descale=11,
                 a=M["AO"], d=M["WOT"]))
    P.append(enc(OP["VADD"], a=M["TMP"], b=M["WOT"], d=M["X1"], out_dim=NE))
    P.append(enc(OP["NORM"], a=M["X1"], d=M["XN2"], gsel=GS["G2"]))
    P.append(enc(OP["MATV"], wsel=WS["FC1"], in_dim=NE, out_dim=MLP, descale=11,
                 a=M["XN2"], d=M["HID"]))
    P.append(enc(OP["RELU"], a=M["HID"], d=M["HID"], out_dim=MLP))
    P.append(enc(OP["MATV"], wsel=WS["FC2"], in_dim=MLP, out_dim=NE, descale=11,
                 a=M["HID"], d=M["H2T"]))
    P.append(enc(OP["VADD"], a=M["X1"], b=M["H2T"], d=M["X2"], out_dim=NE))
    P.append(enc(OP["NORM"], a=M["X2"], d=M["XF"], gsel=GS["GF"]))
    P.append(enc(OP["MATV"], wsel=WS["LM"], in_dim=NE, out_dim=VOCAB, descale=11,
                 a=M["XF"], d=M["LOG"]))
    P.append(enc(OP["SAMPLE"]))                                # fixed LOG
    P.append(enc(OP["HALT"]))
    return P


def main():
    prog = build()
    with open(os.path.join(ROOT, "generated", "ucode.hex"), "w") as f:
        for w in prog:
            f.write(f"{w & ((1 << 72) - 1):018x}\n")
    with open(os.path.join(ROOT, "core", "coremap.vh"), "w") as f:
        f.write("// Auto-generated memory map + opcodes. Do not edit.\n")
        f.write(f"localparam integer NINSTR = {len(prog)};\n")
        for k, v in MAP.items():
            f.write(f"localparam [9:0] A_{k} = 10'd{v};\n")
        for k, v in OP.items():
            f.write(f"localparam [3:0] OP_{k} = 4'd{v};\n")
    print(f"emitted {len(prog)} instructions -> generated/ucode.hex")


if __name__ == "__main__":
    main()
