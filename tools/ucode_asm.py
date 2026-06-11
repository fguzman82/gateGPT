"""
Microassembler: emit the core's control program (microcode) for INCREMENTAL decoding
with a persistent KV cache. Per token the core processes ONE position: embed the new
token, compute its K/V into the cache slot KC[pos]/VC[pos], then attend over the valid
positions 0..pos, MLP, LM head, sample. The KC/VC cache lives in vmem and survives
across tokens (the working vectors are packed into 0..255 by live-range so they never
touch the cache region 256..1023).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---- vmem memory map (AW=10). Working vectors packed in 0..255 (reused by live range);
#      the KV cache (KC/VC) is persistent at 256..1023 and never overwritten by scratch. ----
MAP = dict(
    TMP=0, XN=24, QV=48, AO=72, WOT=96, X1=120, XN2=144,   # phase A working set
    HID=0, H2T=96, X2=144, XF=0, LOG=24,                    # phase B (reuses dead phase-A slots)
    KC=256, VC=640,                                         # persistent KV cache (16*24 each)
)
NE, BLOCK, MLP, VOCAB = 24, 16, 96, 27

OP = dict(NOP=0, EMBED=1, NORM=2, MATV=3, ATTN=4, VADD=5, RELU=6, SAMPLE=7, HALT=8)
WS = dict(WQ=0, WK=1, WV=2, WO=3, FC1=4, FC2=5, LM=6)
GS = dict(G1=0, G2=1, GF=2)


def enc(op, wsel=0, in_dim=0, out_dim=0, descale=0, gsel=0, a=0, b=0, d=0, use_pos=0):
    w = (op & 0xF)
    w |= (wsel & 0xF) << 4
    w |= (in_dim & 0x7F) << 8
    w |= (out_dim & 0x7F) << 15
    w |= (descale & 0x1F) << 22
    w |= (gsel & 0x3) << 27
    # bits 29..32 unused
    w |= (a & 0x7FF) << 33
    w |= (b & 0x7FF) << 44
    w |= (d & 0x7FF) << 55
    w |= (use_pos & 0x1) << 66       # d += pos_in*N_EMBED (KV cache write slot)
    return w


def build():
    M = MAP
    return [
        enc(OP["EMBED"], d=M["TMP"]),                                              # token_in @ pos_in
        enc(OP["NORM"], a=M["TMP"], d=M["XN"], gsel=GS["G1"]),
        enc(OP["MATV"], wsel=WS["WK"], in_dim=NE, out_dim=NE, descale=11, a=M["XN"], d=M["KC"], use_pos=1),
        enc(OP["MATV"], wsel=WS["WV"], in_dim=NE, out_dim=NE, descale=11, a=M["XN"], d=M["VC"], use_pos=1),
        enc(OP["MATV"], wsel=WS["WQ"], in_dim=NE, out_dim=NE, descale=11, a=M["XN"], d=M["QV"]),
        enc(OP["ATTN"]),                                                           # ctx_len = pos_in+1
        enc(OP["MATV"], wsel=WS["WO"], in_dim=NE, out_dim=NE, descale=11, a=M["AO"], d=M["WOT"]),
        enc(OP["VADD"], a=M["TMP"], b=M["WOT"], d=M["X1"], out_dim=NE),
        enc(OP["NORM"], a=M["X1"], d=M["XN2"], gsel=GS["G2"]),
        enc(OP["MATV"], wsel=WS["FC1"], in_dim=NE, out_dim=MLP, descale=11, a=M["XN2"], d=M["HID"]),
        enc(OP["RELU"], a=M["HID"], d=M["HID"], out_dim=MLP),
        enc(OP["MATV"], wsel=WS["FC2"], in_dim=MLP, out_dim=NE, descale=11, a=M["HID"], d=M["H2T"]),
        enc(OP["VADD"], a=M["X1"], b=M["H2T"], d=M["X2"], out_dim=NE),
        enc(OP["NORM"], a=M["X2"], d=M["XF"], gsel=GS["GF"]),
        enc(OP["MATV"], wsel=WS["LM"], in_dim=NE, out_dim=VOCAB, descale=11, a=M["XF"], d=M["LOG"]),
        enc(OP["SAMPLE"]),
        enc(OP["HALT"]),
    ]


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
