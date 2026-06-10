# microGPT-FPGA

A character-level transformer that generates names, implemented from scratch for the
**Xilinx Virtex-5 XC5VLX110T** (XUPV5 / ML509 board, ISE 14.7, Verilog-2001). The model
is trained in Python, quantized to fixed point, and run entirely in hardware; generated
names scroll on the board's character LCD and a rotary encoder sets the generation speed.

## Architecture

The inference core is built as a **microcode-ROM sequencer driving modular datapath
actuators** — not a hand-coded monolithic state machine. A small program ROM
(`generated/ucode.hex`, produced by `tools/ucode_asm.py`) encodes the transformer
schedule as macro-ops; a micro-PC fetches one per step and starts the matching actuator,
which does the work over a shared activation scratchpad (`vmem`).

Datapath actuators (`core/`):

| Module | Role |
|---|---|
| `matvec` | streaming multiply-accumulate (the linear projections) |
| `norm` | RMSNorm (`udiv` + `isqrt` primitives) |
| `attn` | single-position multi-head causal attention (scores → softmax → weighted sum) |
| `exp_unit` | fixed-point `exp` via table + linear interpolation |
| `sampler` | temperature softmax + LCG categorical sampling, or greedy argmax |
| `embed`, `vecop` | embedding lookup, residual add / ReLU |
| `wrom`, `grom`, `vmem` | weight ROMs, RMSNorm gains, activation scratchpad |

The model: 1 transformer block, `n_embed=24`, 4 heads × 6, MLP hidden 96, context 16,
vocabulary 27 (`.` + `a`–`z`). All arithmetic is signed **Q5.11** fixed point. The Python
integer reference (`tools/fixedpoint.py`) is the bit-exact specification the RTL matches.

## Layout

```
core/         independent inference core (RTL) + generated includes (*.vh)
board/        XUPV5 top, HD44780 LCD driver, rotary throttle, tokens/sec meter, UCF
tools/        model, training, fixed-point reference, weight/microcode export
data/         public makemore names corpus (training data)
generated/    fixed-point weight ROMs (*.hex) + microcode program (ucode.hex)
sim/          iSim testbenches (per-actuator + end-to-end golden)
```

## Build & run

Train and export the model artifacts (Python 3 + numpy + torch):

```bash
python tools/train.py            # -> tools/weights.npz
python tools/export.py           # -> generated/*.hex, core/core_params.vh, gains.vh
python tools/ucode_asm.py        # -> generated/ucode.hex, core/coremap.vh
```

Simulate the core against the golden (Xilinx iSim):

```bash
vlogcomp -work work core/*.v sim/tb_core.v
fuse -incremental -o sim/tb_core_sim work.tb_core
./sim/tb_core_sim -tclbatch sim/isim_run.tcl     # prints CORE PASS
```

Build the board bitstream (ISE 14.7):

```bash
xtclsh build_board_ise_project.tcl     # generates the ISE project
xtclsh run_board_bitgen.tcl            # synth -> map -> par -> bitgen
```

## Board

- 100 MHz oscillator; the DCM divides it to a 33.3 MHz core clock.
- On reset, names auto-generate at ~1 Hz; turn the rotary encoder to speed up (push to
  freeze). LCD row 1 shows the current name, row 2 the measured tokens/second. LED[7] is
  a 1 Hz heartbeat.
