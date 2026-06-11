// Wide weight ROMs for the 24-lane, 2-columns/cycle parallel matvec engine. One
// distributed ROM per projection tensor; each ROM word holds LANES=24 Q5.11 weights for
// TWO consecutive input columns (low half = col 2j, high half = col 2j+1), addressed by
// tile*(in_dim/2) + j. wdata[lane*16 +:16] is the lane's weight for col 2j and
// wdata[LANES*16 + lane*16 +:16] for col 2j+1. Loaded from generated/*_t.hex (48 x 4 =
// 192 hex digits per line).
module wrom #(
    parameter integer LANES = 24
) (
    input  wire [2:0]            sel,    // WQ WK WV WO FC1 FC2 LM
    input  wire [11:0]           addr,   // tile-word address: tile*(in_dim/2) + j
    output wire [2*LANES*16-1:0] wdata
);
    localparam [2:0] S_WQ=3'd0, S_WK=3'd1, S_WV=3'd2, S_WO=3'd3,
                     S_FC1=3'd4, S_FC2=3'd5, S_LM=3'd6;
    localparam integer W = 2*LANES*16;

    // words per tensor = tiles * in_dim/2:
    //   wq/wk/wv/wo: 1 tile  x 12 = 12     fc1: 4 tiles x 12 = 48
    //   fc2:        1 tile  x 48 = 48      lm:  2 tiles x 12 = 24
    (* rom_style="distributed" *) reg [W-1:0] wq  [0:11];
    (* rom_style="distributed" *) reg [W-1:0] wk  [0:11];
    (* rom_style="distributed" *) reg [W-1:0] wv  [0:11];
    (* rom_style="distributed" *) reg [W-1:0] wo  [0:11];
    (* rom_style="distributed" *) reg [W-1:0] fc1 [0:47];
    (* rom_style="distributed" *) reg [W-1:0] fc2 [0:47];
    (* rom_style="distributed" *) reg [W-1:0] lm  [0:23];

    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/wq_t.hex", wq);
        $readmemh("/home/hermes/microgpt_fpga/generated/wk_t.hex", wk);
        $readmemh("/home/hermes/microgpt_fpga/generated/wv_t.hex", wv);
        $readmemh("/home/hermes/microgpt_fpga/generated/wo_t.hex", wo);
        $readmemh("/home/hermes/microgpt_fpga/generated/fc1_t.hex", fc1);
        $readmemh("/home/hermes/microgpt_fpga/generated/fc2_t.hex", fc2);
        $readmemh("/home/hermes/microgpt_fpga/generated/lm_t.hex", lm);
    end

    assign wdata =
        (sel == S_WQ)  ? wq[addr[3:0]]  :
        (sel == S_WK)  ? wk[addr[3:0]]  :
        (sel == S_WV)  ? wv[addr[3:0]]  :
        (sel == S_WO)  ? wo[addr[3:0]]  :
        (sel == S_FC1) ? fc1[addr[5:0]] :
        (sel == S_FC2) ? fc2[addr[5:0]] :
                         lm[addr[4:0]];
endmodule
