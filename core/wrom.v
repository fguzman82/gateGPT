// Wide weight ROMs for the 24-lane parallel matvec engine. One distributed ROM per
// projection tensor; each ROM word holds LANES=24 Q5.11 weights (one per output row in a
// tile) for a single input index. A word is addressed by tile*in_dim + i and packed with
// lane 0 in the least-significant 16 bits, so wdata[lane*16 +: 16] is that lane's weight.
// Loaded from the project's own generated/*_t.hex (24 x 4 = 96 hex digits per line).
module wrom #(
    parameter integer LANES = 24
) (
    input  wire [2:0]            sel,    // WQ WK WV WO FC1 FC2 LM
    input  wire [11:0]           addr,   // tile-word address: tile*in_dim + i
    output wire [LANES*16-1:0]   wdata
);
    localparam [2:0] S_WQ=3'd0, S_WK=3'd1, S_WV=3'd2, S_WO=3'd3,
                     S_FC1=3'd4, S_FC2=3'd5, S_LM=3'd6;
    localparam integer W = LANES*16;

    // words per tensor = tiles * in_dim:
    //   wq/wk/wv/wo: 1 tile  x 24 = 24      fc1: 4 tiles x 24 = 96
    //   fc2:        1 tile  x 96 = 96       lm:  2 tiles x 24 = 48
    (* rom_style="distributed" *) reg [W-1:0] wq  [0:23];
    (* rom_style="distributed" *) reg [W-1:0] wk  [0:23];
    (* rom_style="distributed" *) reg [W-1:0] wv  [0:23];
    (* rom_style="distributed" *) reg [W-1:0] wo  [0:23];
    (* rom_style="distributed" *) reg [W-1:0] fc1 [0:95];
    (* rom_style="distributed" *) reg [W-1:0] fc2 [0:95];
    (* rom_style="distributed" *) reg [W-1:0] lm  [0:47];

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
        (sel == S_WQ)  ? wq[addr[4:0]]  :
        (sel == S_WK)  ? wk[addr[4:0]]  :
        (sel == S_WV)  ? wv[addr[4:0]]  :
        (sel == S_WO)  ? wo[addr[4:0]]  :
        (sel == S_FC1) ? fc1[addr[6:0]] :
        (sel == S_FC2) ? fc2[addr[6:0]] :
                         lm[addr[5:0]];
endmodule
