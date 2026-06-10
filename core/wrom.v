// Weight ROMs for the matvec engine. One distributed ROM per projection tensor,
// row-major (addr = out_row*in_dim + in_col), Q5.11 16-bit. Loaded from the
// project's own generated/*.hex (4 hex digits == 16-bit, exact width).
module wrom (
    input  wire [2:0]            sel,    // WQ WK WV WO FC1 FC2 LM
    input  wire [11:0]           addr,
    output wire signed [15:0]    wdata
);
    localparam [2:0] S_WQ=3'd0, S_WK=3'd1, S_WV=3'd2, S_WO=3'd3,
                     S_FC1=3'd4, S_FC2=3'd5, S_LM=3'd6;

    (* rom_style="distributed" *) reg signed [15:0] wq  [0:575];
    (* rom_style="distributed" *) reg signed [15:0] wk  [0:575];
    (* rom_style="distributed" *) reg signed [15:0] wv  [0:575];
    (* rom_style="distributed" *) reg signed [15:0] wo  [0:575];
    (* rom_style="distributed" *) reg signed [15:0] fc1 [0:2303];
    (* rom_style="distributed" *) reg signed [15:0] fc2 [0:2303];
    (* rom_style="distributed" *) reg signed [15:0] lm  [0:647];

    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/wq.hex", wq);
        $readmemh("/home/hermes/microgpt_fpga/generated/wk.hex", wk);
        $readmemh("/home/hermes/microgpt_fpga/generated/wv.hex", wv);
        $readmemh("/home/hermes/microgpt_fpga/generated/wo.hex", wo);
        $readmemh("/home/hermes/microgpt_fpga/generated/fc1.hex", fc1);
        $readmemh("/home/hermes/microgpt_fpga/generated/fc2.hex", fc2);
        $readmemh("/home/hermes/microgpt_fpga/generated/lm_head.hex", lm);
    end

    assign wdata =
        (sel == S_WQ)  ? wq[addr[9:0]]  :
        (sel == S_WK)  ? wk[addr[9:0]]  :
        (sel == S_WV)  ? wv[addr[9:0]]  :
        (sel == S_WO)  ? wo[addr[9:0]]  :
        (sel == S_FC1) ? fc1[addr]      :
        (sel == S_FC2) ? fc2[addr]      :
                         lm[addr[9:0]];
endmodule
