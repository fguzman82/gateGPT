// Fixed-point exp for z <= 0: e = round(exp(z/2048) * 2048) in [0,2048], via a
// 17-entry table (exp(-k)) + linear interpolation. Combinational. Bit-exact with
// tools/fixedpoint.exp_neg_q11. (z >= 0 returns 2048 = exp(0).)
module exp_unit (
    input  wire signed [15:0] z,
    output wire signed [15:0] e
);
    (* rom_style="distributed" *) reg signed [15:0] tab [0:16];
    initial $readmemh("/home/hermes/microgpt_fpga/generated/exp_tab.hex", tab);

    wire signed [17:0] zx = {{2{z[15]}}, z};
    wire [17:0]        u  = -zx;            // |z| when z<=0  (0..32768)
    wire [4:0]         ui = u[15:11];       // integer part (<=16)
    wire [10:0]        uf = u[10:0];        // fractional part (Q11)

    wire signed [15:0] lo = tab[ui[4:0]];
    wire signed [15:0] hi = (ui >= 5'd16) ? 16'sd0 : tab[ui + 5'd1];
    wire signed [31:0] interp = lo + (($signed(hi) - $signed(lo)) * $signed({1'b0, uf}) >>> 11);

    assign e = (z >= 0)        ? 16'sd2048 :
               (ui >= 5'd16)   ? 16'sd0    :
               (interp < 0)    ? 16'sd0    : interp[15:0];
endmodule
