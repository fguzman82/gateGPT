// Fixed-point exp for z <= 0: e = round(exp(z/2048) * 2048) in [0,2048], via a
// 17-entry table (exp(-k)) + linear interpolation. Bit-exact with
// tools/fixedpoint.exp_neg_q11. (z >= 0 returns 2048 = exp(0).)
//
// PIPELINED (latency 1): the table lookup + decode is registered, the interpolation
// multiply runs the next cycle -> shorter combinational path (this chain was the
// post-pipeline Fmax limiter). Callers feed z and read e one cycle later (it replaces
// the dz_r register they used to keep before a combinational exp_unit).
module exp_unit (
    input  wire               clk,
    input  wire signed [15:0] z,
    output wire signed [15:0] e
);
    (* rom_style="distributed" *) reg signed [15:0] tab [0:16];
    initial $readmemh("/home/hermes/microgpt_fpga/generated/exp_tab.hex", tab);

    // stage 1 (combinational): |z|, table lookup, decode
    wire signed [17:0] zx = {{2{z[15]}}, z};
    wire [17:0]        u  = -zx;            // |z| when z<=0  (0..32768)
    wire [4:0]         ui = u[15:11];       // integer part (<=16)
    wire [10:0]        uf = u[10:0];        // fractional part (Q11)
    wire signed [15:0] lo = tab[ui[4:0]];
    wire signed [15:0] hi = (ui >= 5'd16) ? 16'sd0 : tab[ui + 5'd1];

    // pipeline register (cut between the ROM lookup and the interpolation multiply)
    reg signed [15:0] lo_r, hi_r;
    reg [10:0]        uf_r;
    reg               pos_r, big_r;
    always @(posedge clk) begin
        lo_r <= lo; hi_r <= hi; uf_r <= uf;
        pos_r <= (z >= 0); big_r <= (ui >= 5'd16);
    end

    // stage 2 (combinational from the registered values): interpolate + clamp
    wire signed [31:0] interp = lo_r + (($signed(hi_r) - $signed(lo_r)) * $signed({1'b0, uf_r}) >>> 11);
    assign e = pos_r ? 16'sd2048 : big_r ? 16'sd0 : (interp < 0) ? 16'sd0 : interp[15:0];
endmodule
