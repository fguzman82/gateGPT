// Activation scratchpad: block RAM, TRUE dual-port. Two independent ports (A, B); each
// port registers its read every cycle (rdata valid one cycle after addr) and writes when
// its we is asserted. This gives two memory accesses per cycle -- either two reads (e.g.
// RMSNorm sum-of-squares, attention score/weighted-sum), two writes (RMSNorm scale,
// matvec writeback), or the legacy one-read + one-write. Callers must not write the same
// address on both ports in one cycle. 1024x16 fits one RAMB18 in true-dual-port mode.
module vmem2 #(
    parameter integer AW = 10,
    parameter integer DW = 16
) (
    input  wire                 clk,
    // port A
    input  wire                 we_a,
    input  wire [AW-1:0]        addr_a,
    input  wire signed [DW-1:0] wdata_a,
    output reg  signed [DW-1:0] rdata_a,
    // port B
    input  wire                 we_b,
    input  wire [AW-1:0]        addr_b,
    input  wire signed [DW-1:0] wdata_b,
    output reg  signed [DW-1:0] rdata_b
);
    (* ram_style = "block" *) reg signed [DW-1:0] mem [0:(1<<AW)-1];
    always @(posedge clk) begin
        if (we_a) mem[addr_a] <= wdata_a;
        rdata_a <= mem[addr_a];
        if (we_b) mem[addr_b] <= wdata_b;
        rdata_b <= mem[addr_b];
    end
endmodule
