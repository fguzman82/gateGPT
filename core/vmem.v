// Activation scratchpad: distributed RAM, 1 sync write port + 1 async read port.
// Holds every intermediate vector (residual, normed, Q, K/V cache, attn out, MLP
// hidden, logits). Async read lets the MAC stream activations while a result from
// the previous output row is written back the same cycle.
module vmem #(
    parameter integer AW = 10,   // address width (1024 words covers all scratch)
    parameter integer DW = 16
) (
    input  wire                 clk,
    input  wire                 we,
    input  wire [AW-1:0]        waddr,
    input  wire signed [DW-1:0] wdata,
    input  wire [AW-1:0]        raddr,
    output wire signed [DW-1:0] rdata
);
    (* ram_style = "distributed" *) reg signed [DW-1:0] mem [0:(1<<AW)-1];
    always @(posedge clk) if (we) mem[waddr] <= wdata;
    assign rdata = mem[raddr];
endmodule
