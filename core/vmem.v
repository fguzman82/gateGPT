// Activation scratchpad: block RAM, 1 write port + 1 *registered* read port.
// The registered read (rdata valid one cycle after raddr) keeps the read-address
// fanout tiny (a single BRAM, vs ~256 LUT-RAM primitives for distributed RAM),
// which was the dominant routing delay. Actuators present the address one cycle
// ahead and consume the data the next cycle (read-ahead).
module vmem #(
    parameter integer AW = 10,   // address width (1024 words covers all scratch)
    parameter integer DW = 16
) (
    input  wire                 clk,
    input  wire                 we,
    input  wire [AW-1:0]        waddr,
    input  wire signed [DW-1:0] wdata,
    input  wire [AW-1:0]        raddr,
    output reg  signed [DW-1:0] rdata
);
    (* ram_style = "block" *) reg signed [DW-1:0] mem [0:(1<<AW)-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];          // registered read (1-cycle latency)
    end
endmodule
