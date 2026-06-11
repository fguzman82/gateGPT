// RMSNorm gain ROMs (Q5.11), three gains selected by sel (0=g1, 1=g2, 2=gf).
// Emitted as a combinational case (core/gains.vh) -- XST does NOT reliably infer/
// initialize a $readmemh ROM for arrays this small (it ties them to 0), which zeroed
// the gains in hardware and produced garbage. Constants synthesize correctly.
// Dual read (addr_a/addr_b) so the 2-elements/cycle scale pass can fetch both gains.
module grom (
    input  wire [1:0]         sel,
    input  wire [5:0]         addr_a,
    input  wire [5:0]         addr_b,
    output wire signed [15:0] gdata_a,
    output wire signed [15:0] gdata_b
);
`include "/home/hermes/microgpt_fpga/core/gains.vh"
    assign gdata_a = gain_lut(sel, addr_a[4:0]);
    assign gdata_b = gain_lut(sel, addr_b[4:0]);
endmodule
