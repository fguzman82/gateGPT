// RMSNorm gain ROMs (Q5.11), three gains selected by sel (0=g1, 1=g2, 2=gf).
// Emitted as a combinational case (core/gains.vh) -- XST does NOT reliably infer/
// initialize a $readmemh ROM for arrays this small (it ties them to 0), which zeroed
// the gains in hardware and produced garbage. Constants synthesize correctly.
module grom (
    input  wire [1:0]         sel,
    input  wire [5:0]         addr,
    output wire signed [15:0] gdata
);
`include "/home/hermes/microgpt_fpga/core/gains.vh"
    assign gdata = gain_lut(sel, addr[4:0]);
endmodule
