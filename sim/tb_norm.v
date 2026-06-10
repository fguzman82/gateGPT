// Unit test for the RMSNorm engine vs the Python fixed-point reference.
`timescale 1ns/1ps
module tb_norm;
    localparam N = 24;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    reg signed [15:0] tin [0:N-1], texp [0:N-1];

    reg        load;
    reg        tb_we;  reg [9:0] tb_waddr, tb_raddr;  reg signed [15:0] tb_wdata;
    wire       n_we;   wire [9:0] n_waddr, n_raddr;   wire signed [15:0] n_wdata;

    wire       v_we    = load ? tb_we    : n_we;
    wire [9:0] v_waddr = load ? tb_waddr : n_waddr;
    wire signed [15:0] v_wdata = load ? tb_wdata : n_wdata;
    wire [9:0] v_raddr = load ? tb_raddr : n_raddr;
    wire signed [15:0] v_rdata;

    vmem #(.AW(10), .DW(16)) u_vmem (.clk(clk), .we(v_we), .waddr(v_waddr),
        .wdata(v_wdata), .raddr(v_raddr), .rdata(v_rdata));

    wire [5:0] g_addr;  wire signed [15:0] g_rdata;
    grom u_grom (.sel(2'd0), .addr(g_addr), .gdata(g_rdata));   // g1

    wire n_busy, n_done;
    norm #(.N(N), .FRAC(11)) u_norm (
        .clk(clk), .resetn(resetn), .start(start),
        .src_base(10'd0), .dst_base(10'd64), .gain_sel(2'd0),
        .v_raddr(n_raddr), .v_rdata(v_rdata),
        .v_we(n_we), .v_waddr(n_waddr), .v_wdata(n_wdata),
        .g_addr(g_addr), .g_rdata(g_rdata),
        .busy(n_busy), .done(n_done));

    integer k, errors;
    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/test_norm_in.hex", tin);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_norm_out.hex", texp);
        errors = 0; load = 1; tb_we = 0; resetn = 0;
        repeat (4) @(posedge clk); resetn = 1;
        for (k = 0; k < N; k = k + 1) begin
            @(negedge clk); tb_we = 1; tb_waddr = k[9:0]; tb_wdata = tin[k];
        end
        @(negedge clk); tb_we = 0; load = 0;
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait (n_done); @(posedge clk);
        load = 1;
        for (k = 0; k < N; k = k + 1) begin
            tb_raddr = 10'd64 + k[9:0]; @(posedge clk); #1;
            if (v_rdata !== texp[k]) begin
                $display("NORM MISMATCH i=%0d got=%0d exp=%0d", k, $signed(v_rdata), $signed(texp[k]));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("NORM PASS: all %0d outputs match", N);
        else             $display("NORM FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
