// Unit test for the RMSNorm engine vs the Python fixed-point reference (dual-port vmem).
`timescale 1ns/1ps
module tb_norm;
    localparam N = 24;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    reg signed [15:0] tin [0:N-1], texp [0:N-1];

    reg        load;
    reg        tb_we;  reg [9:0] tb_addr;  reg signed [15:0] tb_wdata;

    wire [9:0] na_a, na_b; wire na_wea, na_web; wire signed [15:0] na_wda, na_wdb;
    wire [5:0] ga_a, ga_b; wire signed [15:0] gd_a, gd_b;

    // ports muxed between the TB (load/readback) and norm (run)
    wire        pa_we   = load ? tb_we    : na_wea;
    wire [9:0]  pa_addr = load ? tb_addr  : na_a;
    wire signed [15:0] pa_wd = load ? tb_wdata : na_wda;
    wire        pb_we   = load ? 1'b0     : na_web;
    wire [9:0]  pb_addr = load ? 10'd0    : na_b;
    wire signed [15:0] pb_wd = na_wdb;
    wire signed [15:0] rda, rdb;

    vmem2 #(.AW(10), .DW(16)) u_vmem (.clk(clk),
        .we_a(pa_we), .addr_a(pa_addr), .wdata_a(pa_wd), .rdata_a(rda),
        .we_b(pb_we), .addr_b(pb_addr), .wdata_b(pb_wd), .rdata_b(rdb));

    grom u_grom (.sel(2'd0), .addr_a(ga_a), .addr_b(ga_b), .gdata_a(gd_a), .gdata_b(gd_b));

    wire n_busy, n_done;
    norm #(.N(N), .FRAC(11)) u_norm (
        .clk(clk), .resetn(resetn), .start(start),
        .src_base(10'd0), .dst_base(10'd64), .gain_sel(2'd0),
        .addr_a(na_a), .rd_a(rda), .we_a(na_wea), .wd_a(na_wda),
        .addr_b(na_b), .rd_b(rdb), .we_b(na_web), .wd_b(na_wdb),
        .g_addr_a(ga_a), .g_addr_b(ga_b), .g_rdata_a(gd_a), .g_rdata_b(gd_b),
        .busy(n_busy), .done(n_done));

    integer k, errors;
    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/test_norm_in.hex", tin);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_norm_out.hex", texp);
        errors = 0; load = 1; tb_we = 0; resetn = 0;
        repeat (4) @(posedge clk); resetn = 1;
        for (k = 0; k < N; k = k + 1) begin
            @(negedge clk); tb_we = 1; tb_addr = k[9:0]; tb_wdata = tin[k];
        end
        @(negedge clk); tb_we = 0; load = 0;
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait (n_done); @(posedge clk);
        load = 1;
        for (k = 0; k < N; k = k + 1) begin
            tb_addr = 10'd64 + k[9:0]; @(posedge clk); #1;
            if (rda !== texp[k]) begin
                $display("NORM MISMATCH i=%0d got=%0d exp=%0d", k, $signed(rda), $signed(texp[k]));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("NORM PASS: all %0d outputs match", N);
        else             $display("NORM FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
