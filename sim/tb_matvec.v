// Unit test for the matvec engine: load a known activation vector, run wq, and
// compare the 24 outputs against the Python fixed-point reference (dual-port vmem).
`timescale 1ns/1ps
module tb_matvec;
    localparam N = 24;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    reg signed [15:0] tin  [0:N-1];
    reg signed [15:0] texp [0:N-1];

    reg        load;          // 1 = TB owns vmem ports
    reg        tb_we;
    reg  [9:0]  tb_addr;
    reg  signed [15:0] tb_wdata;

    wire [9:0] mv_aa, mv_ab; wire mv_wea, mv_web; wire signed [15:0] mv_wda, mv_wdb;

    // ports muxed between the TB (load/readback) and matvec (run)
    wire        pa_we   = load ? tb_we   : mv_wea;
    wire [9:0]  pa_addr = load ? tb_addr : mv_aa;
    wire signed [15:0] pa_wd = load ? tb_wdata : mv_wda;
    wire        pb_we   = load ? 1'b0    : mv_web;
    wire [9:0]  pb_addr = load ? 10'd0   : mv_ab;
    wire signed [15:0] pb_wd = mv_wdb;
    wire signed [15:0] rda, rdb;

    vmem2 #(.AW(10), .DW(16)) u_vmem (.clk(clk),
        .we_a(pa_we), .addr_a(pa_addr), .wdata_a(pa_wd), .rdata_a(rda),
        .we_b(pb_we), .addr_b(pb_addr), .wdata_b(pb_wd), .rdata_b(rdb));

    wire [11:0] w_addr;
    wire [767:0] w_rdata;                                     // 24 lanes x 16-bit x 2 cols
    wrom u_wrom (.sel(3'd0), .addr(w_addr), .wdata(w_rdata));  // sel=WQ

    wire mv_busy, mv_done;
    matvec u_mv (
        .clk(clk), .resetn(resetn), .start(start),
        .wsel(3'd0), .in_dim(7'd24), .out_dim(7'd24),
        .act_base(10'd0), .dst_base(10'd64), .descale(5'd11),
        .addr_a(mv_aa), .rd_a(rda), .we_a(mv_wea), .wd_a(mv_wda),
        .addr_b(mv_ab), .rd_b(rdb), .we_b(mv_web), .wd_b(mv_wdb),
        .w_addr(w_addr), .w_rdata(w_rdata),
        .busy(mv_busy), .done(mv_done)
    );

    integer k, errors;
    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/test_in.hex", tin);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_wq.hex", texp);
        errors = 0;
        load = 1; tb_we = 0; resetn = 0;
        repeat (4) @(posedge clk);
        resetn = 1;
        // load activation vector into vmem[0..23]
        for (k = 0; k < N; k = k + 1) begin
            @(negedge clk); tb_we = 1; tb_addr = k[9:0]; tb_wdata = tin[k];
        end
        @(negedge clk); tb_we = 0;
        // run matvec
        load = 0;
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait (mv_done);
        @(posedge clk);
        // read back vmem[64..64+23] and compare
        load = 1;
        for (k = 0; k < N; k = k + 1) begin
            tb_addr = 10'd64 + k[9:0];
            @(posedge clk); #1;
            if (rda !== texp[k]) begin
                $display("MISMATCH o=%0d got=%0d exp=%0d", k, $signed(rda), $signed(texp[k]));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("MATVEC PASS: all %0d outputs match", N);
        else             $display("MATVEC FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
