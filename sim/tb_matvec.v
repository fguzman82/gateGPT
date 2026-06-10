// Unit test for the matvec engine: load a known activation vector, run wq, and
// compare the 24 outputs against the Python fixed-point reference.
`timescale 1ns/1ps
module tb_matvec;
    localparam N = 24;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    // test vectors from the reference
    reg signed [15:0] tin  [0:N-1];
    reg signed [15:0] texp [0:N-1];

    // vmem port muxing: TB loads/reads, matvec drives during compute
    reg         load;          // 1 = TB owns vmem ports
    reg         tb_we;
    reg  [9:0]  tb_waddr, tb_raddr;
    reg  signed [15:0] tb_wdata;

    wire        mv_we;
    wire [9:0]  mv_waddr, mv_raddr;
    wire signed [15:0] mv_wdata;

    wire        v_we    = load ? tb_we    : mv_we;
    wire [9:0]  v_waddr = load ? tb_waddr : mv_waddr;
    wire signed [15:0] v_wdata = load ? tb_wdata : mv_wdata;
    wire [9:0]  v_raddr = load ? tb_raddr : mv_raddr;
    wire signed [15:0] v_rdata;

    vmem #(.AW(10), .DW(16)) u_vmem (
        .clk(clk), .we(v_we), .waddr(v_waddr), .wdata(v_wdata),
        .raddr(v_raddr), .rdata(v_rdata)
    );

    wire [11:0] w_addr;
    wire signed [15:0] w_rdata;
    wrom u_wrom (.sel(3'd0), .addr(w_addr), .wdata(w_rdata));  // sel=WQ

    wire mv_busy, mv_done;
    matvec u_mv (
        .clk(clk), .resetn(resetn), .start(start),
        .wsel(3'd0), .in_dim(7'd24), .out_dim(7'd24),
        .act_base(10'd0), .dst_base(10'd64), .descale(5'd11),
        .v_raddr(mv_raddr), .v_rdata(v_rdata),
        .v_we(mv_we), .v_waddr(mv_waddr), .v_wdata(mv_wdata),
        .w_addr(w_addr), .w_rdata(w_rdata),
        .busy(mv_busy), .done(mv_done)
    );

    // debug: print acc at each S_WB (st==2), and the done pulse
    always @(posedge clk) if (!load && u_mv.st == 2'd2)
        $display("[%0t] WB o=%0d i=%0d acc=%0d sat=%0d",
                 $time, u_mv.o, u_mv.i, u_mv.acc, $signed(u_mv.sat));
    always @(posedge clk) if (mv_done) $display("[%0t] DONE", $time);

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
            @(negedge clk); tb_we = 1; tb_waddr = k[9:0]; tb_wdata = tin[k];
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
            tb_raddr = 10'd64 + k[9:0];
            @(posedge clk); #1;
            if (v_rdata !== texp[k]) begin
                $display("MISMATCH o=%0d got=%0d exp=%0d", k, $signed(v_rdata), $signed(texp[k]));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("MATVEC PASS: all %0d outputs match", N);
        else             $display("MATVEC FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
