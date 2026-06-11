// Unit test for the attention engine vs the Python reference.
`timescale 1ns/1ps
module tb_attn;
    localparam N = 24, KV = 16*24;
    localparam QB = 0, KB = 32, VB = 448, OB = 864;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    reg signed [15:0] tq [0:N-1], tk [0:KV-1], tv [0:KV-1], texp [0:N-1];

    reg        load;
    reg        tb_we;  reg [9:0] tb_waddr, tb_raddr;  reg signed [15:0] tb_wdata;
    wire       a_we;   wire [9:0] a_waddr, a_raddr;    wire signed [15:0] a_wdata;
    wire       v_we    = load ? tb_we    : a_we;
    wire [9:0] v_waddr = load ? tb_waddr : a_waddr;
    wire signed [15:0] v_wdata = load ? tb_wdata : a_wdata;
    wire [9:0] v_raddr = load ? tb_raddr : a_raddr;
    wire signed [15:0] v_rdata;
    vmem #(.AW(10), .DW(16)) u_vmem (.clk(clk), .we(v_we), .waddr(v_waddr),
        .wdata(v_wdata), .raddr(v_raddr), .rdata(v_rdata));

    wire a_busy, a_done;
    attn #(.N_EMBED(24), .N_HEAD(4), .HEAD_DIM(6), .BLOCK(16), .FRAC(11)) u_attn (
        .clk(clk), .resetn(resetn), .start(start), .attn_scale(16'sd836), .ctx_len(5'd16),
        .q_base(QB), .k_base(KB), .v_base(VB), .o_base(OB),
        .v_raddr(a_raddr), .v_rdata(v_rdata),
        .v_we(a_we), .v_waddr(a_waddr), .v_wdata(a_wdata),
        .busy(a_busy), .done(a_done));

    integer k, errors;
    task wload(input [9:0] base, input integer cnt, input integer which);
        integer j;
        begin
            for (j = 0; j < cnt; j = j + 1) begin
                @(negedge clk); tb_we = 1; tb_waddr = base + j[9:0];
                tb_wdata = (which==0) ? tq[j] : (which==1) ? tk[j] : tv[j];
            end
        end
    endtask

    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/test_attn_q.hex", tq);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_attn_k.hex", tk);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_attn_v.hex", tv);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_attn_out.hex", texp);
        errors = 0; load = 1; tb_we = 0; resetn = 0;
        repeat (4) @(posedge clk); resetn = 1;
        wload(QB, N, 0); wload(KB, KV, 1); wload(VB, KV, 2);
        @(negedge clk); tb_we = 0; load = 0;
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait (a_done); @(posedge clk);
        load = 1;
        for (k = 0; k < N; k = k + 1) begin
            tb_raddr = OB + k[9:0]; @(posedge clk); #1;
            if (v_rdata !== texp[k]) begin
                $display("ATTN MISMATCH i=%0d got=%0d exp=%0d", k, $signed(v_rdata), $signed(texp[k]));
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("ATTN PASS: all %0d outputs match", N);
        else             $display("ATTN FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
