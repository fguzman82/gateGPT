// Unit test for udiv and isqrt against the Python reference values.
`timescale 1ns/1ps
module tb_mathops;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    // udiv
    reg         d_start = 0;
    reg  [47:0] d_num, d_den;
    wire        d_busy, d_done;
    wire [47:0] d_quo;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num(d_num), .den(d_den), .busy(d_busy), .done(d_done), .quo(d_quo));

    // isqrt
    reg         s_start = 0;
    reg  [47:0] s_rad;
    wire        s_busy, s_done;
    wire [23:0] s_root;
    isqrt #(.W(48)) u_sqrt (.clk(clk), .resetn(resetn), .start(s_start),
        .radicand(s_rad), .busy(s_busy), .done(s_done), .root(s_root));

    integer errors = 0;

    task chk_div(input [47:0] a, input [47:0] b, input [47:0] exp);
        begin
            @(negedge clk); d_num = a; d_den = b; d_start = 1;
            @(negedge clk); d_start = 0;
            wait (d_done); #1;
            if (d_quo !== exp) begin
                $display("DIV FAIL %0d/%0d = %0d exp %0d", a, b, d_quo, exp); errors = errors + 1;
            end
        end
    endtask

    task chk_sqrt(input [47:0] n, input [23:0] exp);
        begin
            @(negedge clk); s_rad = n; s_start = 1;
            @(negedge clk); s_start = 0;
            wait (s_done); #1;
            if (s_root !== exp) begin
                $display("SQRT FAIL %0d -> %0d exp %0d", n, s_root, exp); errors = errors + 1;
            end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk); resetn = 1; @(posedge clk);
        chk_div(10962944, 24, 456789);
        chk_div(4194304, 675, 6213);
        chk_div(123456789, 1000, 123456);
        chk_div(65535, 256, 255);
        chk_div(1, 1, 1);
        chk_div(0, 7, 0);
        chk_sqrt(456789, 675);
        chk_sqrt(4194304, 2048);
        chk_sqrt(100, 10);
        chk_sqrt(10, 3);
        chk_sqrt(0, 0);
        chk_sqrt(1, 1);
        chk_sqrt(999999999, 31622);
        if (errors == 0) $display("MATHOPS PASS");
        else             $display("MATHOPS FAIL: %0d errors", errors);
        $finish;
    end
endmodule
