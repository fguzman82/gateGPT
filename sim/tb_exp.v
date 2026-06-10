// Unit test for exp_unit vs the Python reference (sweep of z values).
`timescale 1ns/1ps
module tb_exp;
    localparam M = 102;       // upper bound on cases; extra entries stay x and are skipped
    reg signed [15:0] zs [0:M-1], es [0:M-1];
    reg signed [15:0] zin;
    wire signed [15:0] eo;
    exp_unit u_exp (.z(zin), .e(eo));

    integer k, errors, n;
    initial begin
        for (k = 0; k < M; k = k + 1) begin zs[k] = 16'shxxxx; es[k] = 16'shxxxx; end
        $readmemh("/home/hermes/microgpt_fpga/generated/test_exp_z.hex", zs);
        $readmemh("/home/hermes/microgpt_fpga/generated/test_exp_e.hex", es);
        errors = 0; n = 0;
        for (k = 0; k < M; k = k + 1) begin
            if (zs[k] !== 16'shxxxx) begin
                zin = zs[k]; #1;
                n = n + 1;
                if (eo !== es[k]) begin
                    $display("EXP MISMATCH z=%0d got=%0d exp=%0d", zs[k], $signed(eo), $signed(es[k]));
                    errors = errors + 1;
                end
            end
        end
        if (errors == 0) $display("EXP PASS: all %0d cases match", n);
        else             $display("EXP FAIL: %0d/%0d mismatches", errors, n);
        $finish;
    end
endmodule
