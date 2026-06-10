// Unsigned iterative divider: quo = num / den (floor), W-bit. Restoring, MSB-first,
// W cycles. den==0 yields all-ones (guard). Shared by RMSNorm (ss/N, 2^22/r) and
// attention (weighted-sum / softmax-sum). Synthesizable (no '/' operator).
module udiv #(
    parameter integer W = 48
) (
    input  wire         clk,
    input  wire         resetn,
    input  wire         start,
    input  wire [W-1:0] num,
    input  wire [W-1:0] den,
    output reg          busy,
    output reg          done,
    output reg  [W-1:0] quo
);
    reg [W-1:0]  ncur, q, dreg;
    reg [W:0]    rem;
    reg [7:0]    cnt;
    reg          st;

    wire [W:0] rshift = {rem[W-1:0], ncur[W-1]};      // bring down next num bit (MSB-first)
    wire       qbit   = (rshift >= {1'b0, dreg});
    wire [W:0] remn   = qbit ? (rshift - {1'b0, dreg}) : rshift;

    always @(posedge clk) begin
        if (!resetn) begin
            busy <= 1'b0; done <= 1'b0; st <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!st) begin
                if (start) begin
                    ncur <= num; dreg <= (den == 0) ? {W{1'b1}} : den;
                    rem <= {(W+1){1'b0}}; q <= {W{1'b0}}; cnt <= W - 1;
                    busy <= 1'b1; st <= 1'b1;
                end
            end else begin
                rem  <= remn;
                q    <= {q[W-2:0], qbit};
                ncur <= {ncur[W-2:0], 1'b0};
                if (cnt == 0) begin
                    quo <= {q[W-2:0], qbit};
                    busy <= 1'b0; done <= 1'b1; st <= 1'b0;
                end else cnt <= cnt - 8'd1;
            end
        end
    end
endmodule
