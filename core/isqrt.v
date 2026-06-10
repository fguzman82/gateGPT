// Unsigned integer square root: root = floor(sqrt(radicand)), W-bit radicand ->
// W/2-bit root. Classic bit-pair (non-restoring) algorithm, W/2 cycles. Matches
// Python math.isqrt. Used by RMSNorm. Synthesizable.
module isqrt #(
    parameter integer W = 48        // radicand width (even)
) (
    input  wire         clk,
    input  wire         resetn,
    input  wire         start,
    input  wire [W-1:0] radicand,
    output reg          busy,
    output reg          done,
    output reg  [W/2-1:0] root
);
    reg [W-1:0]  op;          // remaining radicand
    reg [W-1:0]  res;         // result accumulator
    reg [W-1:0]  bitm;        // current power-of-four
    reg          st;
    reg [7:0]    cnt;

    wire [W-1:0] resbit = res + bitm;

    always @(posedge clk) begin
        if (!resetn) begin
            busy <= 1'b0; done <= 1'b0; st <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!st) begin
                if (start) begin
                    op <= radicand; res <= {W{1'b0}};
                    bitm <= {2'b01, {(W-2){1'b0}}};   // 1 << (W-2): top even bit
                    cnt <= (W/2) - 1; busy <= 1'b1; st <= 1'b1;
                end
            end else begin
                if (op >= resbit) begin
                    op  <= op - resbit;
                    res <= (res >> 1) + bitm;
                end else begin
                    res <= res >> 1;
                end
                bitm <= bitm >> 2;
                if (cnt == 0) begin
                    // res after this cycle holds floor(sqrt); expose via a final reg
                    root <= ((op >= resbit) ? ((res >> 1) + bitm) : (res >> 1)) >> 0;
                    busy <= 1'b0; done <= 1'b1; st <= 1'b0;
                end else cnt <= cnt - 8'd1;
            end
        end
    end
endmodule
