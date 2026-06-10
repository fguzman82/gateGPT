// Real tokens/second meter. Counts token_valid strobes over a 1-second window and
// latches the count once per second, as 5 BCD digits (0..99999). Counting directly
// in BCD avoids any binary->decimal division (XST only divides by powers of 2).
module tok_meter #(
    parameter integer CLK_HZ = 50_000_000
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        token_valid,    // 1-cycle pulse per generated token
    output reg  [19:0] tok_bcd         // {d4,d3,d2,d1,d0}, each 4-bit, latched 1/sec
);
    reg [3:0]  d0, d1, d2, d3, d4;      // ones .. ten-thousands
    reg [31:0] sec_timer;
    wire at_cap = (d4==9) && (d3==9) && (d2==9) && (d1==9) && (d0==9);

    always @(posedge clk) begin
        if (!resetn) begin
            d0<=0; d1<=0; d2<=0; d3<=0; d4<=0; sec_timer<=0; tok_bcd<=20'd0;
        end else if (sec_timer >= (CLK_HZ - 1)) begin
            tok_bcd   <= {d4, d3, d2, d1, d0};   // publish this second
            d0<=0; d1<=0; d2<=0; d3<=0; d4<=0;
            sec_timer <= 32'd0;
        end else begin
            sec_timer <= sec_timer + 32'd1;
            if (token_valid && !at_cap) begin    // BCD increment with ripple carry
                if (d0 != 9) d0 <= d0 + 4'd1;
                else begin
                    d0 <= 4'd0;
                    if (d1 != 9) d1 <= d1 + 4'd1;
                    else begin
                        d1 <= 4'd0;
                        if (d2 != 9) d2 <= d2 + 4'd1;
                        else begin
                            d2 <= 4'd0;
                            if (d3 != 9) d3 <= d3 + 4'd1;
                            else begin d3 <= 4'd0; d4 <= d4 + 4'd1; end
                        end
                    end
                end
            end
        end
    end

endmodule
