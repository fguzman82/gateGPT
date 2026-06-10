// Token sampler. Reads VOCAB logits from vmem. sample_mode=0 -> argmax (greedy).
// sample_mode=1 -> temperature softmax categorical: scaled=logit/temp, softmax via
// max+exp+sum, draw r = LCG(rng) mod total, pick first cumulative > r. Bit-exact
// with tools/fixedpoint.generate. Emits the token and the advanced LCG state.
module sampler #(
    parameter integer VOCAB = 27,
    parameter integer FRAC  = 11
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire        sample_mode,
    input  wire signed [15:0] inv_temp,    // (1/temperature) in Q11
    input  wire [31:0] rng_in,
    input  wire [9:0]  lm_base,
    output reg  [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg  [4:0]  token,
    output reg  [31:0] rng_out,
    output reg         busy,
    output reg         done
);
    localparam [2:0] S_IDLE=0, S_SCALE=1, S_EXP=2, S_MOD=3, S_PICK=4;
    reg [2:0]  st;
    reg [4:0]  i, amax;
    reg signed [15:0] scaled [0:31];
    reg [15:0]        ev [0:31];
    reg signed [15:0] mmax;       // max of scaled logits (for softmax)
    reg signed [15:0] maxlog;     // max of raw logits (for greedy argmax)
    reg [31:0]        total, cum, rval, rngs;

    // temperature scale: sat16( (logit * inv_temp) >> FRAC )
    wire signed [31:0] sc  = $signed(v_rdata) * inv_temp;
    wire signed [31:0] scs = sc >>> FRAC;
    wire signed [15:0] scaled_v =
        (scs >  32'sd32767) ? 16'sd32767 : (scs < -32'sd32768) ? -16'sd32768 : scs[15:0];

    // exp(scaled[i]-max)
    wire signed [16:0] diff = $signed(scaled[i]) - $signed(mmax);
    wire signed [15:0] dz = (diff < -17'sd32768) ? -16'sd32768 : diff[15:0];
    wire signed [15:0] eo;
    exp_unit u_exp (.z(dz), .e(eo));

    // modulo via udiv: q = rng/total, r = rng - q*total
    reg         d_start;
    wire        d_done;
    wire [47:0] d_quo;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num({16'd0, rngs}), .den({16'd0, total}), .busy(), .done(d_done), .quo(d_quo));
    wire [31:0] rmod = rngs - (d_quo[31:0] * total);

    always @(posedge clk) begin
        if (!resetn) begin st <= S_IDLE; busy <= 0; done <= 0; d_start <= 0; end
        else begin
            done <= 0; d_start <= 0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; i <= 0; mmax <= -16'sd32768; maxlog <= -16'sd32768; amax <= 0;
                    total <= 0; v_raddr <= lm_base; st <= S_SCALE;
                end
                S_SCALE: begin
                    scaled[i] <= scaled_v;
                    if (scaled_v > mmax) mmax <= scaled_v;
                    if ($signed(v_rdata) > maxlog) begin maxlog <= v_rdata; amax <= i; end
                    if (i == VOCAB - 1) begin
                        i <= 0;
                        if (sample_mode) st <= S_EXP;
                        else begin token <= amax; rng_out <= rng_in;
                                   busy <= 0; done <= 1; st <= S_IDLE; end
                    end else begin i <= i + 1; v_raddr <= lm_base + i + 1; end
                end
                S_EXP: begin
                    ev[i] <= eo;
                    total <= total + {16'd0, eo};
                    if (i == VOCAB - 1) begin
                        // advance LCG and start the modulo
                        rngs <= rng_in * 32'd1664525 + 32'd1013904223;
                        st <= S_MOD;
                    end else i <= i + 1;
                end
                S_MOD: begin
                    if (!d_start && !d_done) d_start <= 1;
                    if (d_done) begin
                        rval <= rmod; rng_out <= rngs;
                        cum <= 0; i <= 0; token <= VOCAB - 1; st <= S_PICK;
                    end
                end
                S_PICK: begin
                    if ((cum + {16'd0, ev[i]}) > rval) begin
                        token <= i; busy <= 0; done <= 1; st <= S_IDLE;
                    end else if (i == VOCAB - 1) begin
                        busy <= 0; done <= 1; st <= S_IDLE;   // token stays VOCAB-1
                    end else begin
                        cum <= cum + {16'd0, ev[i]}; i <= i + 1;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
