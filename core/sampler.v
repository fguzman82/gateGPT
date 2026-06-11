// Token sampler. Reads VOCAB logits from vmem (registered read -> read-ahead).
// sample_mode=0 -> argmax (greedy). sample_mode=1 -> temperature softmax categorical:
// scaled=logit/temp, softmax via max+exp+sum, draw r = LCG(rng) mod total, pick first
// cumulative > r. Bit-exact with tools/fixedpoint.generate. Emits token + advanced LCG.
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
    output wire [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg  [4:0]  token,
    output reg  [31:0] rng_out,
    output reg         busy,
    output reg         done
);
    localparam [2:0] S_IDLE=0, S_SCALE=1, S_EXP=2, S_MOD=3, S_PICK=4;
    reg [2:0]  st;
    reg [4:0]  i, fi, fi_d, fi_d2, amax;
    reg        feeding, vld, vld2;
    reg signed [15:0] logit_r;    // logit registered before the variable temp multiply
    reg signed [15:0] scaled [0:31];
    reg [15:0]        ev [0:31];
    reg signed [15:0] mmax;       // max of scaled logits (for softmax)
    reg signed [15:0] maxlog;     // max of raw logits (for greedy argmax)
    reg [31:0]        total, cum, rval, rngs;

    assign v_raddr = lm_base + {5'd0, fi};

    // temperature scale: sat16( (logit * inv_temp) >> FRAC ). inv_temp is now variable
    // (rotary-selected), so register the logit first -> the 16x16 multiply starts from a
    // fabric register, keeping the BRAM-output net off the critical path.
    wire signed [31:0] sc  = $signed(logit_r) * inv_temp;
    wire signed [31:0] scs = sc >>> FRAC;
    wire signed [15:0] scaled_v =
        (scs >  32'sd32767) ? 16'sd32767 : (scs < -32'sd32768) ? -16'sd32768 : scs[15:0];

    // exp(scaled[fi]-max); exp_unit registers its input internally (latency 1)
    wire signed [16:0] diff = $signed(scaled[fi]) - $signed(mmax);
    wire signed [15:0] dz = (diff < -17'sd32768) ? -16'sd32768 : diff[15:0];
    wire signed [15:0] eo;
    exp_unit u_exp (.clk(clk), .z(dz), .e(eo));

    // modulo via udiv: the divider already produces the remainder (rng mod total),
    // so no separate q*total multiply is needed.
    reg         d_start;
    wire        d_done;
    wire [47:0] d_quo, d_rem;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num({16'd0, rngs}), .den({16'd0, total}), .busy(), .done(d_done),
        .quo(d_quo), .rem_out(d_rem));

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; d_start <= 0; feeding <= 0; vld <= 0;
        end else begin
            done <= 0; d_start <= 0;
            fi_d <= fi; vld <= feeding;
            fi_d2 <= fi_d; vld2 <= vld; logit_r <= v_rdata;     // scale-pass pipeline stage
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; mmax <= -16'sd32768; maxlog <= -16'sd32768; amax <= 0;
                    total <= 0; feeding <= 1; st <= S_SCALE;
                end
                S_SCALE: begin
                    if (vld2) begin                              // consume the registered logit
                        scaled[fi_d2] <= scaled_v;
                        if (scaled_v > mmax) mmax <= scaled_v;
                        if ($signed(logit_r) > maxlog) begin maxlog <= logit_r; amax <= fi_d2; end
                    end
                    if (feeding) begin
                        if (fi == VOCAB - 1) feeding <= 0;
                        else fi <= fi + 1;
                    end
                    if (vld2 && fi_d2 == VOCAB - 1) begin
                        if (sample_mode) begin fi <= 0; feeding <= 1; st <= S_EXP; end
                        else begin token <= amax; rng_out <= rng_in;
                                   busy <= 0; done <= 1; st <= S_IDLE; end
                    end
                end
                S_EXP: begin
                    if (vld) begin                           // exp_unit output -> accumulate
                        ev[fi_d] <= eo;
                        total <= total + {16'd0, eo};
                        if (fi_d == VOCAB - 1) begin
                            rngs <= rng_in * 32'd1664525 + 32'd1013904223;
                            st <= S_MOD;
                        end
                    end
                    if (feeding) begin
                        if (fi == VOCAB - 1) feeding <= 0;
                        else fi <= fi + 1;
                    end
                end
                S_MOD: begin
                    if (!d_start && !d_done) d_start <= 1;
                    if (d_done) begin
                        rval <= d_rem[31:0]; rng_out <= rngs;
                        cum <= 0; i <= 0; token <= VOCAB - 1; st <= S_PICK;
                    end
                end
                S_PICK: begin
                    if ((cum + {16'd0, ev[i]}) > rval) begin
                        token <= i; busy <= 0; done <= 1; st <= S_IDLE;
                    end else if (i == VOCAB - 1) begin
                        busy <= 0; done <= 1; st <= S_IDLE;
                    end else begin
                        cum <= cum + {16'd0, ev[i]}; i <= i + 1;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
