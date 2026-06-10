// Autoregressive name generator: drives the independent microgpt_core, which takes
// the full 16-token context each step (recompute, no shared KV cache). Maintains the
// context shift-register and the RNG state, emits the name as packed bytes plus a
// per-token strobe. Tokens are 0='.', 1..26='a'..'z'; name_buf stores (token-1) so a
// 0..25 -> 'a'..'z' display mapping works.
module name_generator #(
    parameter integer MAX_LEN = 16
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [31:0] seed,
    input  wire signed [15:0] inv_temp,     // (1/temperature) in Q5.11
    input  wire        sample_mode,
    output reg         busy,
    output reg         done,
    output reg  [7:0]  token_out,
    output reg         token_valid,
    output reg  [4:0]  name_len,
    output wire [(MAX_LEN*8)-1:0] name_flat
);
    localparam [1:0] G_IDLE=0, G_FIRE=1, G_WAIT=2, G_DONE=3;
    reg [1:0]  state;
    reg [79:0] ctx;          // 16 tokens x 5 bits (token 0 = leftmost/oldest)
    reg [31:0] rng;
    reg [7:0]  name_buf [0:MAX_LEN-1];
    integer    k;

    reg         core_start;
    wire        core_busy, core_done;
    wire [4:0]  core_tok;
    wire [31:0] core_rng;
    microgpt_core u_core (
        .clk(clk), .resetn(resetn), .start(core_start),
        .ctx_flat(ctx), .sample_mode(sample_mode), .inv_temp(inv_temp), .rng_in(rng),
        .busy(core_busy), .done(core_done), .next_token(core_tok), .rng_out(core_rng));

    genvar g;
    generate for (g = 0; g < MAX_LEN; g = g + 1) begin : GEN_FLAT
        assign name_flat[(g*8) +: 8] = name_buf[g];
    end endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            state <= G_IDLE; busy <= 0; done <= 0; token_valid <= 0; name_len <= 0;
            ctx <= 80'd0; rng <= 32'd1; core_start <= 0; token_out <= 0;
            for (k = 0; k < MAX_LEN; k = k + 1) name_buf[k] <= 8'd0;
        end else begin
            done <= 0; token_valid <= 0; core_start <= 0;
            case (state)
                G_IDLE: begin
                    busy <= 0;
                    if (start) begin
                        busy <= 1; ctx <= 80'd0; rng <= seed; name_len <= 0;
                        for (k = 0; k < MAX_LEN; k = k + 1) name_buf[k] <= 8'd0;
                        state <= G_FIRE;
                    end
                end
                G_FIRE: if (!core_busy && !core_done) begin core_start <= 1; state <= G_WAIT; end
                G_WAIT: if (core_done) begin
                    rng <= core_rng;
                    if (core_tok == 5'd0 || name_len == MAX_LEN[4:0]) state <= G_DONE;
                    else begin
                        name_buf[name_len[3:0]] <= {3'd0, core_tok} - 8'd1;  // 'a'..'z' index
                        token_out   <= {3'd0, core_tok};
                        token_valid <= 1'b1;
                        name_len    <= name_len + 5'd1;
                        ctx         <= {core_tok, ctx[79:5]};
                        state       <= G_FIRE;
                    end
                end
                G_DONE: begin busy <= 0; done <= 1; state <= G_IDLE; end
                default: state <= G_IDLE;
            endcase
        end
    end
endmodule
