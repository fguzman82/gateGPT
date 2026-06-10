// Single-position multi-head attention. The query is the last position; it attends
// to all BLOCK cached positions (causal-correct for the last token, no mask). Per
// head: score = scale * (q . k); softmax via max-subtract + exp + sum; output =
// sum(e*v) / sum(e) (truncating divide). Bit-exact with QModel.logits_last.
// Reads q/K/V from vmem regions, writes the attention output vector to O_BASE.
module attn #(
    parameter integer N_EMBED  = 24,
    parameter integer N_HEAD   = 4,
    parameter integer HEAD_DIM = 6,
    parameter integer BLOCK    = 16,
    parameter integer FRAC     = 11
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire signed [15:0] attn_scale,
    input  wire [9:0]  q_base,
    input  wire [9:0]  k_base,
    input  wire [9:0]  v_base,
    input  wire [9:0]  o_base,
    output reg  [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  signed [15:0] v_wdata,
    output reg         busy,
    output reg         done
);
    localparam [2:0] P_IDLE=0, P_QLOAD=1, P_SCORE=2, P_EXP=3, P_WSUM=4, P_DIV=5, P_NEXTH=6;
    reg [2:0]  ph;
    reg [3:0]  h;             // head index
    reg [4:0]  s;             // context position
    reg [3:0]  d;             // within-head dim
    reg [9:0]  hbase;         // h*HEAD_DIM
    reg [9:0]  soff;          // s*N_EMBED

    reg signed [15:0] qreg [0:15];   // current head's query slice (HEAD_DIM used)
    reg signed [15:0] score [0:15];  // per-position scores
    reg [15:0]        ev [0:15];     // per-position exp weights
    reg signed [15:0] mmax;
    reg [31:0]        sum_e;
    reg signed [47:0] acc;

    // score finalize: scale * sat16( (q.k including the current/last term) >> FRAC )
    wire signed [47:0] acc_full = acc + $signed(qreg[d]) * $signed(v_rdata);
    wire signed [47:0] accf_sh = acc_full >>> FRAC;
    wire signed [15:0] scf1 =
        (accf_sh >  48'sd32767) ? 16'sd32767 : (accf_sh < -48'sd32768) ? -16'sd32768 : accf_sh[15:0];
    wire signed [31:0] scfsc = scf1 * attn_scale;
    wire signed [31:0] scfsc_sh = scfsc >>> FRAC;
    wire signed [15:0] sc2_next =
        (scfsc_sh >  32'sd32767) ? 16'sd32767 : (scfsc_sh < -32'sd32768) ? -16'sd32768 : scfsc_sh[15:0];

    // exp(score[s]-max)
    wire signed [16:0] diff = $signed(score[s]) - $signed(mmax);
    wire signed [15:0] dz = (diff < -17'sd32768) ? -16'sd32768 : diff[15:0];
    wire signed [15:0] eo;
    exp_unit u_exp (.z(dz), .e(eo));

    // weighted-sum division: |num| / sum_e, sign of num
    reg         d_start;
    wire        d_done;
    wire [47:0] d_quo;
    wire [47:0] num_abs = acc[47] ? (~acc + 48'd1) : acc;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num(num_abs), .den({31'd0, sum_e[16:0]}), .busy(), .done(d_done), .quo(d_quo));
    wire signed [47:0] q_signed = acc[47] ? -$signed(d_quo) : $signed(d_quo);
    wire signed [15:0] o_sat =
        (q_signed >  48'sd32767) ? 16'sd32767 : (q_signed < -48'sd32768) ? -16'sd32768 : q_signed[15:0];

    always @(posedge clk) begin
        if (!resetn) begin
            ph <= P_IDLE; busy <= 0; done <= 0; v_we <= 0; d_start <= 0;
        end else begin
            done <= 0; v_we <= 0; d_start <= 0;
            case (ph)
                P_IDLE: if (start) begin
                    busy <= 1; h <= 0; hbase <= 0; d <= 0; v_raddr <= q_base;
                    ph <= P_QLOAD;
                end
                // load this head's query slice into qreg[0..HEAD_DIM-1]
                P_QLOAD: begin
                    qreg[d] <= v_rdata;
                    if (d == HEAD_DIM - 1) begin
                        d <= 0; s <= 0; soff <= 0; acc <= 0;
                        mmax <= -16'sd32768;
                        v_raddr <= k_base + 0 + hbase + 0;   // k[s=0,d=0]
                        ph <= P_SCORE;
                    end else begin
                        d <= d + 1; v_raddr <= q_base + hbase + (d + 1);
                    end
                end
                // scores: per s, MAC q.k over HEAD_DIM, then scale + track max
                P_SCORE: begin
                    acc <= acc + $signed(qreg[d]) * $signed(v_rdata);
                    if (d == HEAD_DIM - 1) begin
                        // finalize score[s] using the value that includes this term
                        // (computed next cycle in P_SCORE2-style: fold here)
                        score[s] <= sc2_next;
                        if (sc2_next > mmax) mmax <= sc2_next;
                        d <= 0; acc <= 0;
                        if (s == BLOCK - 1) begin
                            s <= 0; sum_e <= 0; ph <= P_EXP;
                        end else begin
                            s <= s + 1; soff <= soff + N_EMBED;
                            v_raddr <= k_base + (soff + N_EMBED) + hbase + 0;
                        end
                    end else begin
                        d <= d + 1;
                        v_raddr <= k_base + soff + hbase + (d + 1);
                    end
                end
                // exp weights + running sum
                P_EXP: begin
                    ev[s] <= eo;
                    sum_e <= sum_e + {15'd0, eo};
                    if (s == BLOCK - 1) begin
                        s <= 0; d <= 0; acc <= 0;
                        v_raddr <= v_base + 0 + hbase + 0;   // v[s=0,d=0]
                        soff <= 0;
                        ph <= P_WSUM;
                    end else s <= s + 1;
                end
                // weighted sum: per d, sum_s ev[s]*v[s,d]
                P_WSUM: begin
                    acc <= acc + $signed({1'b0, ev[s]}) * $signed(v_rdata);
                    if (s == BLOCK - 1) begin
                        d_start <= 1; ph <= P_DIV;            // divide acc / sum_e
                    end else begin
                        s <= s + 1; soff <= soff + N_EMBED;
                        v_raddr <= v_base + (soff + N_EMBED) + hbase + d;
                    end
                end
                P_DIV: if (d_done) begin
                    v_we <= 1; v_waddr <= o_base + hbase + d; v_wdata <= o_sat;
                    acc <= 0; s <= 0; soff <= 0;
                    if (d == HEAD_DIM - 1) ph <= P_NEXTH;
                    else begin
                        d <= d + 1;
                        v_raddr <= v_base + 0 + hbase + (d + 1);
                        ph <= P_WSUM;
                    end
                end
                P_NEXTH: begin
                    if (h == N_HEAD - 1) begin busy <= 0; done <= 1; ph <= P_IDLE; end
                    else begin
                        h <= h + 1; hbase <= hbase + HEAD_DIM; d <= 0;
                        v_raddr <= q_base + (hbase + HEAD_DIM) + 0;
                        ph <= P_QLOAD;
                    end
                end
                default: ph <= P_IDLE;
            endcase
        end
    end
endmodule
