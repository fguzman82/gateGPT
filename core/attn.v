// Single-position multi-head attention with a REGISTERED vmem read (1-cycle latency,
// read-ahead). Per head: load the query slice, score = scale*(q.k), softmax via
// max-subtract + exp + sum, output = sum(e*v)/sum(e) (truncating divide). Each vmem
// read loop presents the address one cycle before consuming the data; a delayed index
// (x_d) tags which element the just-arrived data belongs to. Bit-exact with
// QModel.attn_debug.
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
    localparam [3:0] P_IDLE=0, P_QLOAD=1, P_SCORE=2, P_EXP=3, P_WSUM=4, P_WDIV=5, P_NEXTH=6;
    reg [3:0]  ph;
    reg [3:0]  h;
    reg [9:0]  hbase;
    reg [4:0]  s, d;             // feed indices (s = context pos, d = within-head dim)
    reg [4:0]  s_d, d_d;         // delayed: index of the data valid THIS cycle
    reg [9:0]  soff;             // s*N_EMBED for the presented s
    reg        feeding, vld;

    reg signed [15:0] qreg [0:HEAD_DIM-1];
    reg signed [15:0] score [0:BLOCK-1];
    reg [15:0]        ev [0:BLOCK-1];
    reg signed [15:0] mmax;
    reg [31:0]        sum_e;
    reg signed [47:0] acc;

    // address presented this cycle (registered into vmem -> data next cycle)
    always @(*) begin
        case (ph)
            P_QLOAD: v_raddr = q_base + hbase + {5'd0, d};
            P_SCORE: v_raddr = k_base + soff + hbase + {5'd0, d};
            P_WSUM:  v_raddr = v_base + soff + hbase + {5'd0, d};
            default: v_raddr = 10'd0;
        endcase
    end

    // score scaling: attn_scale * sat16(acc >> FRAC)
    function signed [15:0] scale_score;
        input signed [47:0] a;
        reg signed [47:0] ash; reg signed [15:0] s1; reg signed [31:0] m, msh;
        begin
            ash = a >>> FRAC;
            s1 = (ash > 48'sd32767) ? 16'sd32767 : (ash < -48'sd32768) ? -16'sd32768 : ash[15:0];
            m = s1 * attn_scale; msh = m >>> FRAC;
            scale_score = (msh > 32'sd32767) ? 16'sd32767 : (msh < -32'sd32768) ? -16'sd32768 : msh[15:0];
        end
    endfunction

    // exp(score[s]-max)
    wire signed [16:0] diff = $signed(score[s]) - $signed(mmax);
    wire signed [15:0] dz = (diff < -17'sd32768) ? -16'sd32768 : diff[15:0];
    wire signed [15:0] eo;
    exp_unit u_exp (.z(dz), .e(eo));

    // weighted-sum divide: |acc| / sum_e, sign of acc
    reg         d_start;
    wire        d_done;  wire [47:0] d_quo;
    wire [47:0] num_abs = acc[47] ? (~acc + 48'd1) : acc;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num(num_abs), .den({31'd0, sum_e[16:0]}), .busy(), .done(d_done), .quo(d_quo));
    wire signed [47:0] q_signed = acc[47] ? -$signed(d_quo) : $signed(d_quo);
    wire signed [15:0] o_sat =
        (q_signed > 48'sd32767) ? 16'sd32767 : (q_signed < -48'sd32768) ? -16'sd32768 : q_signed[15:0];

    // MAC terms from the just-arrived data
    wire signed [47:0] kprod = $signed(qreg[d_d[2:0]]) * $signed(v_rdata);          // q.k
    wire signed [47:0] vprod = $signed({1'b0, ev[s_d[3:0]]}) * $signed(v_rdata);    // e.v
    wire signed [47:0] kacc  = (d_d == 0) ? kprod : acc + kprod;
    wire signed [47:0] vacc  = (s_d == 0) ? vprod : acc + vprod;

    always @(posedge clk) begin
        if (!resetn) begin
            ph <= P_IDLE; busy <= 0; done <= 0; v_we <= 0; d_start <= 0;
            feeding <= 0; vld <= 0;
        end else begin
            done <= 0; v_we <= 0; d_start <= 0;
            s_d <= s; d_d <= d; vld <= feeding;
            case (ph)
                P_IDLE: if (start) begin
                    busy <= 1; h <= 0; hbase <= 0; d <= 0; feeding <= 1; ph <= P_QLOAD;
                end
                P_QLOAD: begin
                    if (vld) qreg[d_d[2:0]] <= v_rdata;
                    if (feeding) begin
                        if (d == HEAD_DIM - 1) feeding <= 0;
                        else d <= d + 1;
                    end
                    if (vld && d_d == HEAD_DIM - 1) begin
                        s <= 0; d <= 0; soff <= 0; acc <= 0; mmax <= -16'sd32768;
                        feeding <= 1; ph <= P_SCORE;
                    end
                end
                P_SCORE: begin
                    if (vld) begin
                        acc <= kacc;
                        if (d_d == HEAD_DIM - 1) begin
                            score[s_d[3:0]] <= scale_score(kacc);
                            if (scale_score(kacc) > mmax) mmax <= scale_score(kacc);
                        end
                    end
                    if (feeding) begin
                        if (d == HEAD_DIM - 1) begin
                            d <= 0;
                            if (s == BLOCK - 1) feeding <= 0;
                            else begin s <= s + 1; soff <= soff + N_EMBED; end
                        end else d <= d + 1;
                    end
                    if (vld && s_d == BLOCK - 1 && d_d == HEAD_DIM - 1) begin
                        s <= 0; sum_e <= 0; ph <= P_EXP;
                    end
                end
                P_EXP: begin
                    ev[s[3:0]] <= eo;
                    sum_e <= sum_e + {16'd0, eo};
                    if (s == BLOCK - 1) begin
                        d <= 0; s <= 0; soff <= 0; acc <= 0; feeding <= 1; ph <= P_WSUM;
                    end else s <= s + 1;
                end
                P_WSUM: begin
                    if (vld) begin
                        acc <= vacc;
                        if (s_d == BLOCK - 1) begin d_start <= 1; feeding <= 0; ph <= P_WDIV; end
                    end
                    if (feeding) begin
                        if (s == BLOCK - 1) feeding <= 0;
                        else begin s <= s + 1; soff <= soff + N_EMBED; end
                    end
                end
                P_WDIV: if (d_done) begin
                    v_we <= 1; v_waddr <= o_base + hbase + {5'd0, d}; v_wdata <= o_sat;
                    if (d == HEAD_DIM - 1) ph <= P_NEXTH;
                    else begin d <= d + 1; s <= 0; soff <= 0; acc <= 0; feeding <= 1; ph <= P_WSUM; end
                end
                P_NEXTH: begin
                    if (h == N_HEAD - 1) begin busy <= 0; done <= 1; ph <= P_IDLE; end
                    else begin
                        h <= h + 1; hbase <= hbase + HEAD_DIM; d <= 0; feeding <= 1; ph <= P_QLOAD;
                    end
                end
                default: ph <= P_IDLE;
            endcase
        end
    end
endmodule
