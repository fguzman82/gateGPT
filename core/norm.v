// RMSNorm engine: y[i] = sat16( sat16(x[i]*scale >> FRAC) * gain[i] >> FRAC ),
// scale = min( 2^(2*FRAC) / isqrt(sum(x^2)/N), 32767 ). Bit-exact with the Python
// reference (tools/fixedpoint.rmsnorm). vmem read is REGISTERED, so the sum-of-squares
// pass reads x[i] read-ahead and caches it in xreg[]; the scale pass then works from
// the local cache (+ combinational gain), with no second vmem read.
module norm #(
    parameter integer N    = 24,
    parameter integer FRAC = 11
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [9:0]  src_base,
    input  wire [9:0]  dst_base,
    input  wire [1:0]  gain_sel,
    output wire [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  signed [15:0] v_wdata,
    output wire [5:0]  g_addr,
    input  wire signed [15:0] g_rdata,
    output reg         busy,
    output reg         done
);
    localparam [2:0] S_IDLE=0, S_SUM=1, S_SUMD=2, S_DIV1=3, S_SQRT=4, S_DIV2=5, S_SCALE=6;
    reg [2:0]  st;
    reg [6:0]  fi, fi_d;
    reg        feeding, vld;
    reg signed [47:0] ss;
    reg [31:0] scale_q;
    reg signed [15:0] xreg [0:N-1];
    reg signed [15:0] t1_r, gain_r;          // scale-pass pipeline registers

    assign v_raddr = src_base + {3'd0, fi};
    assign g_addr  = fi[5:0];

    wire signed [47:0] xsq = $signed(v_rdata) * $signed(v_rdata);

    // shared udiv / isqrt
    reg  [47:0] d_num, d_den;
    wire        d_done;  wire [47:0] d_quo;
    reg         d_start;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num(d_num), .den(d_den), .busy(), .done(d_done), .quo(d_quo));
    reg         s_start;
    wire        s_done;  wire [15:0] s_root;       // mean-square <= 2^30 -> 32-bit radicand
    isqrt #(.W(32)) u_sqrt (.clk(clk), .resetn(resetn), .start(s_start),
        .radicand(d_num[31:0]), .busy(), .done(s_done), .root(s_root));

    // scale pass, stage 1: t1 = sat16( x*scale >> FRAC )  (registered into t1_r)
    wire signed [31:0] xs    = $signed(xreg[fi[4:0]]) * $signed(scale_q[15:0]);
    wire signed [31:0] xs_sh = xs >>> FRAC;
    wire signed [15:0] t1 =
        (xs_sh >  32'sd32767) ? 16'sd32767 : (xs_sh < -32'sd32768) ? -16'sd32768 : xs_sh[15:0];
    // scale pass, stage 2: y = sat16( t1 * gain >> FRAC )  (from registered t1_r/gain_r)
    wire signed [31:0] tg    = t1_r * gain_r;
    wire signed [31:0] tg_sh = tg >>> FRAC;
    wire signed [15:0] yval =
        (tg_sh >  32'sd32767) ? 16'sd32767 : (tg_sh < -32'sd32768) ? -16'sd32768 : tg_sh[15:0];

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0; d_start <= 0; s_start <= 0;
            feeding <= 0; vld <= 0;
        end else begin
            done <= 0; v_we <= 0; d_start <= 0; s_start <= 0;
            fi_d <= fi; vld <= feeding;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; ss <= 0; feeding <= 1; st <= S_SUM;
                end
                S_SUM: begin
                    if (vld) begin ss <= ss + xsq; xreg[fi_d[4:0]] <= v_rdata; end
                    if (fi == N - 1) begin feeding <= 0; st <= S_SUMD; end
                    else fi <= fi + 1;
                end
                S_SUMD: begin
                    xreg[fi_d[4:0]] <= v_rdata;            // last x
                    d_num <= ss + xsq; d_den <= N; d_start <= 1;   // full sum / N
                    st <= S_DIV1;
                end
                S_DIV1: if (d_done) begin
                    d_num <= (d_quo[31:0] < 1) ? 48'd1 : {16'd0, d_quo[31:0]};   // feed isqrt
                    s_start <= 1; st <= S_SQRT;
                end
                S_SQRT: if (s_done) begin
                    d_num <= 48'd1 << (2*FRAC);
                    d_den <= {32'd0, ((s_root < 1) ? 16'd1 : s_root)};
                    d_start <= 1; st <= S_DIV2;
                end
                S_DIV2: if (d_done) begin
                    scale_q <= (d_quo > 48'd32767) ? 32'd32767 : d_quo[31:0];
                    fi <= 0; feeding <= 1; st <= S_SCALE;
                end
                S_SCALE: begin
                    t1_r <= t1; gain_r <= g_rdata;            // stage 1 -> register
                    if (vld) begin                            // stage 2 -> write
                        v_we <= 1; v_waddr <= dst_base + {3'd0, fi_d}; v_wdata <= yval;
                        if (fi_d == N - 1) begin busy <= 0; done <= 1; st <= S_IDLE; end
                    end
                    if (feeding) begin
                        if (fi == N - 1) feeding <= 0;
                        else fi <= fi + 1;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
