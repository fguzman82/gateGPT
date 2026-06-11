// RMSNorm engine: y[i] = sat16( sat16(x[i]*scale >> FRAC) * gain[i] >> FRAC ),
// scale = min( 2^(2*FRAC) / isqrt(sum(x^2)/N), 32767 ). Bit-exact with the Python
// reference (tools/fixedpoint.rmsnorm). Uses the TRUE dual-port vmem: the sum-of-squares
// pass reads TWO elements per cycle (ports A+B) and the scale pass writes TWO per cycle,
// so both N-length loops run in N/2 cycles (N must be even). Read addresses are driven
// combinationally (registered read inside vmem -> 1-cycle latency, read-ahead).
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
    // port A
    output reg  [9:0]  addr_a,
    input  wire signed [15:0] rd_a,
    output reg         we_a,
    output reg  signed [15:0] wd_a,
    // port B
    output reg  [9:0]  addr_b,
    input  wire signed [15:0] rd_b,
    output reg         we_b,
    output reg  signed [15:0] wd_b,
    // gains (two per cycle)
    output wire [5:0]  g_addr_a,
    output wire [5:0]  g_addr_b,
    input  wire signed [15:0] g_rdata_a,
    input  wire signed [15:0] g_rdata_b,
    output reg         busy,
    output reg         done
);
    localparam [2:0] S_IDLE=0, S_SUM=1, S_SUMD=2, S_DIV1=3, S_SQRT=4, S_DIV2=5, S_SCALE=6;
    reg [2:0]  st;
    reg [6:0]  fi, fi_d;                  // element index (advances by 2), delayed copy
    reg        feeding, vld;
    reg signed [47:0] ss;
    reg [31:0] scale_q;
    reg signed [15:0] xreg [0:N-1];
    reg signed [15:0] t1a_r, t1b_r, ga_r, gb_r;   // scale-pass pipeline registers

    assign g_addr_a = fi[5:0];
    assign g_addr_b = fi[5:0] + 6'd1;

    wire signed [47:0] xsq_a = $signed(rd_a) * $signed(rd_a);
    wire signed [47:0] xsq_b = $signed(rd_b) * $signed(rd_b);

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

    // scale pass stage 1 (two lanes): t1 = sat16( x*scale >> FRAC )
    wire signed [31:0] xsa    = $signed(xreg[fi[4:0]])        * $signed(scale_q[15:0]);
    wire signed [31:0] xsb    = $signed(xreg[fi[4:0] + 5'd1]) * $signed(scale_q[15:0]);
    wire signed [31:0] xsa_sh = xsa >>> FRAC;
    wire signed [31:0] xsb_sh = xsb >>> FRAC;
    wire signed [15:0] t1a =
        (xsa_sh >  32'sd32767) ? 16'sd32767 : (xsa_sh < -32'sd32768) ? -16'sd32768 : xsa_sh[15:0];
    wire signed [15:0] t1b =
        (xsb_sh >  32'sd32767) ? 16'sd32767 : (xsb_sh < -32'sd32768) ? -16'sd32768 : xsb_sh[15:0];
    // scale pass stage 2 (two lanes): y = sat16( t1 * gain >> FRAC )
    wire signed [31:0] tga    = t1a_r * ga_r;
    wire signed [31:0] tgb    = t1b_r * gb_r;
    wire signed [31:0] tga_sh = tga >>> FRAC;
    wire signed [31:0] tgb_sh = tgb >>> FRAC;
    wire signed [15:0] ya =
        (tga_sh >  32'sd32767) ? 16'sd32767 : (tga_sh < -32'sd32768) ? -16'sd32768 : tga_sh[15:0];
    wire signed [15:0] yb =
        (tgb_sh >  32'sd32767) ? 16'sd32767 : (tgb_sh < -32'sd32768) ? -16'sd32768 : tgb_sh[15:0];

    // combinational port drivers (read pair during SUM, write pair during SCALE)
    wire scale_wr = (st == S_SCALE) && vld;
    always @(*) begin
        addr_a = src_base + {3'd0, fi};
        addr_b = src_base + {3'd0, fi} + 10'd1;
        we_a = 1'b0; we_b = 1'b0; wd_a = ya; wd_b = yb;
        if (scale_wr) begin
            addr_a = dst_base + {3'd0, fi_d};
            addr_b = dst_base + {3'd0, fi_d} + 10'd1;
            we_a = 1'b1; we_b = 1'b1;
        end
    end

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0;
            d_start <= 0; s_start <= 0; feeding <= 0; vld <= 0;
        end else begin
            done <= 0; d_start <= 0; s_start <= 0;
            fi_d <= fi; vld <= feeding;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; ss <= 0; feeding <= 1; st <= S_SUM;
                end
                S_SUM: begin
                    if (vld) begin
                        ss <= ss + xsq_a + xsq_b;
                        xreg[fi_d[4:0]] <= rd_a; xreg[fi_d[4:0] + 5'd1] <= rd_b;
                    end
                    if (fi == N - 2) begin feeding <= 0; st <= S_SUMD; end
                    else fi <= fi + 7'd2;
                end
                S_SUMD: begin
                    xreg[fi_d[4:0]] <= rd_a; xreg[fi_d[4:0] + 5'd1] <= rd_b;   // last pair
                    d_num <= ss + xsq_a + xsq_b; d_den <= N; d_start <= 1;      // full sum / N
                    st <= S_DIV1;
                end
                S_DIV1: if (d_done) begin
                    d_num <= (d_quo[31:0] < 1) ? 48'd1 : {16'd0, d_quo[31:0]};  // feed isqrt
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
                    t1a_r <= t1a; t1b_r <= t1b; ga_r <= g_rdata_a; gb_r <= g_rdata_b;
                    if (vld && fi_d == N - 2) begin busy <= 0; done <= 1; st <= S_IDLE; end
                    if (feeding) begin
                        if (fi == N - 2) feeding <= 0;
                        else fi <= fi + 7'd2;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
