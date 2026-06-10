// RMSNorm engine: y[i] = sat16( sat16(x[i]*scale >> FRAC) * gain[i] >> FRAC ),
// scale = min( 2^(2*FRAC) / isqrt(sum(x^2)/N), 32767 ). Bit-exact with the Python
// reference (tools/fixedpoint.rmsnorm). Uses the shared udiv + isqrt primitives.
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
    // vmem
    output reg  [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  signed [15:0] v_wdata,
    // gain ROM
    output wire [5:0]  g_addr,
    input  wire signed [15:0] g_rdata,
    output reg         busy,
    output reg         done
);
    localparam [3:0] S_IDLE=0, S_SUM=1, S_DIV1=2, S_SQRT=3, S_DIV2=4, S_SCALE=5;
    reg [3:0]  st;
    reg [6:0]  i;
    reg signed [47:0] ss;
    reg [31:0] scale_q;
    reg        opgo;          // 1-cycle start pulse for the active primitive

    assign g_addr = i[5:0];

    // shared udiv / isqrt
    reg  [47:0] d_num, d_den;
    wire        d_done;  wire [47:0] d_quo;
    reg         d_start;
    udiv #(.W(48)) u_div (.clk(clk), .resetn(resetn), .start(d_start),
        .num(d_num), .den(d_den), .busy(), .done(d_done), .quo(d_quo));
    reg         s_start;
    wire        s_done;  wire [23:0] s_root;
    isqrt #(.W(48)) u_sqrt (.clk(clk), .resetn(resetn), .start(s_start),
        .radicand({16'd0, d_num[31:0]}), .busy(), .done(s_done), .root(s_root));

    reg [31:0] mean_sq;
    reg [23:0] rootr;

    // scaling datapath (combinational)
    wire signed [31:0] xs    = $signed(v_rdata) * $signed(scale_q[15:0]);
    wire signed [31:0] xs_sh = xs >>> FRAC;
    wire signed [15:0] t1 =
        (xs_sh >  32'sd32767) ? 16'sd32767 :
        (xs_sh < -32'sd32768) ? -16'sd32768 : xs_sh[15:0];
    wire signed [31:0] tg    = t1 * $signed(g_rdata);
    wire signed [31:0] tg_sh = tg >>> FRAC;
    wire signed [15:0] yval =
        (tg_sh >  32'sd32767) ? 16'sd32767 :
        (tg_sh < -32'sd32768) ? -16'sd32768 : tg_sh[15:0];

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0; d_start <= 0; s_start <= 0;
        end else begin
            done <= 0; v_we <= 0; d_start <= 0; s_start <= 0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; i <= 0; ss <= 0; v_raddr <= src_base; st <= S_SUM;
                end
                S_SUM: begin
                    ss <= ss + $signed(v_rdata) * $signed(v_rdata);
                    if (i == N - 1) begin
                        // launch ss / N
                        d_num <= ss + $signed(v_rdata) * $signed(v_rdata);
                        d_den <= N; d_start <= 1; st <= S_DIV1;
                    end else begin
                        i <= i + 1; v_raddr <= src_base + i + 1;
                    end
                end
                S_DIV1: if (d_done) begin
                    mean_sq <= (d_quo[31:0] < 1) ? 32'd1 : d_quo[31:0];
                    d_num <= (d_quo[31:0] < 1) ? 48'd1 : {16'd0, d_quo[31:0]};  // feed isqrt
                    s_start <= 1; st <= S_SQRT;
                end
                S_SQRT: if (s_done) begin
                    rootr <= (s_root < 1) ? 24'd1 : s_root;
                    d_num <= 48'd1 << (2*FRAC);
                    d_den <= {24'd0, ((s_root < 1) ? 24'd1 : s_root)};
                    d_start <= 1; st <= S_DIV2;
                end
                S_DIV2: if (d_done) begin
                    scale_q <= (d_quo > 48'd32767) ? 32'd32767 : d_quo[31:0];
                    i <= 0; v_raddr <= src_base; st <= S_SCALE;
                end
                S_SCALE: begin
                    v_we <= 1; v_waddr <= dst_base + i; v_wdata <= yval;
                    if (i == N - 1) begin busy <= 0; done <= 1; st <= S_IDLE; end
                    else begin i <= i + 1; v_raddr <= src_base + i + 1; end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
