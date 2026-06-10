// Elementwise vector op over vmem:
//   op=0 ADD : dst[i] = sat16( a[i] + b[i] )      (residual adds)
//   op=1 RELU: dst[i] = max(0, a[i])              (MLP activation)
// ADD takes 2 cycles/element (single vmem read port: read a, then read b+write);
// RELU takes 1 cycle/element.
module vecop (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire        op,            // 0=ADD, 1=RELU
    input  wire [9:0]  a_base,
    input  wire [9:0]  b_base,
    input  wire [9:0]  dst_base,
    input  wire [6:0]  cnt,
    output reg  [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  signed [15:0] v_wdata,
    output reg         busy,
    output reg         done
);
    localparam [1:0] S_IDLE=0, S_RA=1, S_RB=2;
    reg [1:0]  st;
    reg [6:0]  i;
    reg signed [15:0] aval;

    wire signed [16:0] add = $signed(aval) + $signed(v_rdata);
    wire signed [15:0] addsat =
        (add >  17'sd32767) ? 16'sd32767 : (add < -17'sd32768) ? -16'sd32768 : add[15:0];
    wire signed [15:0] reluv = (v_rdata[15]) ? 16'sd0 : v_rdata;

    always @(posedge clk) begin
        if (!resetn) begin st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0; end
        else begin
            done <= 0; v_we <= 0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; i <= 0; v_raddr <= a_base; st <= S_RA;
                end
                S_RA: begin
                    if (op == 1'b1) begin
                        // RELU: one read, write, advance
                        v_we <= 1; v_waddr <= dst_base + i; v_wdata <= reluv;
                        if (i == cnt - 1) begin busy <= 0; done <= 1; st <= S_IDLE; end
                        else begin i <= i + 1; v_raddr <= a_base + i + 1; end
                    end else begin
                        aval <= v_rdata; v_raddr <= b_base + i; st <= S_RB;
                    end
                end
                S_RB: begin
                    v_we <= 1; v_waddr <= dst_base + i; v_wdata <= addsat;
                    if (i == cnt - 1) begin busy <= 0; done <= 1; st <= S_IDLE; end
                    else begin i <= i + 1; v_raddr <= a_base + i + 1; st <= S_RA; end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
