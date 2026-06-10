// Matrix-vector engine: out[o] = sat16( (sum_i act[act_base+i] * W[sel][o*in_dim+i]) >>> descale )
// for o in 0..out_dim-1, written to vmem[dst_base+o]. Streams one MAC/cycle over the
// input dimension. vmem read is REGISTERED (1-cycle latency) so addresses are
// presented one cycle ahead; the weight ROM read is registered (w_rdata_r) to align
// with it -- this also gives the multiply a register stage (MREG) for higher Fmax.
module matvec #(
    parameter integer ACCW = 48
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [2:0]  wsel,
    input  wire [6:0]  in_dim,
    input  wire [6:0]  out_dim,
    input  wire [9:0]  act_base,
    input  wire [9:0]  dst_base,
    input  wire [4:0]  descale,
    output wire [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    output reg         v_we,
    output reg [9:0]   v_waddr,
    output reg signed [15:0] v_wdata,
    output wire [11:0] w_addr,
    input  wire signed [15:0] w_rdata,
    output reg         busy,
    output reg         done
);
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DRAIN=2'd2, S_WB=2'd3;
    reg [1:0]  st;
    reg [6:0]  o, fi;                 // output row, feed (address) index
    reg [11:0] wrow;
    reg signed [ACCW-1:0] acc;
    reg        feeding, vld;          // feeding addresses; product valid this cycle
    reg signed [15:0] w_rdata_r;      // weight registered to align with vmem read

    assign v_raddr = act_base + {3'd0, fi};
    assign w_addr  = wrow + {5'd0, fi};

    wire signed [ACCW-1:0] prod = $signed(v_rdata) * $signed(w_rdata_r);
    wire signed [ACCW-1:0] shifted = acc >>> descale;
    wire signed [15:0] sat =
        (shifted >  48'sd32767)  ? 16'sd32767 :
        (shifted < -48'sd32768)  ? -16'sd32768 : shifted[15:0];

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0;
            o <= 0; fi <= 0; wrow <= 0; acc <= 0; feeding <= 0; vld <= 0;
        end else begin
            done <= 0; v_we <= 0;
            w_rdata_r <= w_rdata;          // align weight with the registered vmem read
            vld <= feeding;                // product valid one cycle after a fed address
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; o <= 0; fi <= 0; wrow <= 0; acc <= 0;
                    feeding <= 1; st <= S_RUN;
                end
                S_RUN: begin
                    if (vld) acc <= acc + prod;
                    if (fi == in_dim - 7'd1) begin feeding <= 0; st <= S_DRAIN; end
                    else fi <= fi + 7'd1;
                end
                S_DRAIN: begin
                    if (vld) acc <= acc + prod;        // last product
                    st <= S_WB;
                end
                S_WB: begin
                    v_we <= 1; v_waddr <= dst_base + {3'd0, o}; v_wdata <= sat;
                    if (o == out_dim - 7'd1) begin busy <= 0; done <= 1; st <= S_IDLE; end
                    else begin
                        o <= o + 7'd1; fi <= 0; acc <= 0; wrow <= wrow + in_dim;
                        feeding <= 1; st <= S_RUN;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
