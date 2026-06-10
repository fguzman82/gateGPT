// Matrix-vector engine: out[o] = sat16( (sum_i act[act_base+i] * W[sel][o*in_dim+i]) >>> descale )
// for o in 0..out_dim-1, written to vmem[dst_base+o]. Streams one MAC/cycle over the
// input dimension; drives vmem (async read for activations, sync write for results)
// and the weight ROM. Correctness-first (combinational multiply); MREG added later.
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
    // vmem activation read
    output wire [9:0]  v_raddr,
    input  wire signed [15:0] v_rdata,
    // vmem result write
    output reg         v_we,
    output reg [9:0]   v_waddr,
    output reg signed [15:0] v_wdata,
    // weight ROM
    output wire [11:0] w_addr,
    input  wire signed [15:0] w_rdata,
    output reg         busy,
    output reg         done
);
    localparam [1:0] S_IDLE=2'd0, S_MAC=2'd1, S_WB=2'd2;
    reg [1:0]  st;
    reg [6:0]  o, i;
    reg [11:0] wrow;
    reg signed [ACCW-1:0] acc;

    assign v_raddr = act_base + {3'd0, i};
    assign w_addr  = wrow + {5'd0, i};

    wire signed [ACCW-1:0] prod = $signed(v_rdata) * $signed(w_rdata);
    wire signed [ACCW-1:0] shifted = acc >>> descale;   // final accumulated sum
    wire signed [15:0] sat =
        (shifted >  48'sd32767)  ? 16'sd32767 :
        (shifted < -48'sd32768)  ? -16'sd32768 :
                                   shifted[15:0];

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 1'b0; done <= 1'b0; v_we <= 1'b0;
            o <= 7'd0; i <= 7'd0; wrow <= 12'd0; acc <= {ACCW{1'b0}};
        end else begin
            done <= 1'b0; v_we <= 1'b0;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1'b1; o <= 7'd0; i <= 7'd0; wrow <= 12'd0;
                    acc <= {ACCW{1'b0}}; st <= S_MAC;
                end
                S_MAC: begin
                    acc <= acc + prod;               // accumulate act[i]*w[o,i]
                    if (i == in_dim - 7'd1) st <= S_WB;
                    else i <= i + 7'd1;
                end
                S_WB: begin
                    v_we    <= 1'b1;
                    v_waddr <= dst_base + {3'd0, o};
                    v_wdata <= sat;                  // sat16(acc >>> descale)
                    if (o == out_dim - 7'd1) begin
                        busy <= 1'b0; done <= 1'b1; st <= S_IDLE;
                    end else begin
                        o <= o + 7'd1; i <= 7'd0; acc <= {ACCW{1'b0}};
                        wrow <= wrow + in_dim; st <= S_MAC;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
