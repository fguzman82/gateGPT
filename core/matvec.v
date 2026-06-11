// Parallel matrix-vector engine: out[o] = sat16( (sum_i act[act_base+i] * W[sel][o,i]) >>> descale )
// for o in 0..out_dim-1, written to vmem[dst_base+o]. Output rows are processed LANES at a
// time (one tile). Within a tile, each cycle broadcasts one activation while the wide weight
// ROM returns LANES weights (one per row) that feed LANES independent accumulators; this turns
// the inner product into one MAC per row per cycle. tiles = ceil(out_dim/LANES).
// vmem read is REGISTERED (1-cycle latency): addresses are presented a cycle ahead and the
// weight bus is registered (w_rdata_r) to align with it -- this also gives the multipliers an
// MREG stage for higher Fmax. Results are written back one row per cycle (single vmem port).
module matvec #(
    parameter integer LANES = 24,
    parameter integer ACCW  = 48
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
    output wire [11:0] w_addr,                // tile-word address: tile*in_dim + i
    input  wire [LANES*16-1:0] w_rdata,       // LANES signed 16-bit weights (lane 0 = LSB)
    output reg         busy,
    output reg         done
);
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DRAIN=2'd2, S_WB=2'd3;
    reg [1:0]  st;
    reg [6:0]  fi;                    // feed (input) index within the current tile
    reg [6:0]  obase;                 // first output row of the current tile = tile*LANES
    reg [11:0] wbase;                 // first word of the current tile = tile*in_dim
    reg [6:0]  wbi;                   // writeback lane index within the tile
    reg        feeding, vld;          // feeding addresses; product valid this cycle
    reg [LANES*16-1:0] w_rdata_r;     // weights registered to align with the vmem read
    reg signed [ACCW-1:0] acc [0:LANES-1];
    integer L;

    assign v_raddr = act_base + {3'd0, fi};
    assign w_addr  = wbase + {5'd0, fi};

    // writeback: shift + saturate the selected lane's accumulator
    wire signed [ACCW-1:0] wb_shift = acc[wbi] >>> descale;
    wire signed [15:0] wb_sat =
        (wb_shift >  48'sd32767) ? 16'sd32767 :
        (wb_shift < -48'sd32768) ? -16'sd32768 : wb_shift[15:0];
    wire [6:0] wb_row = obase + wbi;                 // absolute output row for this lane

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0;
            fi <= 0; obase <= 0; wbase <= 0; wbi <= 0; feeding <= 0; vld <= 0;
            for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
        end else begin
            done <= 0; v_we <= 0;
            w_rdata_r <= w_rdata;          // align weights with the registered vmem read
            vld <= feeding;                // product valid one cycle after a fed address
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; obase <= 0; wbase <= 0;
                    for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
                    feeding <= 1; st <= S_RUN;
                end
                S_RUN: begin
                    if (vld)
                        for (L = 0; L < LANES; L = L + 1)
                            acc[L] <= acc[L] + $signed(v_rdata) * $signed(w_rdata_r[L*16 +: 16]);
                    if (fi == in_dim - 7'd1) begin feeding <= 0; st <= S_DRAIN; end
                    else fi <= fi + 7'd1;
                end
                S_DRAIN: begin
                    if (vld)                                   // last product of the tile
                        for (L = 0; L < LANES; L = L + 1)
                            acc[L] <= acc[L] + $signed(v_rdata) * $signed(w_rdata_r[L*16 +: 16]);
                    wbi <= 0; st <= S_WB;
                end
                S_WB: begin
                    if (wb_row < out_dim) begin                // skip padding rows
                        v_we <= 1; v_waddr <= dst_base + {3'd0, wb_row}; v_wdata <= wb_sat;
                    end
                    if (wbi == LANES - 1) begin
                        if (obase + LANES >= out_dim) begin     // last tile -> finished
                            busy <= 0; done <= 1; st <= S_IDLE;
                        end else begin                          // next tile
                            obase <= obase + LANES[6:0]; wbase <= wbase + {5'd0, in_dim};
                            fi <= 0;
                            for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
                            feeding <= 1; st <= S_RUN;
                        end
                    end else wbi <= wbi + 7'd1;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
