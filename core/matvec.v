// Parallel matrix-vector engine: out[o] = sat16( (sum_i act[act_base+i] * W[sel][o,i]) >>> descale )
// for o in 0..out_dim-1, written to vmem[dst_base+o]. Output rows are processed LANES at a
// time (one tile). Using the TRUE dual-port vmem, each compute cycle reads TWO activations
// (act[2j], act[2j+1]) and the wide weight ROM returns the LANES weights for both columns,
// so every lane does TWO MACs/cycle -> a tile computes in in_dim/2 cycles. Writeback then
// drains TWO rows/cycle on the two write ports -> LANES/2 cycles. tiles = ceil(out_dim/LANES).
// Reads are registered (1-cycle): addresses are driven combinationally a cycle ahead and the
// weight bus is registered (w_rdata_r) to align -- giving the multipliers an MREG stage too.
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
    // port A (reads act[2j]; writes even rows)
    output reg  [9:0]  addr_a,
    input  wire signed [15:0] rd_a,
    output reg         we_a,
    output reg  signed [15:0] wd_a,
    // port B (reads act[2j+1]; writes odd rows)
    output reg  [9:0]  addr_b,
    input  wire signed [15:0] rd_b,
    output reg         we_b,
    output reg  signed [15:0] wd_b,
    output wire [11:0] w_addr,                  // tile-word address: tile*(in_dim/2) + j
    input  wire [2*LANES*16-1:0] w_rdata,       // LANES weights for col 2j (low) + col 2j+1 (high)
    output reg         busy,
    output reg         done
);
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DRAIN=2'd2, S_WB=2'd3;
    localparam integer HW = LANES*16;           // half-word boundary (col 2j+1 offset)
    reg [1:0]  st;
    reg [6:0]  fi;                    // column-pair (feed) index within the current tile
    reg [6:0]  obase;                 // first output row of the current tile = tile*LANES
    reg [11:0] wbase;                 // first word of the current tile = tile*(in_dim/2)
    reg [6:0]  wbi;                   // writeback row index within the tile (advances by 2)
    reg        feeding, vld;
    reg [2*LANES*16-1:0] w_rdata_r;   // weights registered to align with the vmem read
    reg signed [ACCW-1:0] acc [0:LANES-1];
    integer L;

    assign w_addr = wbase + {5'd0, fi};

    // writeback saturate for the current row pair
    wire signed [ACCW-1:0] sh_a = acc[wbi[4:0]]        >>> descale;
    wire signed [ACCW-1:0] sh_b = acc[wbi[4:0] + 5'd1] >>> descale;
    wire signed [15:0] sat_a =
        (sh_a > 48'sd32767) ? 16'sd32767 : (sh_a < -48'sd32768) ? -16'sd32768 : sh_a[15:0];
    wire signed [15:0] sat_b =
        (sh_b > 48'sd32767) ? 16'sd32767 : (sh_b < -48'sd32768) ? -16'sd32768 : sh_b[15:0];

    // combinational port drivers: read the activation pair during compute, write a row pair in S_WB
    always @(*) begin
        addr_a = act_base + {2'd0, fi, 1'b0};       // act[2j]
        addr_b = act_base + {2'd0, fi, 1'b0} + 10'd1; // act[2j+1]
        we_a = 1'b0; we_b = 1'b0; wd_a = sat_a; wd_b = sat_b;
        if (st == S_WB) begin
            addr_a = dst_base + {3'd0, obase + wbi};
            addr_b = dst_base + {3'd0, obase + wbi} + 10'd1;
            we_a = (obase + wbi      < out_dim);
            we_b = (obase + wbi + 7'd1 < out_dim);
        end
    end

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0;
            fi <= 0; obase <= 0; wbase <= 0; wbi <= 0; feeding <= 0; vld <= 0;
            for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
        end else begin
            done <= 0;
            w_rdata_r <= w_rdata;          // align weights with the registered vmem read
            vld <= feeding;
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; obase <= 0; wbase <= 0;
                    for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
                    feeding <= 1; st <= S_RUN;
                end
                S_RUN: begin
                    if (vld)
                        for (L = 0; L < LANES; L = L + 1)
                            acc[L] <= acc[L]
                                + $signed(rd_a) * $signed(w_rdata_r[L*16 +: 16])
                                + $signed(rd_b) * $signed(w_rdata_r[HW + L*16 +: 16]);
                    if (fi == (in_dim >> 1) - 7'd1) begin feeding <= 0; st <= S_DRAIN; end
                    else fi <= fi + 7'd1;
                end
                S_DRAIN: begin
                    if (vld)                                   // last column pair
                        for (L = 0; L < LANES; L = L + 1)
                            acc[L] <= acc[L]
                                + $signed(rd_a) * $signed(w_rdata_r[L*16 +: 16])
                                + $signed(rd_b) * $signed(w_rdata_r[HW + L*16 +: 16]);
                    wbi <= 0; st <= S_WB;
                end
                S_WB: begin                                    // write two rows per cycle
                    if (wbi >= LANES - 7'd2) begin
                        if (obase + LANES >= out_dim) begin    // last tile -> finished
                            busy <= 0; done <= 1; st <= S_IDLE;
                        end else begin                         // next tile
                            obase <= obase + LANES[6:0]; wbase <= wbase + {5'd0, (in_dim >> 1)};
                            fi <= 0;
                            for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;
                            feeding <= 1; st <= S_RUN;
                        end
                    end else wbi <= wbi + 7'd2;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
