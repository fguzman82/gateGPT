// Parallel matrix-vector engine: out[o] = sat16( (sum_i act[act_base+i] * W[sel][o,i]) >>> descale )
// for o in 0..out_dim-1, written to vmem[dst_base+o]. Output rows are processed LANES at a
// time (one tile). Within a tile, each cycle broadcasts one activation while the wide weight
// ROM returns LANES weights (one per row) feeding LANES independent accumulators; this turns
// the inner product into one MAC per row per cycle. tiles = ceil(out_dim/LANES).
// The accumulators are DOUBLE-BUFFERED: while one bank's results are written back one row per
// cycle on the vmem WRITE port, the next tile is already computing into the other bank on the
// vmem READ port (vmem is 1-write + 1-registered-read), so a multi-tile matmul hides its
// writeback behind the next tile's compute. vmem read is registered (1-cycle), so addresses
// are presented a cycle ahead and the weight bus is registered (w_rdata_r) to align -- this
// also gives the multipliers an MREG stage for higher Fmax.
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
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DRAIN=2'd2, S_FLUSH=2'd3;
    reg [1:0]  st;
    reg [6:0]  fi;                    // feed (input) index within the current tile
    reg [6:0]  obase;                 // first output row of the current (compute) tile
    reg [11:0] wbase;                 // first word of the current (compute) tile = tile*in_dim
    reg        cbank;                 // accumulator bank being computed into
    reg        feeding, vld;          // feeding addresses; product valid this cycle
    reg [LANES*16-1:0] w_rdata_r;     // weights registered to align with the vmem read
    // double-buffered accumulators: 2 banks of LANES, addressed {bank, lane}
    reg signed [ACCW-1:0] acc [0:2*32-1];
    integer L;

    // concurrent writeback engine (drains a bank on the write port while compute runs)
    reg        wb_active;
    reg [6:0]  wbi;                   // writeback lane index
    reg        wbbank;                // bank being written back
    reg [6:0]  wb_obase;             // output base of the bank being written back
    wire signed [ACCW-1:0] wb_shift = acc[{wbbank, wbi[4:0]}] >>> descale;
    wire signed [15:0] wb_sat =
        (wb_shift >  48'sd32767) ? 16'sd32767 :
        (wb_shift < -48'sd32768) ? -16'sd32768 : wb_shift[15:0];
    wire [6:0] wb_row = wb_obase + wbi;

    assign v_raddr = act_base + {3'd0, fi};
    assign w_addr  = wbase + {5'd0, fi};

    always @(posedge clk) begin
        if (!resetn) begin
            st <= S_IDLE; busy <= 0; done <= 0; v_we <= 0;
            fi <= 0; obase <= 0; wbase <= 0; cbank <= 0; feeding <= 0; vld <= 0;
            wb_active <= 0; wbi <= 0; wbbank <= 0; wb_obase <= 0;
            for (L = 0; L < 2*32; L = L + 1) acc[L] <= 0;
        end else begin
            done <= 0; v_we <= 0;
            w_rdata_r <= w_rdata;          // align weights with the registered vmem read
            vld <= feeding;                // product valid one cycle after a fed address

            // ---- concurrent writeback (vmem write port) ----
            if (wb_active) begin
                if (wb_row < out_dim) begin   // skip padding rows
                    v_we <= 1; v_waddr <= dst_base + {3'd0, wb_row}; v_wdata <= wb_sat;
                end
                if (wbi == LANES - 1) wb_active <= 0;
                else wbi <= wbi + 7'd1;
            end

            // ---- compute (vmem read port) ----
            case (st)
                S_IDLE: if (start) begin
                    busy <= 1; fi <= 0; obase <= 0; wbase <= 0; cbank <= 0;
                    for (L = 0; L < LANES; L = L + 1) acc[L] <= 0;     // bank 0
                    feeding <= 1; st <= S_RUN;
                end
                S_RUN: begin
                    if (vld)
                        for (L = 0; L < LANES; L = L + 1)
                            acc[{cbank, L[4:0]}] <= acc[{cbank, L[4:0]}]
                                + $signed(v_rdata) * $signed(w_rdata_r[L*16 +: 16]);
                    if (fi == in_dim - 7'd1) begin feeding <= 0; st <= S_DRAIN; end
                    else fi <= fi + 7'd1;
                end
                S_DRAIN: begin
                    if (vld)                                   // last product of the tile
                        for (L = 0; L < LANES; L = L + 1)
                            acc[{cbank, L[4:0]}] <= acc[{cbank, L[4:0]}]
                                + $signed(v_rdata) * $signed(w_rdata_r[L*16 +: 16]);
                    // tile compute complete: hand this bank to the writeback engine
                    wbbank <= cbank; wb_obase <= obase; wbi <= 0; wb_active <= 1;
                    if (obase + LANES >= out_dim) begin
                        st <= S_FLUSH;                         // last tile: drain alone
                    end else begin                             // start next tile in the other bank
                        cbank <= ~cbank;
                        for (L = 0; L < LANES; L = L + 1) acc[{~cbank, L[4:0]}] <= 0;
                        obase <= obase + LANES[6:0]; wbase <= wbase + {5'd0, in_dim};
                        fi <= 0; feeding <= 1; st <= S_RUN;
                    end
                end
                S_FLUSH: if (!wb_active) begin                 // last writeback finished
                    busy <= 0; done <= 1; st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
