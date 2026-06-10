// Rotary-encoder throttle for the name generator (Panasonic EVQWK4001, 15 detents).
// The two phase outputs (INCA/INCB) set an auto-rotation SPEED LEVEL; the level maps
// to an auto-start interval, exponentially, from 1 Hz (level 0) up to as fast as the
// core can generate (back-to-back). The push button freezes auto-rotation while held
// (to read the current name). start_btn (handled in the top) still does single names.
module rotary_throttle #(
    parameter integer CLK_HZ    = 50_000_000,  // core clock
    parameter integer MAX_LEVEL = 15,          // 1 rev (15 detents) spans the range
    parameter integer FILTER    = 2500          // ~50 us deglitch (cycles a level must hold)
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        rot_a,        // INCA (async, active high)
    input  wire        rot_b,        // INCB (async)
    input  wire        rot_push,     // PUSH (async, active high) -> pause while held
    input  wire        gen_busy,     // name generator busy
    output reg         auto_start,   // 1-cycle pulse: launch a generation
    output reg  [4:0]  speed_level
);
    // ---- synchronize the async encoder inputs (2 FF) ----
    reg [1:0] a_ff, b_ff, p_ff;
    always @(posedge clk) begin
        a_ff <= {a_ff[0], rot_a};
        b_ff <= {b_ff[0], rot_b};
        p_ff <= {p_ff[0], rot_push};
    end
    wire a_sync = a_ff[1];
    wire b_sync = b_ff[1];
    wire paused = p_ff[1];

    // ---- deglitch: only accept a level after it's stable for FILTER cycles ----
    // Mechanical/noisy encoder lines were producing a steady stream of edges at rest
    // that the decoder counted (level powered up around 6 -> ~330 t/s). A real detent
    // state lasts milliseconds, so filtering sub-50us glitches keeps it at 0 at rest.
    reg a_clean, b_clean;
    reg [11:0] a_cnt, b_cnt;
    always @(posedge clk) begin
        if (!resetn) begin
            a_clean <= 1'b0; b_clean <= 1'b0; a_cnt <= 12'd0; b_cnt <= 12'd0;
        end else begin
            if (a_sync == a_clean) a_cnt <= 12'd0;
            else if (a_cnt >= FILTER) begin a_clean <= a_sync; a_cnt <= 12'd0; end
            else a_cnt <= a_cnt + 12'd1;

            if (b_sync == b_clean) b_cnt <= 12'd0;
            else if (b_cnt >= FILTER) begin b_clean <= b_sync; b_cnt <= 12'd0; end
            else b_cnt <= b_cnt + 12'd1;
        end
    end

    // ---- robust quadrature decode: accumulate signed edges; one LEVEL step per
    // detent (EDGES_PER_DETENT quadrature edges). This rejects contact bounce (an
    // up edge immediately followed by a down edge nets to zero) and reads direction
    // from the full {A,B} state -- unlike the old "A-edge + B-level" decoder, which
    // mis-read direction (only ever went up) and counted bounce, so the level never
    // started at 0. A startup holdoff ignores power-up/DCM-lock glitches.
    localparam integer EDGES_PER_DETENT = 4;
    localparam integer STARTUP_HOLD     = CLK_HZ / 5;   // ~200 ms

    reg [1:0] ab, ab_d;
    always @(posedge clk) begin
        ab   <= {a_clean, b_clean};
        ab_d <= ab;
    end
    wire [3:0] tr = {ab_d, ab};
    wire up_edge = (tr == 4'b0001) || (tr == 4'b0111) || (tr == 4'b1110) || (tr == 4'b1000);
    wire dn_edge = (tr == 4'b0010) || (tr == 4'b1011) || (tr == 4'b1101) || (tr == 4'b0100);

    reg signed [3:0] acc;
    reg [31:0]       startup;
    wire armed = (startup == 32'd0);

    always @(posedge clk) begin
        if (!resetn) begin
            speed_level <= 5'd0;
            acc         <= 4'sd0;
            startup     <= STARTUP_HOLD[31:0];
        end else begin
            if (startup != 32'd0) startup <= startup - 32'd1;
            if (armed && (up_edge ^ dn_edge)) begin
                if (up_edge) begin
                    if (acc >= EDGES_PER_DETENT - 1) begin
                        acc <= 4'sd0;
                        if (speed_level < MAX_LEVEL) speed_level <= speed_level + 5'd1;
                    end else acc <= acc + 4'sd1;
                end else begin
                    if (acc <= -(EDGES_PER_DETENT - 1)) begin
                        acc <= 4'sd0;
                        if (speed_level > 5'd0) speed_level <= speed_level - 5'd1;
                    end else acc <= acc - 4'sd1;
                end
            end
        end
    end

    // ---- auto-start interval = CLK_HZ >> speed_level (level 0 = 1 Hz) ----
    // At high levels the interval is shorter than a generation, so the timer simply
    // waits for the generator to go idle -> effectively back-to-back (max rate).
    // NOTE: assign the integer parameter to a *sized* localparam first. A direct
    // bit-select on an `integer` parameter (CLK_HZ[31:0]) was synthesizing to a
    // bogus (tiny) interval under XST 14.7 -> auto_start fired ~66 Hz even at
    // speed_level 0 (the ~333 t/s board bug); behavioral/gate sim hid it.
    localparam [31:0] CLK_HZ_W = CLK_HZ;
    wire [31:0] interval = CLK_HZ_W >> speed_level;
    reg  [31:0] timer;
    always @(posedge clk) begin
        if (!resetn) begin
            timer      <= 32'd0;
            auto_start <= 1'b0;
        end else begin
            auto_start <= 1'b0;
            if (paused) begin
                timer <= 32'd0;                    // frozen: hold on current name
            end else if (timer >= interval) begin
                if (!gen_busy) begin               // fire only when the core is idle
                    auto_start <= 1'b1;
                    timer      <= 32'd0;
                end
            end else begin
                timer <= timer + 32'd1;
            end
        end
    end

endmodule
