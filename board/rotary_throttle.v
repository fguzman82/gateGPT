// Rotary-encoder control for the name generator (Panasonic EVQWK4001, 15 detents).
// Turning the encoder adjusts ONE of two settings, selected by the push button:
//   cfg_mode = 0 (RATE) : level 0..MAX_LEVEL sets the auto-rotation interval,
//                         exponentially from 1 Hz (level 0) up to back-to-back.
//   cfg_mode = 1 (TEMP) : level 0..NTEMP-1 selects the sampling temperature (the top
//                         maps temp_sel -> inv_temp via a small LUT).
// A debounced PRESS toggles cfg_mode; the LED/LCD in the top shows which is active.
module rotary_throttle #(
    parameter integer CLK_HZ    = 50_000_000,  // core clock
    parameter integer MAX_LEVEL = 15,          // 1 rev (15 detents) spans the rate range
    parameter integer NTEMP     = 8,           // number of temperature presets
    parameter integer FILTER    = 2500         // ~50 us deglitch (cycles a level must hold)
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        rot_a,        // INCA (async, active high)
    input  wire        rot_b,        // INCB (async)
    input  wire        rot_push,     // PUSH (async, active high) -> toggles cfg_mode
    input  wire        gen_busy,     // name generator busy
    output reg         auto_start,   // 1-cycle pulse: launch a generation
    output reg  [4:0]  speed_level,  // RATE setting
    output reg  [2:0]  temp_sel,     // TEMP preset index (0..NTEMP-1)
    output reg         cfg_mode      // 0 = adjusting rate, 1 = adjusting temperature
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

    // ---- deglitch the two phase lines: accept a level only after FILTER stable cycles ----
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

    // ---- push button: debounce (~2 ms) + rising edge -> toggle cfg_mode ----
    localparam integer PUSH_FILTER = CLK_HZ / 500;     // ~2 ms
    reg        push_clean, push_clean_d;
    reg [19:0] push_cnt;
    always @(posedge clk) begin
        if (!resetn) begin
            push_clean <= 1'b0; push_clean_d <= 1'b0; push_cnt <= 20'd0; cfg_mode <= 1'b0;
        end else begin
            if (p_ff[1] == push_clean) push_cnt <= 20'd0;
            else if (push_cnt >= PUSH_FILTER[19:0]) begin push_clean <= p_ff[1]; push_cnt <= 20'd0; end
            else push_cnt <= push_cnt + 20'd1;
            push_clean_d <= push_clean;
            if (push_clean & ~push_clean_d) cfg_mode <= ~cfg_mode;   // toggle on press
        end
    end

    // ---- quadrature decode: accumulate signed edges, one step per detent ----
    localparam integer EDGES_PER_DETENT = 4;
    localparam integer STARTUP_HOLD     = CLK_HZ / 5;   // ~200 ms power-up/DCM-lock holdoff

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
    wire detent_up = armed && up_edge && !dn_edge && (acc >=  (EDGES_PER_DETENT - 1));
    wire detent_dn = armed && dn_edge && !up_edge && (acc <= -(EDGES_PER_DETENT - 1));

    always @(posedge clk) begin
        if (!resetn) begin
            speed_level <= 5'd0;
            temp_sel    <= 3'd2;                 // default preset (T=0.7 in the top's LUT)
            acc         <= 4'sd0;
            startup     <= STARTUP_HOLD[31:0];
        end else begin
            if (startup != 32'd0) startup <= startup - 32'd1;
            if (armed && (up_edge ^ dn_edge)) begin
                if (up_edge) begin
                    if (acc >= EDGES_PER_DETENT - 1) acc <= 4'sd0;
                    else acc <= acc + 4'sd1;
                end else begin
                    if (acc <= -(EDGES_PER_DETENT - 1)) acc <= 4'sd0;
                    else acc <= acc - 4'sd1;
                end
            end
            // a completed detent steps the SELECTED setting up or down
            if (detent_up) begin
                if (cfg_mode) begin if (temp_sel < NTEMP - 1) temp_sel <= temp_sel + 3'd1; end
                else          begin if (speed_level < MAX_LEVEL) speed_level <= speed_level + 5'd1; end
            end else if (detent_dn) begin
                if (cfg_mode) begin if (temp_sel > 3'd0) temp_sel <= temp_sel - 3'd1; end
                else          begin if (speed_level > 5'd0) speed_level <= speed_level - 5'd1; end
            end
        end
    end

    // ---- auto-start interval = CLK_HZ >> speed_level (level 0 = 1 Hz) ----
    localparam [31:0] CLK_HZ_W = CLK_HZ;
    wire [31:0] interval = CLK_HZ_W >> speed_level;
    reg  [31:0] timer;
    always @(posedge clk) begin
        if (!resetn) begin
            timer      <= 32'd0;
            auto_start <= 1'b0;
        end else begin
            auto_start <= 1'b0;
            if (timer >= interval) begin
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
