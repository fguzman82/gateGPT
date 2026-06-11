// XUPV5 (Virtex-5 XC5VLX110T) board top for the microGPT name generator.
//
// Demo: names auto-rotate on the 16x2 LCD. The ROTARY ENCODER is a throttle --
// turn it to set the rotation speed from 1 Hz (slow, readable) up to the core's
// maximum (back-to-back); press it to freeze the current name. Row 2 shows the
// measured throughput in tokens/second. The start button still forces one name,
// and the DIP switches perturb the random seed.
//
// PC-side over the USB JTAG cable: an optional ChipScope VIO core (CHIPSCOPE_VIO).
//
// Core runs at 50 MHz (DCM CLKDV). Real post-PAR Fmax ~52 MHz with weights loaded.
module xupv5_microgpt_top (
    input  wire        clk_100,      // 100 MHz board oscillator
    input  wire        rst_btn,      // reset push button (active high)
    input  wire        start_btn,    // "generate one" push button (active high)
    input  wire [7:0]  dip_sw,       // 8 DIP switches (seed bits)
    input  wire        rot_a,        // rotary INCA
    input  wire        rot_b,        // rotary INCB
    input  wire        rot_push,     // rotary push (freeze while held)
    output wire [7:0]  led,          // status LEDs (speed level + flags)
    // 16x2 character LCD (HD44780, 4-bit)
    output wire        lcd_rs,
    output wire        lcd_rw,
    output wire        lcd_e,
    output wire [3:0]  lcd_db        // DB[7:4]
);
    localparam integer CLK_HZ = 60_000_000;   // core runs at DCM CLKFX (100*3/5) = 60 MHz

    // ---------------- clocking: 100 MHz osc -> DCM CLKFX (x3/5) -> 60 MHz core ------
    // After the block-RAM + pipeline rework the core is ~75 MHz post-synth; clock it
    // at 60 MHz (CLKFX 3/5, 16.7 ns) with post-PAR margin. CLK0 is the DCM feedback.
    wire clk100_g, clk0, clk0_g, clkfx, clk, dcm_locked;
    IBUFG u_ibufg (.I(clk_100), .O(clk100_g));
    DCM_BASE #(
        .CLKIN_PERIOD(10.0),
        .CLKFX_MULTIPLY(3),      // CLKFX = 100 MHz * 3/5 = 60 MHz
        .CLKFX_DIVIDE(5)
    ) u_dcm (
        .CLKIN(clk100_g), .CLKFB(clk0_g), .RST(rst_btn),
        .CLK0(clk0), .CLKFX(clkfx),
        .CLK90(), .CLK180(), .CLK270(),
        .CLK2X(), .CLK2X180(), .CLKDV(), .CLKFX180(),
        .LOCKED(dcm_locked)
    );
    BUFG u_bufg0  (.I(clk0),  .O(clk0_g));   // CLK0 feedback (DCM deskew/lock)
    BUFG u_bufgfx (.I(clkfx), .O(clk));      // core clock = CLKFX = 60 MHz

    // ---------------- reset button: sync + debounce -------------------------------
    // rst_btn bounces on press/release; require the level stable for RST_FILTER
    // (~2 ms) before it changes the synchronous reset. DCM keeps the raw button on
    // its RST (it must reset at power-up and self-clears on lock).
    localparam integer RST_FILTER = 100000;        // ~2 ms @ 50 MHz
    reg [1:0]  rb_sync   = 2'd0;
    reg        rb_clean  = 1'b1;                    // start held in reset until lock
    reg [17:0] rb_cnt    = 18'd0;
    always @(posedge clk) begin
        rb_sync <= {rb_sync[0], rst_btn};
        if (rb_sync[1] == rb_clean)        rb_cnt <= 18'd0;
        else if (rb_cnt >= RST_FILTER)     begin rb_clean <= rb_sync[1]; rb_cnt <= 18'd0; end
        else                               rb_cnt <= rb_cnt + 18'd1;
    end
    wire resetn = dcm_locked & ~rb_clean;

    // ---------------- start button: sync + deglitch + one-cycle edge --------------
    // The cursor-button line glitches and was firing ~66 spurious presses/sec (a
    // ~330 t/s floor at rest). Require the level stable for BTN_FILTER (~2 ms) before
    // a press registers; a real press lasts far longer, glitches are rejected.
    localparam integer BTN_FILTER = 100000;        // ~2 ms @ 50 MHz
    reg [1:0]  sb_sync = 2'd0;
    reg        sb_clean = 1'b0, sb_clean_d = 1'b0;
    reg [17:0] sb_cnt = 18'd0;
    always @(posedge clk) begin
        sb_sync <= {sb_sync[0], start_btn};
        if (sb_sync[1] == sb_clean)            sb_cnt <= 18'd0;
        else if (sb_cnt >= BTN_FILTER)         begin sb_clean <= sb_sync[1]; sb_cnt <= 18'd0; end
        else                                   sb_cnt <= sb_cnt + 18'd1;
        sb_clean_d <= sb_clean;
    end
    wire btn_pulse = sb_clean & ~sb_clean_d;       // rising edge of the debounced press

    // ---------------- seed source: free-running counter ---------------------------
    reg [31:0] seed_live = 32'd1;
    always @(posedge clk) seed_live <= seed_live + 32'd1;

    // ---------------- VIO controls (default standalone) ---------------------------
    wire        vio_start, vio_use_host;
    wire [31:0] vio_seed;
    wire [15:0] vio_temp;

    // ---------------- name generator ----------------------------------------------
    wire        gen_busy, gen_done, tok_valid;
    wire [7:0]  tok_out;
    wire [4:0]  name_len;
    wire [(16*8)-1:0] name_flat;

    // rotary throttle drives auto-rotation
    wire        auto_start;
    wire [4:0]  speed_level;
    rotary_throttle #(.CLK_HZ(CLK_HZ)) u_rot (
        .clk(clk), .resetn(resetn),
        .rot_a(rot_a), .rot_b(rot_b), .rot_push(rot_push),
        .gen_busy(gen_busy),
        .auto_start(auto_start), .speed_level(speed_level)
    );

    // Generation trigger = rotary throttle (auto-rotation) + optional host (VIO).
    // start_btn (AJ6) is intentionally NOT a trigger: on this board the line
    // free-runs at ~66 Hz, which sailed straight through the 2 ms debounce (a 66 Hz
    // square wave is stable >2 ms per half-period) and produced a ~330 t/s floor.
    // Proven by the gen_start=0 bring-up test (LCD frozen at the banner, 0 t/s -> the
    // generator never self-runs), so the only source was btn_pulse. The throttle is
    // bounded to 1 Hz at level 0 by construction, so the demo now starts at ~5 t/s.
    wire        gen_start = auto_start | vio_start;
    wire        _unused   = btn_pulse;   // start_btn kept wired/debounced but unused
    wire [31:0] gen_seed  = vio_use_host ? vio_seed : (seed_live ^ {24'd0, dip_sw});
    wire signed [15:0] gen_inv_temp = 16'sd2926;   // (1/0.7) in Q5.11 -> T=0.7

    name_generator #(.MAX_LEN(16)) u_gen (
        .clk(clk), .resetn(resetn), .start(gen_start),
        .seed(gen_seed), .inv_temp(gen_inv_temp), .sample_mode(1'b1),
        .busy(gen_busy), .done(gen_done), .token_out(tok_out), .token_valid(tok_valid),
        .name_len(name_len), .name_flat(name_flat)
    );

    // latch the last completed name so the LCD shows a stable string between names
    reg [(16*8)-1:0] name_show;
    reg [4:0]        name_len_show;
    always @(posedge clk) begin
        if (!resetn) begin name_show <= {16{8'd0}}; name_len_show <= 5'd0; end
        else if (gen_done) begin name_show <= name_flat; name_len_show <= name_len; end
    end

    // ---------------- tokens/second meter -----------------------------------------
    wire [19:0] tok_bcd;
    tok_meter #(.CLK_HZ(CLK_HZ)) u_meter (
        .clk(clk), .resetn(resetn), .token_valid(tok_valid),
        .tok_bcd(tok_bcd)
    );

    // ---------------- LCD line 1: welcome banner, then the generated name ---------
    reg show_welcome;
    always @(posedge clk) begin
        if (!resetn)      show_welcome <= 1'b1;
        else if (gen_done) show_welcome <= 1'b0;
    end

    function [7:0] tok_ascii;     // token 0..25 -> 'a'..'z', else space
        input [7:0] t;
        tok_ascii = (t < 8'd26) ? (8'd97 + t) : 8'h20;
    endfunction

    // welcome banner as a flat constant, col0 in the LSB byte (reversed literal)
    wire [(16*8)-1:0] welcome_str = "   omed TPGorcim";   // = "microGPT demo   "

    wire [(16*8)-1:0] line1;
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : GEN_LINE1
            assign line1[(gi*8) +: 8] =
                show_welcome ? welcome_str[(gi*8) +: 8] :
                (gi < name_len_show) ? tok_ascii(name_show[(gi*8) +: 8]) : 8'h20;
        end
    endgenerate

    // ---------------- LCD line 2: "rate: NNNNN t/s" (measured tok/s, BCD) ---------
    wire [3:0] d4 = tok_bcd[19:16];
    wire [3:0] d3 = tok_bcd[15:12];
    wire [3:0] d2 = tok_bcd[11:8];
    wire [3:0] d1 = tok_bcd[7:4];
    wire [3:0] d0 = tok_bcd[3:0];
    wire z4 = (d4 == 0);
    wire z3 = z4 & (d3 == 0);
    wire z2 = z3 & (d2 == 0);
    wire z1 = z2 & (d1 == 0);
    wire [7:0] c4 = z4 ? 8'h20 : (8'h30 + d4);   // blank leading zeros
    wire [7:0] c3 = z3 ? 8'h20 : (8'h30 + d3);
    wire [7:0] c2 = z2 ? 8'h20 : (8'h30 + d2);
    wire [7:0] c1 = z1 ? 8'h20 : (8'h30 + d1);
    wire [7:0] c0 =                8'h30 + d0;    // always >=1 digit
    // bytes (col0..col15): r a t e :  _  c4 c3 c2 c1 c0  _  t / s _
    wire [(16*8)-1:0] line2 = {
        8'h20, 8'h73, 8'h2f, 8'h74, 8'h20,            // [15..11] ' ' s / t ' '
        c0, c1, c2, c3, c4,                           // [10..6]
        8'h20, 8'h3a, 8'h65, 8'h74, 8'h61, 8'h72      // [5..0]  ' ' : e t a r
    };

    // ---------------- LCD ---------------------------------------------------------
    wire lcd_ready;
    lcd_hd44780 #(.CLK_HZ(CLK_HZ)) u_lcd (
        .clk(clk), .resetn(resetn), .line1(line1), .line2(line2),
        .lcd_rs(lcd_rs), .lcd_rw(lcd_rw), .lcd_e(lcd_e), .lcd_db(lcd_db),
        .ready(lcd_ready)
    );

    // ---------------- LED status + timebase heartbeat -----------------------------
    // led[7] = independent 1 Hz heartbeat (toggles every 0.5 s) -> sanity-checks that
    //          the core clock really is ~80 MHz. If this blinks ~1/s, the timebase is
    //          right and any fast generation is a throttle-interval problem, not clock.
    // led[6] = gen_busy (at 1 Hz it blinks briefly; if solid-on, generation is back-to-back).
    // led[4:0] = speed_level.
    reg [25:0] hb_cnt = 26'd0;
    reg        hb     = 1'b0;
    always @(posedge clk) begin
        if (hb_cnt >= 26'd29_999_999) begin hb_cnt <= 26'd0; hb <= ~hb; end   // 0.5 s @ 60 MHz
        else                                hb_cnt <= hb_cnt + 26'd1;
    end
    assign led = {hb, gen_busy, 1'b0, speed_level};

    // ---------------- ChipScope VIO (PC over USB JTAG) ----------------------------
`ifdef CHIPSCOPE_VIO
    wire [35:0]  vio_control;
    wire [49:0]  vio_sync_out;
    wire [138:0] vio_sync_in;
    reg  [1:0] vstart_sync = 2'd0;
    always @(posedge clk) vstart_sync <= {vstart_sync[0], vio_sync_out[0]};
    assign vio_start    = (vstart_sync == 2'b01);
    assign vio_use_host = vio_sync_out[1];
    assign vio_seed     = vio_sync_out[33:2];
    assign vio_temp     = vio_sync_out[49:34];
    assign vio_sync_in  = {2'd0, speed_level, lcd_ready, gen_busy, name_len_show, name_show};
    chipscope_icon u_icon (.CONTROL0(vio_control));
    chipscope_vio  u_vio  (.CONTROL(vio_control), .CLK(clk),
                           .SYNC_IN(vio_sync_in), .SYNC_OUT(vio_sync_out));
`else
    assign vio_start = 1'b0; assign vio_use_host = 1'b0;
    assign vio_seed  = 32'd0; assign vio_temp = 16'd0;
`endif

endmodule
