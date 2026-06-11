// Independent microGPT inference core: a microcode-ROM sequencer driving modular
// datapath actuators. The program ROM (generated/ucode.hex) holds the schedule as
// macro-ops; the sequencer fetches one per step, starts the matching actuator, waits
// for done. INCREMENTAL decoding with a persistent KV cache: each call processes ONE
// new token at position pos_in (token_in), writing its K/V into the cache slot KC[pos]/
// VC[pos] (use_pos), and attends over positions 0..pos_in. The KC/VC cache lives in
// vmem and survives across calls. Bit-exact with tools/fixedpoint.QModel.logits_last.
module microgpt_core (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [4:0]  token_in,             // new token at this position
    input  wire [4:0]  pos_in,               // absolute position (0..BLOCK-1)
    input  wire        sample_mode,
    input  wire signed [15:0] inv_temp,      // (1/temperature) in Q5.11
    input  wire [31:0] rng_in,
    output reg         busy,
    output reg         done,
    output reg  [4:0]  next_token,
    output reg  [31:0] rng_out
);
`include "/home/hermes/microgpt_fpga/core/core_params.vh"
`include "/home/hermes/microgpt_fpga/core/coremap.vh"

    // ---------------- program ROM + fetch/decode ----------------
    (* rom_style="distributed" *) reg [71:0] prog [0:NINSTR-1];
    initial $readmemh("/home/hermes/microgpt_fpga/generated/ucode.hex", prog);

    reg [7:0]  pc;
    wire [71:0] instr   = prog[pc];
    wire [3:0]  op      = instr[3:0];
    wire [3:0]  wsel    = instr[7:4];
    wire [6:0]  in_dim  = instr[14:8];
    wire [6:0]  out_dim = instr[21:15];
    wire [4:0]  descale = instr[26:22];
    wire [1:0]  gsel    = instr[28:27];
    wire [9:0]  a_base  = instr[42:33];
    wire [9:0]  b_base  = instr[53:44];
    wire [9:0]  d_base  = instr[64:55];
    wire        use_pos = instr[66];

    // latched per-token inputs
    reg [4:0] tok_r, pos_r;
    wire [9:0] pos_off = pos_r * N_EMBED;                    // cache slot offset
    wire [9:0] mv_dst  = use_pos ? (d_base + pos_off) : d_base;

    // ---------------- shared vmem (true dual-port) ----------------
    // port A is the primary read (rdata=v_rdata) and norm's 2nd write; port B is the
    // primary write and norm's 2nd read (rdata=v_rdata_b).
    wire [9:0]  pa_addr, pb_addr;
    wire        pa_we, pb_we;
    wire signed [15:0] pa_wd, pb_wd, v_rdata, v_rdata_b;
    vmem2 #(.AW(10), .DW(16)) u_vmem (.clk(clk),
        .we_a(pa_we), .addr_a(pa_addr), .wdata_a(pa_wd), .rdata_a(v_rdata),
        .we_b(pb_we), .addr_b(pb_addr), .wdata_b(pb_wd), .rdata_b(v_rdata_b));

    // ---------------- actuators ----------------
    reg  em_go, no_go, mv_go, at_go, vo_go, sp_go;
    wire em_we; wire [9:0] em_wa; wire signed [15:0] em_wd; wire em_busy, em_done;
    embed #(.N_EMBED(N_EMBED)) u_embed (.clk(clk), .resetn(resetn), .start(em_go),
        .token(tok_r), .pos(pos_r[3:0]), .dst_base(d_base),
        .v_we(em_we), .v_waddr(em_wa), .v_wdata(em_wd), .busy(em_busy), .done(em_done));

    wire [9:0] no_aa, no_ab; wire no_wea, no_web; wire signed [15:0] no_wda, no_wdb;
    wire no_busy, no_done;
    wire [5:0] g_addr_a, g_addr_b; wire signed [15:0] g_rdata_a, g_rdata_b;
    norm #(.N(N_EMBED), .FRAC(FRAC_BITS)) u_norm (.clk(clk), .resetn(resetn), .start(no_go),
        .src_base(a_base), .dst_base(d_base), .gain_sel(gsel),
        .addr_a(no_aa), .rd_a(v_rdata), .we_a(no_wea), .wd_a(no_wda),
        .addr_b(no_ab), .rd_b(v_rdata_b), .we_b(no_web), .wd_b(no_wdb),
        .g_addr_a(g_addr_a), .g_addr_b(g_addr_b), .g_rdata_a(g_rdata_a), .g_rdata_b(g_rdata_b),
        .busy(no_busy), .done(no_done));
    grom u_grom (.sel(gsel), .addr_a(g_addr_a), .addr_b(g_addr_b),
        .gdata_a(g_rdata_a), .gdata_b(g_rdata_b));

    wire [9:0] mv_aa, mv_ab; wire mv_wea, mv_web; wire signed [15:0] mv_wda, mv_wdb;
    wire mv_busy, mv_done;
    wire [11:0] w_addr; wire [768-1:0] w_rdata;   // 24 lanes x 16-bit x 2 columns/cycle
    matvec u_mv (.clk(clk), .resetn(resetn), .start(mv_go),
        .wsel(wsel[2:0]), .in_dim(in_dim), .out_dim(out_dim),
        .act_base(a_base), .dst_base(mv_dst), .descale(descale),
        .addr_a(mv_aa), .rd_a(v_rdata), .we_a(mv_wea), .wd_a(mv_wda),
        .addr_b(mv_ab), .rd_b(v_rdata_b), .we_b(mv_web), .wd_b(mv_wdb),
        .w_addr(w_addr), .w_rdata(w_rdata), .busy(mv_busy), .done(mv_done));
    wrom u_wrom (.sel(wsel[2:0]), .addr(w_addr), .wdata(w_rdata));

    wire [9:0] at_ra, at_wa; wire at_we; wire signed [15:0] at_wd; wire at_busy, at_done;
    attn #(.N_EMBED(N_EMBED), .N_HEAD(N_HEAD), .HEAD_DIM(HEAD_DIM), .BLOCK(BLOCK), .FRAC(FRAC_BITS))
      u_attn (.clk(clk), .resetn(resetn), .start(at_go), .attn_scale(ATTN_SCALE),
        .ctx_len(pos_r + 5'd1),
        .q_base(A_QV), .k_base(A_KC), .v_base(A_VC), .o_base(A_AO),
        .v_raddr(at_ra), .v_rdata(v_rdata), .v_we(at_we), .v_waddr(at_wa), .v_wdata(at_wd),
        .busy(at_busy), .done(at_done));

    wire [9:0] vo_ra, vo_wa; wire vo_we; wire signed [15:0] vo_wd; wire vo_busy, vo_done;
    vecop u_vecop (.clk(clk), .resetn(resetn), .start(vo_go), .op(op == OP_RELU),
        .a_base(a_base), .b_base(b_base), .dst_base(d_base), .cnt(out_dim),
        .v_raddr(vo_ra), .v_rdata(v_rdata), .v_we(vo_we), .v_waddr(vo_wa), .v_wdata(vo_wd),
        .busy(vo_busy), .done(vo_done));

    wire [9:0] sp_ra; wire [4:0] sp_tok; wire [31:0] sp_rng; wire sp_busy, sp_done;
    sampler #(.VOCAB(VOCAB), .FRAC(FRAC_BITS)) u_samp (.clk(clk), .resetn(resetn), .start(sp_go),
        .sample_mode(sample_mode), .inv_temp(inv_temp), .rng_in(rng_in), .lm_base(A_LOG),
        .v_raddr(sp_ra), .v_rdata(v_rdata), .token(sp_tok), .rng_out(sp_rng),
        .busy(sp_busy), .done(sp_done));

    // ---------------- vmem dual-port mux (active actuator) ----------------
    // port A: primary read for every actuator (norm/matvec also write on it)
    assign pa_addr =
        (op == OP_NORM) ? no_aa : (op == OP_MATV) ? mv_aa :
        (op == OP_ATTN) ? at_ra : (op == OP_VADD || op == OP_RELU) ? vo_ra :
        (op == OP_SAMPLE) ? sp_ra : 10'd0;
    assign pa_we = (op == OP_NORM) ? no_wea : (op == OP_MATV) ? mv_wea : 1'b0;
    assign pa_wd = (op == OP_MATV) ? mv_wda : no_wda;
    // port B: primary write for every actuator (norm/matvec also read on it)
    assign pb_addr =
        (op == OP_NORM) ? no_ab : (op == OP_MATV) ? mv_ab : (op == OP_EMBED) ? em_wa :
        (op == OP_ATTN) ? at_wa : (op == OP_VADD || op == OP_RELU) ? vo_wa : 10'd0;
    assign pb_we =
        (op == OP_NORM) ? no_web : (op == OP_MATV) ? mv_web : (op == OP_EMBED) ? em_we :
        (op == OP_ATTN) ? at_we : (op == OP_VADD || op == OP_RELU) ? vo_we : 1'b0;
    assign pb_wd =
        (op == OP_NORM) ? no_wdb : (op == OP_MATV) ? mv_wdb : (op == OP_EMBED) ? em_wd :
        (op == OP_ATTN) ? at_wd : vo_wd;

    wire act_done =
        (op == OP_EMBED) ? em_done : (op == OP_NORM) ? no_done : (op == OP_MATV) ? mv_done :
        (op == OP_ATTN) ? at_done : (op == OP_VADD || op == OP_RELU) ? vo_done :
        (op == OP_SAMPLE) ? sp_done : 1'b1;

    // ---------------- sequencer ----------------
    localparam [1:0] Q_IDLE=0, Q_EXEC=1, Q_WAIT=2;
    reg [1:0] q;
    always @(posedge clk) begin
        if (!resetn) begin
            q <= Q_IDLE; pc <= 0; busy <= 0; done <= 0;
            em_go<=0; no_go<=0; mv_go<=0; at_go<=0; vo_go<=0; sp_go<=0;
        end else begin
            done <= 0;
            em_go<=0; no_go<=0; mv_go<=0; at_go<=0; vo_go<=0; sp_go<=0;
            case (q)
                Q_IDLE: if (start) begin
                    busy <= 1; pc <= 0; tok_r <= token_in; pos_r <= pos_in; q <= Q_EXEC;
                end
                Q_EXEC: begin
                    case (op)
                        OP_EMBED:  em_go <= 1;
                        OP_NORM:   no_go <= 1;
                        OP_MATV:   mv_go <= 1;
                        OP_ATTN:   at_go <= 1;
                        OP_VADD:   vo_go <= 1;
                        OP_RELU:   vo_go <= 1;
                        OP_SAMPLE: sp_go <= 1;
                        default: ;
                    endcase
                    if (op == OP_HALT) begin busy <= 0; done <= 1; q <= Q_IDLE; end
                    else q <= Q_WAIT;
                end
                Q_WAIT: if (act_done) begin
                    if (op == OP_SAMPLE) begin next_token <= sp_tok; rng_out <= sp_rng; end
                    pc <= pc + 1; q <= Q_EXEC;
                end
                default: q <= Q_IDLE;
            endcase
        end
    end
endmodule
