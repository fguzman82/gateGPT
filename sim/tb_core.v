// End-to-end core test: run the incremental generation loop (one token per core call
// at a growing absolute position, with the persistent KV cache) and compare the token
// sequence to the Python golden (greedy and sampled).
`timescale 1ns/1ps
module tb_core;
    reg clk = 0, resetn = 0, start = 0;
    always #5 clk = ~clk;

    reg [4:0]   token_in, pos_in;
    reg         smode;
    reg signed [15:0] inv_temp;
    reg [31:0]  rng_in;
    wire        busy, done;
    wire [4:0]  next_token;
    wire [31:0] rng_out;

    microgpt_core u_core (.clk(clk), .resetn(resetn), .start(start),
        .token_in(token_in), .pos_in(pos_in),
        .sample_mode(smode), .inv_temp(inv_temp), .rng_in(rng_in),
        .busy(busy), .done(done), .next_token(next_token), .rng_out(rng_out));

    integer step, errors;
    reg [4:0] seq [0:15];
    integer   slen;

    // measure cycles per token (core start -> done)
    integer cyc = 0; reg counting = 0; integer reported = 0;
    always @(posedge clk) begin
        if (start && !counting) begin counting <= 1; cyc <= 0; end
        else if (counting) cyc <= cyc + 1;
        if (done && counting) begin
            if (reported == 0) $display("CYCLES_PER_TOKEN = %0d", cyc);
            reported <= reported + 1; counting <= 0;
        end
    end

    // expected sequences (absolute-position model): greedy "alaya", sampled "rosphod"
    reg [4:0] exp_greedy [0:4];
    reg [4:0] exp_samp   [0:6];

    task run_gen(input mode, input [31:0] seed, input signed [15:0] itemp);
        begin
            token_in = 0; pos_in = 0; rng_in = seed; smode = mode; inv_temp = itemp; slen = 0;
            for (step = 0; step < 16; step = step + 1) begin
                @(negedge clk); start = 1; @(negedge clk); start = 0;
                wait (done); @(posedge clk); #1;
                if (next_token == 0) step = 100;          // stop on delimiter
                else begin
                    seq[slen] = next_token; slen = slen + 1;
                    token_in = next_token; pos_in = pos_in + 1;   // advance position
                    rng_in = rng_out;
                end
            end
        end
    endtask

    integer k;
    initial begin
        exp_greedy[0]=1; exp_greedy[1]=12; exp_greedy[2]=1; exp_greedy[3]=25; exp_greedy[4]=1;
        exp_samp[0]=18; exp_samp[1]=15; exp_samp[2]=19; exp_samp[3]=16; exp_samp[4]=8; exp_samp[5]=15; exp_samp[6]=4;
        errors = 0;
        repeat (6) @(posedge clk); resetn = 1; @(posedge clk);

        run_gen(1'b0, 32'd0, 16'sd0);     // greedy
        $write("greedy tokens:"); for (k=0;k<slen;k=k+1) $write(" %0d", seq[k]); $write("\n");
        if (slen != 5) begin $display("GREEDY LEN FAIL %0d", slen); errors=errors+1; end
        else for (k=0;k<5;k=k+1) if (seq[k]!==exp_greedy[k]) begin
            $display("GREEDY MISMATCH %0d got %0d exp %0d", k, seq[k], exp_greedy[k]); errors=errors+1; end

        run_gen(1'b1, 32'd2, 16'sd2926);  // sampled seed=2 T=0.7
        $write("sampled tokens:"); for (k=0;k<slen;k=k+1) $write(" %0d", seq[k]); $write("\n");
        if (slen != 7) begin $display("SAMP LEN FAIL %0d", slen); errors=errors+1; end
        else for (k=0;k<7;k=k+1) if (seq[k]!==exp_samp[k]) begin
            $display("SAMP MISMATCH %0d got %0d exp %0d", k, seq[k], exp_samp[k]); errors=errors+1; end

        if (errors == 0) $display("CORE PASS: greedy + sampled match golden");
        else             $display("CORE FAIL: %0d errors", errors);
        $finish;
    end
endmodule
