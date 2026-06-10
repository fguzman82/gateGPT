// Embedding lookup: emb[i] = sat16( tok_embed[token][i] + pos_embed[pos][i] ),
// i = 0..N_EMBED-1, written to vmem[dst_base+i]. Token/pos embedding ROMs (Q5.11).
module embed #(
    parameter integer N_EMBED = 24
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [4:0]  token,
    input  wire [3:0]  pos,
    input  wire [9:0]  dst_base,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  signed [15:0] v_wdata,
    output reg         busy,
    output reg         done
);
    (* rom_style="distributed" *) reg signed [15:0] tok  [0:647];   // 27 x 24
    (* rom_style="distributed" *) reg signed [15:0] posr [0:383];   // 16 x 24
    initial begin
        $readmemh("/home/hermes/microgpt_fpga/generated/tok_embed.hex", tok);
        $readmemh("/home/hermes/microgpt_fpga/generated/pos_embed.hex", posr);
    end

    reg [6:0]  i;
    reg [9:0]  tbase, pbase;
    reg        st;

    wire signed [16:0] sum = $signed(tok[tbase + i]) + $signed(posr[pbase + i]);
    wire signed [15:0] esat =
        (sum >  17'sd32767) ? 16'sd32767 : (sum < -17'sd32768) ? -16'sd32768 : sum[15:0];

    always @(posedge clk) begin
        if (!resetn) begin st <= 0; busy <= 0; done <= 0; v_we <= 0; end
        else begin
            done <= 0; v_we <= 0;
            if (!st) begin
                if (start) begin
                    busy <= 1; i <= 0;
                    tbase <= token * N_EMBED;   // <= 26*24 = 624
                    pbase <= pos * N_EMBED;      // <= 15*24 = 360
                    st <= 1;
                end
            end else begin
                v_we <= 1; v_waddr <= dst_base + i; v_wdata <= esat;
                if (i == N_EMBED - 1) begin busy <= 0; done <= 1; st <= 0; end
                else i <= i + 1;
            end
        end
    end
endmodule
