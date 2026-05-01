// =============================================================================
// mmul_dotproduct.sv - Single-cycle K-wide dot-product unit
// =============================================================================
// K parallel signed multipliers feeding a balanced adder tree.  Conceptually
// represents one lane of an NVIDIA-style Tensor Core: one cycle to issue the
// 1xK x Kx1 inner product, latency = ceil(log2(K)) + 1 if registered.
//
// This implementation is purely combinational.  Wrap it in a flop for
// pipelining when target frequency is the limit.
// =============================================================================

`timescale 1ns/1ps

module mmul_dotproduct
  import mmul_pkg::*;
#(
  parameter int K = K_DIM
)(
  input  logic signed [K-1:0][DATA_WIDTH-1:0] a_vec,
  input  logic signed [K-1:0][DATA_WIDTH-1:0] b_vec,
  output logic signed [ACC_WIDTH-1:0]         result
);

  logic signed [ACC_WIDTH-1:0] prod [K];

  genvar gi;
  generate
    for (gi = 0; gi < K; gi++) begin : g_mul
      assign prod[gi] = ACC_WIDTH'(a_vec[gi] * b_vec[gi]);
    end
  endgenerate

  // Combinational reduction tree (synthesised to Wallace/Dadda)
  logic signed [ACC_WIDTH-1:0] sum_w;

  always_comb begin
    sum_w = '0;
    for (int k = 0; k < K; k++) begin
      sum_w = sum_w + prod[k];
    end
  end

  assign result = sum_w;

endmodule : mmul_dotproduct
