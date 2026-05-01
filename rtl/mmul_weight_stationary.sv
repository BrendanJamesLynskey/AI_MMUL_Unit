// =============================================================================
// mmul_weight_stationary.sv - Weight-Stationary MMUL (NVDLA / edge-NPU style)
// =============================================================================
// Weights B[k][c] are pre-loaded into a register file ("weight RF").  The
// caller then streams activation matrix A one K-step per cycle:
//
//   for k in 0..K-1:
//     for r in 0..M-1, c in 0..N-1: out[r][c] += A[r][k] * B[k][c]
//
// Inner loop is fully parallel; one cycle per K-step.  Total compute latency
// for an (MxK) x (KxN) tile is K cycles after weight load.
//
// Compared with output-stationary:
//   - lower latency (no skew), higher peak throughput per cycle
//   - higher operand-bandwidth requirement per cycle (M*N reads/cycle)
//   - weight reload is a separate phase, paid every fresh tile.
// =============================================================================

`timescale 1ns/1ps

module mmul_weight_stationary
  import mmul_pkg::*;
#(
  parameter int M = ARRAY_ROWS,
  parameter int N = ARRAY_COLS,
  parameter int K = K_DIM
)(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   clear,

  input  logic   load_w,
  input  logic [$clog2(K)-1:0] load_w_row,
  input  logic signed [N-1:0][DATA_WIDTH-1:0] w_row,

  input  logic   compute_en,
  input  logic [$clog2(K)-1:0] k_index,
  input  logic signed [M-1:0][DATA_WIDTH-1:0] a_col,

  output logic signed [M*N-1:0][ACC_WIDTH-1:0] acc
);

  // Flat weight RF: [k*N + n]  (avoids unpacked-array port issues in Icarus)
  logic signed [K*N-1:0][DATA_WIDTH-1:0] weight_rf;

  // Weight load (one row of B per cycle)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_rf <= '0;
    end else if (load_w) begin
      for (int nn = 0; nn < N; nn++) begin
        weight_rf[load_w_row*N + nn] <= w_row[nn];
      end
    end
  end

  // Compute / accumulate
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= '0;
    end else if (clear) begin
      acc <= '0;
    end else if (compute_en) begin
      for (int r = 0; r < M; r++) begin
        for (int c = 0; c < N; c++) begin
          acc[r*N + c] <= acc[r*N + c]
                       + ACC_WIDTH'(a_col[r] * weight_rf[k_index*N + c]);
        end
      end
    end
  end

endmodule : mmul_weight_stationary
