// =============================================================================
// mmul_systolic_os.sv - Output-Stationary Systolic Array (TPU style)
// =============================================================================
// ROWS x COLS grid of mmul_pe instances.  Inputs and outputs are packed multi-
// dimensional vectors for compatibility with Icarus Verilog and synthesis
// tools that have weak support for unpacked-array ports.
//
//        b_in[0]    b_in[1]    b_in[2]   ...
//           │          │          │
//           ▼          ▼          ▼
//   a_in[0]►PE────────►PE────────►PE──► ...
//           │          │          │
//   a_in[1]►PE────────►PE────────►PE──► ...
//           │          │          │
//           ▼          ▼          ▼
//
// A streams left -> right (row activations); B streams top -> bottom (column
// weights).  Each PE accumulates a single C[r][c] over multiple cycles, hence
// "output stationary".
//
// Caller responsibility: skew the inputs.  See tb_mmul_systolic_os.sv.
// =============================================================================

`timescale 1ns/1ps

module mmul_systolic_os
  import mmul_pkg::*;
#(
  parameter int ROWS = ARRAY_ROWS,
  parameter int COLS = ARRAY_COLS
)(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   clear,
  input  logic   enable,

  // Packed: a_in[ROWS-1 : 0], each element DATA_WIDTH wide
  input  logic signed [ROWS-1:0][DATA_WIDTH-1:0]  a_in,
  input  logic signed [COLS-1:0][DATA_WIDTH-1:0]  b_in,

  // Packed result: [r*COLS + c] indexes into ROWS*COLS array
  output logic signed [ROWS*COLS-1:0][ACC_WIDTH-1:0] result
);

  // Internal mesh wiring (use simple 2D arrays - inferred to flops via PE)
  logic signed [DATA_WIDTH-1:0] a_wire [ROWS][COLS+1];
  logic signed [DATA_WIDTH-1:0] w_wire [ROWS+1][COLS];

  genvar gr, gc;
  generate
    for (gr = 0; gr < ROWS; gr++) begin : g_a_edge
      assign a_wire[gr][0] = a_in[gr];
    end
    for (gc = 0; gc < COLS; gc++) begin : g_b_edge
      assign w_wire[0][gc] = b_in[gc];
    end
  endgenerate

  generate
    for (gr = 0; gr < ROWS; gr++) begin : g_row
      for (gc = 0; gc < COLS; gc++) begin : g_col
        mmul_pe u_pe (
          .clk     (clk),
          .rst_n   (rst_n),
          .clear   (clear),
          .enable  (enable),
          .a_in    (a_wire[gr][gc]),
          .w_in    (w_wire[gr][gc]),
          .a_out   (a_wire[gr][gc+1]),
          .w_out   (w_wire[gr+1][gc]),
          .acc_out (result[gr*COLS + gc])
        );
      end
    end
  endgenerate

endmodule : mmul_systolic_os
