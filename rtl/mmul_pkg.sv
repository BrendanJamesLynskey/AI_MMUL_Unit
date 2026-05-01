// =============================================================================
// mmul_pkg.sv - Parameters and types for the AI MMUL Unit study
// =============================================================================
// Centralised constants used by all RTL and testbenches in this repo.
// Defaults match the SV/CocoTB tests but are designed to be overridden via
// the standard SystemVerilog parameter-override mechanism.
// =============================================================================

package mmul_pkg;

  // ---------- Datapath widths ----------
  parameter int DATA_WIDTH      = 16;             // Operand width (signed Q-fmt)
  parameter int FRAC_BITS       = 8;              // # fractional bits (Q8.8)
  parameter int ACC_WIDTH       = 32;             // Accumulator width
  parameter int PROD_WIDTH      = 2 * DATA_WIDTH; // Multiplier output width

  // ---------- Default array geometry ----------
  parameter int ARRAY_ROWS      = 4;
  parameter int ARRAY_COLS      = 4;
  parameter int K_DIM           = 4;              // inner dimension for tile

  // ---------- Convenience types ----------
  typedef logic signed [DATA_WIDTH-1:0]  data_t;
  typedef logic signed [PROD_WIDTH-1:0]  prod_t;
  typedef logic signed [ACC_WIDTH-1:0]   acc_t;

endpackage : mmul_pkg
