// =============================================================================
// mmul_pe.sv - Output-Stationary Processing Element (single MAC)
// =============================================================================
// One PE of an output-stationary systolic array:
//
//      a_in  ──►┌─────────┐──► a_out  (passes A south on the next clock)
//               │  PE     │
//      w_in  ──►│ a*w + acc├──► w_out  (passes B east on the next clock)
//               └─────────┘
//                    │ acc_out  (output-stationary partial sum)
//
// Each cycle when `enable` is high:
//   1) Latch a_in -> a_out and w_in -> w_out (skew register).
//   2) Add the (signed) product a_in * w_in to the accumulator.
// `clear` zeros the accumulator without touching the skew registers.
// `rst_n` is an asynchronous active-low reset.
//
// The multiplier and adder are both purely combinational; only the three
// flops (a_out, w_out, acc) are stateful.  This matches the canonical
// systolic PE described in Kung & Leiserson (1979) and used in TPUv1.
// =============================================================================

`timescale 1ns/1ps

module mmul_pe
  import mmul_pkg::*;
(
  input  logic   clk,
  input  logic   rst_n,
  input  logic   clear,
  input  logic   enable,

  input  data_t  a_in,
  input  data_t  w_in,

  output data_t  a_out,
  output data_t  w_out,
  output acc_t   acc_out
);

  // Combinational signed multiply
  prod_t prod;
  assign prod = a_in * w_in;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_out   <= '0;
      w_out   <= '0;
      acc_out <= '0;
    end else if (clear) begin
      a_out   <= '0;
      w_out   <= '0;
      acc_out <= '0;
    end else if (enable) begin
      a_out   <= a_in;
      w_out   <= w_in;
      // Sign-extend product to ACC_WIDTH and accumulate.
      acc_out <= acc_out + acc_t'(prod);
    end
  end

endmodule : mmul_pe
