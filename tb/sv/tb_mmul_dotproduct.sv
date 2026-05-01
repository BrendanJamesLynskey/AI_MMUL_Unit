// =============================================================================
// tb_mmul_dotproduct.sv - Combinational K-wide dot-product testbench
// =============================================================================
`timescale 1ns/1ps

module tb_mmul_dotproduct;
  import mmul_pkg::*;

  localparam int K = 8;

  logic signed [K-1:0][DATA_WIDTH-1:0] a_vec;
  logic signed [K-1:0][DATA_WIDTH-1:0] b_vec;
  logic signed [ACC_WIDTH-1:0]         result;

  mmul_dotproduct #(.K(K)) dut (.*);

  int passed = 0;
  int failed = 0;
  acc_t expected_local;

  task automatic compute_ref;
    acc_t s = '0;
    for (int k = 0; k < K; k++) s += acc_t'(a_vec[k] * b_vec[k]);
    expected_local = s;
  endtask

  task automatic check(input string name);
    compute_ref;
    #1;
    if (result === expected_local) begin
      $display("  [PASS] %-30s ref=%0d got=%0d", name, $signed(expected_local), $signed(result));
      passed++;
    end else begin
      $display("  [FAIL] %-30s ref=%0d got=%0d", name, $signed(expected_local), $signed(result));
      failed++;
    end
  endtask

  initial begin
    $display("====================================================");
    $display(" tb_mmul_dotproduct : K=%0d", K);
    $display("====================================================");

    for (int k = 0; k < K; k++) begin a_vec[k] = '0; b_vec[k] = '0; end
    #1;
    check("all zeros");

    for (int k = 0; k < K; k++) begin a_vec[k] = 16'sh0100; b_vec[k] = 16'sh0100; end
    check("ones (1.0 * 1.0 * K)");

    for (int k = 0; k < K; k++) begin
      a_vec[k] = data_t'(k * 256);
      b_vec[k] = data_t'((K-k) * 256);
    end
    check("k * (K-k) pattern");

    for (int trial = 0; trial < 30; trial++) begin
      for (int k = 0; k < K; k++) begin
        a_vec[k] = data_t'($random);
        b_vec[k] = data_t'($random);
      end
      check($sformatf("random #%0d", trial));
    end

    $display("");
    $display("====================================================");
    $display(" tb_mmul_dotproduct : %0d PASSED, %0d FAILED", passed, failed);
    $display("====================================================");
    if (failed) $fatal(1, "tb_mmul_dotproduct failed");
    $finish;
  end

  initial begin
    #20000;
    $fatal(1, "tb_mmul_dotproduct TIMEOUT");
  end

endmodule
