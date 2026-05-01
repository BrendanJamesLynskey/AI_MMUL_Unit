"""
test_mmul_models.py - Self-checking tests for the Python golden model.

Run:    python3 test_mmul_models.py
"""

import random
import sys

from mmul_models import (
    QFixed,
    naive_matmul,
    output_stationary,
    weight_stationary,
    input_stationary,
    dot_product_tree,
    quantise_matrix,
    dequantise_matrix,
    to_bf16, from_bf16,
    to_fp16, from_fp16,
    to_fp8, from_fp8,
    quantize_mx, MXBlock,
)


def _assert_eq_matrix(A, B, label):
    assert len(A) == len(B), f"{label}: row mismatch"
    for r, (ra, rb) in enumerate(zip(A, B)):
        assert len(ra) == len(rb), f"{label}: col mismatch at row {r}"
        for c, (a, b) in enumerate(zip(ra, rb)):
            assert a == b, f"{label}: mismatch at [{r}][{c}] {a:#x} != {b:#x}"


def test_basic_2x2():
    q = QFixed(8, 8)
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    A_q = quantise_matrix(A, q)
    B_q = quantise_matrix(B, q)
    expected = [[19.0, 22.0], [43.0, 50.0]]
    for fn, name in [(naive_matmul, "naive"),
                     (output_stationary, "output_stationary"),
                     (weight_stationary, "weight_stationary"),
                     (input_stationary, "input_stationary"),
                     (dot_product_tree, "dot_product_tree")]:
        if name == "output_stationary":
            C = fn(A_q, B_q, q, rows=2, cols=2)
        else:
            C = fn(A_q, B_q, q)
        Cf = dequantise_matrix(C, q)
        for r in range(2):
            for c in range(2):
                err = abs(Cf[r][c] - expected[r][c])
                assert err < 0.01, f"{name} 2x2 mismatch [{r}][{c}] = {Cf[r][c]} vs {expected[r][c]}"
    print("PASS  basic_2x2 - all 5 dataflows agree on golden output")


def test_random_consistency_across_dataflows(seed=42, M=4, K=4, N=4, trials=20):
    """All 5 dataflows must produce *bit-identical* outputs for fixed-point."""
    random.seed(seed)
    q = QFixed(8, 8)
    fails = 0
    for t in range(trials):
        A = [[random.uniform(-3, 3) for _ in range(K)] for _ in range(M)]
        B = [[random.uniform(-3, 3) for _ in range(N)] for _ in range(K)]
        A_q = quantise_matrix(A, q)
        B_q = quantise_matrix(B, q)
        ref = naive_matmul(A_q, B_q, q)
        for fn, name in [(output_stationary, "OS"),
                         (weight_stationary, "WS"),
                         (input_stationary, "IS"),
                         (dot_product_tree, "DPT")]:
            if name == "OS":
                C = fn(A_q, B_q, q, rows=M, cols=N)
            else:
                C = fn(A_q, B_q, q)
            try:
                _assert_eq_matrix(C, ref, name)
            except AssertionError as e:
                print(f"FAIL trial {t}: {e}")
                fails += 1
    if fails == 0:
        print(f"PASS  random_consistency_across_dataflows ({trials} trials, all dataflows agree)")
    else:
        print(f"FAIL  {fails} mismatch(es)")
        sys.exit(1)


def test_bf16_roundtrip():
    fails = 0
    cases = [0.0, 1.0, -1.0, 3.14159, 1e-4, -1e-4, 65504.0, -65504.0]
    for x in cases:
        b = to_bf16(x)
        y = from_bf16(b)
        # BF16 has only 7 mantissa bits => relative error <~ 1/256
        rel = abs(y - x) / (abs(x) + 1e-12)
        assert rel < 1/64 or abs(y - x) < 1e-3, f"bf16 round-trip failed: {x} -> {y}"
    print(f"PASS  bf16 round-trip ({len(cases)} cases)")


def test_fp16_roundtrip():
    cases = [0.0, 1.0, -1.0, 0.5, 3.14, 1e-4, 6.1e-5]
    for x in cases:
        b = to_fp16(x)
        y = from_fp16(b)
        rel = abs(y - x) / (abs(x) + 1e-12)
        assert rel < 1/512 or abs(y - x) < 1e-4, f"fp16: {x} -> {y}"
    print(f"PASS  fp16 round-trip ({len(cases)} cases)")


def test_fp8_e4m3():
    cases = [0.0, 1.0, -1.0, 0.5, 3.0, 5.0, -2.5]
    for x in cases:
        b = to_fp8(x, 4, 3)
        y = from_fp8(b, 4, 3)
        rel = abs(y - x) / (abs(x) + 1e-12)
        # FP8 e4m3: 3 mantissa bits => ~ 1/8 worst case
        assert rel < 1/4 or abs(y - x) < 0.1, f"fp8 e4m3: {x} -> {y}"
    print(f"PASS  fp8 e4m3 round-trip ({len(cases)} cases)")


def test_mx_block_quantise():
    """MX block-scaled: 32 elements share one E8M0 power-of-two scale."""
    K = MXBlock.K
    vals = [(i - K // 2) * 0.1 for i in range(K)]
    # Quantise into MXFP8 e4m3 elements, with element max-representable ~ 448.
    # E4M3 (bias-7, 3 mantissa) max-representable normal = 1.875 * 2^7 = 240.
    blk = quantize_mx(vals, lambda x: to_fp8(x, 4, 3), elem_max_repr=240.0)
    assert len(blk.elements) == K
    # Reconstruct
    scale = 2.0 ** (blk.scale_e8m0 - 127)
    recon = [from_fp8(e, 4, 3) * scale for e in blk.elements]
    for v, r in zip(vals, recon):
        assert abs(v - r) < 0.5, f"mx recon: {v} vs {r}"
    print(f"PASS  MX block quantise (K={K})")


def test_4x4_systolic_skew():
    """OS-systolic must produce identical answer to naive for a 4x4 case."""
    q = QFixed(8, 8)
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    B = [[1, 0, 0, 1],
         [0, 1, 0, 1],
         [0, 0, 1, 1],
         [1, 1, 1, 1]]
    Af = [[float(v) for v in r] for r in A]
    Bf = [[float(v) for v in r] for r in B]
    Aq = quantise_matrix(Af, q)
    Bq = quantise_matrix(Bf, q)
    ref = naive_matmul(Aq, Bq, q)
    sys_o = output_stationary(Aq, Bq, q, rows=4, cols=4)
    _assert_eq_matrix(sys_o, ref, "OS-systolic 4x4")
    print("PASS  4x4 OS systolic == naive matmul")


def main():
    print("=" * 60)
    print(" Python golden-model verification (mmul_models.py)")
    print("=" * 60)
    test_basic_2x2()
    test_4x4_systolic_skew()
    test_random_consistency_across_dataflows()
    test_bf16_roundtrip()
    test_fp16_roundtrip()
    test_fp8_e4m3()
    test_mx_block_quantise()
    print("=" * 60)
    print(" ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
