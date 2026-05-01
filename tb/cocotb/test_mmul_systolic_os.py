"""
test_mmul_systolic_os.py - CocoTB test for the 4x4 output-stationary
systolic array.  Drives the canonical skew schedule and compares the
final accumulator readout against the Python golden model.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from mmul_models import QFixed, naive_matmul, quantise_matrix   # noqa: E402

Q = QFixed(8, 8)
W = Q.W
ROWS = 4
COLS = 4
K_DIM = 4


def to_unsigned(v: int, w: int) -> int:
    if v < 0:
        v += 1 << w
    return v & ((1 << w) - 1)


def to_signed(v: int, w: int) -> int:
    v &= (1 << w) - 1
    if v & (1 << (w - 1)):
        v -= 1 << w
    return v


def pack_vec(values, elem_w):
    out = 0
    for i, v in enumerate(values):
        out |= to_unsigned(v, elem_w) << (i * elem_w)
    return out


def unpack_result(packed_value, n_elem, elem_w):
    raw = int(packed_value)
    return [(raw >> (i * elem_w)) & ((1 << elem_w) - 1) for i in range(n_elem)]


async def reset(dut):
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def clear_array(dut):
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)


async def run_skewed_tile(dut, A_q, B_q):
    total = K_DIM + ROWS + COLS - 2 + 4
    dut.enable.value = 1
    for t in range(total):
        a_vals = []
        for r in range(ROWS):
            kk = t - r
            a_vals.append(A_q[r][kk] if 0 <= kk < K_DIM else 0)
        b_vals = []
        for c in range(COLS):
            kk = t - c
            b_vals.append(B_q[kk][c] if 0 <= kk < K_DIM else 0)
        dut.a_in.value = pack_vec(a_vals, W)
        dut.b_in.value = pack_vec(b_vals, W)
        await RisingEdge(dut.clk)
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    await RisingEdge(dut.clk)


def random_matrix_q(rows, cols, lo=-3.0, hi=3.0):
    floats = [[random.uniform(lo, hi) for _ in range(cols)] for _ in range(rows)]
    return quantise_matrix(floats, Q)


@cocotb.test()
async def test_2x2_worked_example(dut):
    """[[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]; pad to 4x4."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    A = [[0]*K_DIM for _ in range(ROWS)]
    B = [[0]*COLS for _ in range(K_DIM)]
    A[0][0] = Q.to_int(1.0); A[0][1] = Q.to_int(2.0)
    A[1][0] = Q.to_int(3.0); A[1][1] = Q.to_int(4.0)
    B[0][0] = Q.to_int(5.0); B[0][1] = Q.to_int(6.0)
    B[1][0] = Q.to_int(7.0); B[1][1] = Q.to_int(8.0)

    await clear_array(dut)
    await run_skewed_tile(dut, A, B)

    expected = naive_matmul(A, B, Q)
    got = unpack_result(dut.result.value, ROWS*COLS, 32)
    for r in range(ROWS):
        for c in range(COLS):
            idx = r*COLS + c
            assert got[idx] == expected[r][c], \
                f"C[{r}][{c}] got={to_signed(got[idx],32)} exp={to_signed(expected[r][c],32)}"


@cocotb.test()
async def test_random_4x4_tiles(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    random.seed(11)

    for trial in range(5):
        A = random_matrix_q(ROWS, K_DIM)
        B = random_matrix_q(K_DIM, COLS)
        await clear_array(dut)
        await run_skewed_tile(dut, A, B)
        expected = naive_matmul(A, B, Q)
        got = unpack_result(dut.result.value, ROWS*COLS, 32)
        for r in range(ROWS):
            for c in range(COLS):
                idx = r*COLS + c
                assert got[idx] == expected[r][c], \
                    (f"trial {trial} C[{r}][{c}] got={to_signed(got[idx],32)} "
                     f"exp={to_signed(expected[r][c],32)}")
