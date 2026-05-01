"""
test_mmul_dotproduct.py - CocoTB tests for mmul_dotproduct.

K-wide combinational dot-product. Compared against the Python golden model.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import cocotb
from cocotb.triggers import Timer

from mmul_models import QFixed   # noqa: E402

Q = QFixed(int_bits=8, frac_bits=8)
W = Q.W
K = 8


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
    """Pack a list of signed-int element values into one big unsigned int (LSB-first)."""
    total = 0
    for i, v in enumerate(values):
        total |= to_unsigned(v, elem_w) << (i * elem_w)
    return total


def reference_dot(a_vec, b_vec):
    s = 0
    for a, b in zip(a_vec, b_vec):
        s += to_signed(a, W) * to_signed(b, W)
    return s & ((1 << 32) - 1)


@cocotb.test()
async def test_zeros(dut):
    dut.a_vec.value = 0
    dut.b_vec.value = 0
    await Timer(1, units="ns")
    assert int(dut.result.value) == 0


@cocotb.test()
async def test_ones(dut):
    a = [Q.to_int(1.0)] * K
    b = [Q.to_int(1.0)] * K
    dut.a_vec.value = pack_vec(a, W)
    dut.b_vec.value = pack_vec(b, W)
    await Timer(1, units="ns")
    expected = reference_dot(a, b)
    got = int(dut.result.value)
    assert got == expected, f"ones: got {got:#x} exp {expected:#x}"


@cocotb.test()
async def test_random_vectors(dut):
    random.seed(7)
    for trial in range(40):
        a_vals = [random.randint(-(1 << (W-1)), (1 << (W-1))-1) for _ in range(K)]
        b_vals = [random.randint(-(1 << (W-1)), (1 << (W-1))-1) for _ in range(K)]
        dut.a_vec.value = pack_vec(a_vals, W)
        dut.b_vec.value = pack_vec(b_vals, W)
        await Timer(1, units="ns")
        expected = reference_dot(a_vals, b_vals)
        got = int(dut.result.value)
        assert got == expected, f"trial {trial}: got {got:#x} exp {expected:#x}"
