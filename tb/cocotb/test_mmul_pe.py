"""
test_mmul_pe.py - CocoTB tests for mmul_pe (Processing Element).

Uses the Python golden model in ../../python/mmul_models.py for QFixed
arithmetic.  Verifies MAC, accumulate, signed multiply, clear, and the
A/W skew-register forwarding behaviour.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from mmul_models import QFixed   # noqa: E402

Q = QFixed(int_bits=8, frac_bits=8)
W = Q.W
ACC_W = 32


def to_unsigned(val: int, width: int) -> int:
    if val < 0:
        return (val + (1 << width)) & ((1 << width) - 1)
    return val & ((1 << width) - 1)


def to_signed(val: int, width: int) -> int:
    val &= (1 << width) - 1
    if val & (1 << (width - 1)):
        val -= 1 << width
    return val


async def reset(dut):
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_reset_clears_accumulator(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    assert int(dut.acc_out.value) == 0


@cocotb.test()
async def test_simple_mac(dut):
    """1 PE step: 2.0 * 3.0 -> 6.0 in Q8.8 (raw 0x60000)."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    dut.enable.value = 1
    dut.a_in.value = Q.to_int(2.0)
    dut.w_in.value = Q.to_int(3.0)
    await RisingEdge(dut.clk)
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    await RisingEdge(dut.clk)

    expected = 0x60000  # 6.0 in Q16.16
    assert int(dut.acc_out.value) == expected, \
        f"got {int(dut.acc_out.value):#x} expected {expected:#x}"


@cocotb.test()
async def test_accumulate_random_sequence(dut):
    """Drive 30 random MACs; accumulator must match the Python golden model."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    random.seed(123)

    acc = 0
    dut.enable.value = 1
    for _ in range(30):
        a_f = random.uniform(-2.0, 2.0)
        w_f = random.uniform(-2.0, 2.0)
        a_raw = Q.to_int(a_f)
        w_raw = Q.to_int(w_f)
        dut.a_in.value = a_raw
        dut.w_in.value = w_raw
        acc = Q.mul_acc(a_raw, w_raw, acc, ACC_W)
        await RisingEdge(dut.clk)

    dut.enable.value = 0
    await RisingEdge(dut.clk)

    got = int(dut.acc_out.value)
    assert got == acc, f"acc mismatch got={got:#x} exp={acc:#x}"


@cocotb.test()
async def test_clear(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    dut.enable.value = 1
    dut.a_in.value = Q.to_int(1.5)
    dut.w_in.value = Q.to_int(2.5)
    await RisingEdge(dut.clk)
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.w_in.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.acc_out.value) != 0

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    assert int(dut.acc_out.value) == 0


@cocotb.test()
async def test_a_w_forwarding(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0

    dut.enable.value = 1
    dut.a_in.value = 0xAAA
    dut.w_in.value = 0x123
    await RisingEdge(dut.clk)
    # cocotb's NBA write timing: drive one extra edge so the new inputs are
    # the ones latched into a_out / w_out.
    await RisingEdge(dut.clk)
    assert int(dut.a_out.value) == 0xAAA, f"a_out = {int(dut.a_out.value):#x}"
    assert int(dut.w_out.value) == 0x123, f"w_out = {int(dut.w_out.value):#x}"
