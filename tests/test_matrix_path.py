# -*- coding: utf-8 -*-
"""Generic (bit-packed GF(2)) solver paths for truncated / mixed widths."""
import random

import pytest

from pyrandcracker import RandCracker


def _recover(bits, count, seed=7):
    rd = random.Random()
    rd.seed(seed)
    rc = RandCracker()
    data = [rd.getrandbits(bits) for _ in range(count)]
    for num in data:
        rc.submit(num, bits)
    assert rc.check() is True
    return rd, rc, data


def test_16bit_minimum_horizon():
    # paper Table: n_min(16) = 1247
    rd, rc, data = _recover(16, 1247)
    assert rc.rnd.getrandbits(16) == rd.getrandbits(16)


def test_16bit_two_blocks():
    rd, rc, data = _recover(16, 1248)
    assert all(rc.rnd.getrandbits(16) == rd.getrandbits(16) for _ in range(10))


def test_8bit_minimum_horizon():
    # paper Table: n_min(8) = 2493
    rd, rc, data = _recover(8, 2493)
    assert rc.rnd.getrandbits(8) == rd.getrandbits(8)


def test_1bit_information_bound():
    # paper Table: n_min(1) = 19937
    rd, rc, data = _recover(1, 19937)
    assert rc.rnd.getrandbits(1) == rd.getrandbits(1)


def test_mixed_consecutive_widths():
    # paper "Schedule A": 624 x width-16 then 1248 x width-8 -> rank 19937
    rd = random.Random()
    rd.seed(11)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(16), 16)
    for _ in range(1248):
        rc.submit(rd.getrandbits(8), 8)
    assert rc.check() is True
    assert rc.rnd.getrandbits(16) == rd.getrandbits(16)
    assert rc.rnd.getrandbits(8) == rd.getrandbits(8)


def test_order_dependence_is_rank_deficient():
    # paper "Schedule B": 1248 x width-8 then 624 x width-16 -> rank 19275
    rd = random.Random()
    rd.seed(11)
    rc = RandCracker()
    for _ in range(1248):
        rc.submit(rd.getrandbits(8), 8)
    for _ in range(624):
        rc.submit(rd.getrandbits(16), 16)
    with pytest.raises(ValueError):
        rc.check()   # 19968 bits but still rank < 19937


def test_inconsistent_observation_raises():
    # 1247 x width-16 (19952 bits) carries 15 redundant boundary parity
    # relations (paper, Thm sixteen-bit budgets); interior calls vary freely,
    # so corrupt a BOUNDARY call (call 1) to break a relation.
    rd = random.Random()
    rd.seed(12)
    rc = RandCracker()
    data = [rd.getrandbits(16) for _ in range(1247)]
    data[0] ^= 1                 # flip bit 0 of call 1
    for num in data:
        rc.submit(num, 16)
    with pytest.raises(ValueError):
        rc.check()


def test_offset_true_matrix_path():
    rd = random.Random()
    rd.seed(13)
    rc = RandCracker()
    data = [rd.getrandbits(16) for _ in range(1248)]
    for num in data:
        rc.submit(num, 16)
    rc.check(offset=True)
    assert rc.rnd.getrandbits(16) == data[0]
    assert [rc.rnd.getrandbits(16) for _ in range(1247)] == data[1:]
