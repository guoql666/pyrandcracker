# -*- coding: utf-8 -*-
"""Fast-path (full-word) recovery, offset semantics and input validation."""
import random

import pytest

from pyrandcracker import RandCracker


def test_rand32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    assert rc.check() is True
    assert rc.rnd.getrandbits(32) == rd.getrandbits(32)


def test_offset32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    rc.offset(-1)
    assert rc.rnd.getrandbits(32) == data[-1]


def test_offset_after_consumption():
    # regression: the old implementation mis-rewound by 624 words here
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    n = rc.rnd.getrandbits(32)
    assert n == rd.getrandbits(32)
    rc.offset(-1)
    assert rc.rnd.getrandbits(32) == n


def test_deep_rollback_crosses_block_boundary():
    rd = random.Random()
    rd.seed(3)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(700)]
    for num in data:
        rc.submit(num)
    rc.check()
    rc.offset(-700)
    assert rc.rnd.getrandbits(32) == data[0]


def test_offset_true_replays_retained_submissions():
    rd = random.Random()
    rd.seed(4)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(624)]
    for num in data:
        rc.submit(num)
    rc.check(offset=True)
    assert rc.rnd.getrandbits(32) == data[0]
    assert [rc.rnd.getrandbits(32) for _ in range(623)] == data[1:]


def test_64_bit_submissions():
    rd = random.Random()
    rd.seed(5)
    rc = RandCracker()
    for _ in range(312):
        rc.submit(rd.getrandbits(64), 64)
    rc.check()
    assert rc.rnd.getrandbits(64) == rd.getrandbits(64)


def test_96_bit_submissions():
    rd = random.Random()
    rd.seed(6)
    rc = RandCracker()
    for _ in range(208):
        rc.submit(rd.getrandbits(96), 96)
    rc.check()
    assert rc.rnd.getrandbits(96) == rd.getrandbits(96)


def test_offset_bits_roundtrip():
    rd = random.Random()
    rd.seed(8)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(32))
    rc.check()
    n = rc.rnd.getrandbits(150)
    rc.offset_bits(-150)
    assert rc.rnd.getrandbits(150) == n
    m = rc.rnd.getrandbits(32)
    rc.offset_bits(-32)
    assert rc.rnd.getrandbits(32) == m
    rc.offset_bits(0)  # no-op


def test_forward_offset_bits():
    rd = random.Random()
    rd.seed(9)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(32))
    rc.check()
    rc.offset_bits(16)              # consumes one 32-bit word, like getrandbits(16)
    rd.getrandbits(16)              # consume the same word on the reference
    assert rc.rnd.getrandbits(16) == rd.getrandbits(16)


def test_submit_rejects_oversized_number():

    rc = RandCracker()
    with pytest.raises(ValueError):
        rc.submit(0x1FFFF, 16)   # 17-bit value in a 16-bit slot


def test_submit_rejects_negative_and_bad_bits():
    rc = RandCracker()
    with pytest.raises(ValueError):
        rc.submit(-1, 16)
    with pytest.raises(ValueError):
        rc.submit(0, 0)
    with pytest.raises(TypeError):
        rc.submit("1", 16)


def test_submit_after_check_raises():
    rd = random.Random()
    rd.seed(10)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(32))
    rc.check()
    with pytest.raises(RuntimeError):
        rc.submit(1, 32)


def test_check_returns_false_when_insufficient():
    rc = RandCracker()
    rc.submit(12345, 16)
    assert rc.check() is False


def test_get_random():
    rd = random.Random()
    rd.seed(11)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(32))
    rc.check()
    assert rc.get_random() is rc.rnd
    assert rc.get_random().getrandbits(32) == rd.getrandbits(32)
