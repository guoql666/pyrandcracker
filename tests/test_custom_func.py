# -*- coding: utf-8 -*-
"""set_generator_func: recovery with discontinuous (skipped) observations."""
import random

import pytest

from pyrandcracker import RandCracker


def _getrows_skip_word(n16, skip, n8):
    def getRows(rnd):
        rows = []
        for _ in range(n16):
            rows += list(map(int, bin(rnd.getrandbits(16))[2:].zfill(16)))
        for _ in range(skip):
            rnd.getrandbits(32)             # skipped words: consumed, unseen
        for _ in range(n8):
            rows += list(map(int, bin(rnd.getrandbits(8))[2:].zfill(8)))
        return rows
    return getRows


def test_skip_recovery_with_custom_func():
    # 624 x width-16, skip one 32-bit word, then 1872 x width-8 -> rank 19937
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(16), 16)
    rd.getrandbits(32)                      # the skipped word
    for _ in range(1872):
        rc.submit(rd.getrandbits(8), 8)
    rc.set_generator_func(_getrows_skip_word(624, 1, 1872))
    assert rc.check() is True
    assert rc.rnd.getrandbits(16) == rd.getrandbits(16)
    assert rc.rnd.getrandbits(8) == rd.getrandbits(8)


def test_skip_recovery_offset_true():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data16 = [rd.getrandbits(16) for _ in range(624)]
    rd.getrandbits(32)
    data8 = [rd.getrandbits(8) for _ in range(1872)]
    for num in data16:
        rc.submit(num, 16)
    for num in data8:
        rc.submit(num, 8)
    rc.set_generator_func(_getrows_skip_word(624, 1, 1872))
    rc.check(offset=True)
    assert rc.rnd.getrandbits(16) == data16[0]


def test_broken_readme_scenario_raises_instead_of_wrong_answer():
    # the original README example dropped a 16-bit value then continued at
    # width 8; that schedule has rank 19935 and must be refused, not solved
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    for _ in range(624):
        rc.submit(rd.getrandbits(16), 16)
    rd.getrandbits(16)                      # dropped 16-bit value
    for _ in range(1248):
        rc.submit(rd.getrandbits(8), 8)

    def getRows(rnd):
        rows = []
        for _ in range(624):
            rows += list(map(int, bin(rnd.getrandbits(16))[2:].zfill(16)))
        rnd.getrandbits(16)                 # dropped
        for _ in range(1248):
            rows += list(map(int, bin(rnd.getrandbits(8))[2:].zfill(8)))
        return rows

    rc.set_generator_func(getRows)
    with pytest.raises(ValueError):
        rc.check()
