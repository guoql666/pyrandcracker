# -*- coding: utf-8 -*-
"""The paper's constructive five-bit recovery path."""
import random


import pytest

from pyrandcracker import RandCracker
from pyrandcracker import fivebit, mt19937


def top5_stream(words, count):
    """Top-5 tempered outputs starting from the twist-ready phase."""
    mt = list(words)
    out = []
    for t in range(count):
        if t % 624 == 0:
            mt = mt19937.twist(mt)
        out.append(mt19937.temper(mt[t % 624]) >> 27)
    return out


def test_fivebit_recovery():
    rd = random.Random()
    rd.seed(20)
    rc = RandCracker()
    data = [rd.getrandbits(5) for _ in range(4984)]
    for num in data:
        rc.submit(num, 5)
    assert rc.check() is True
    assert rc.rnd.getrandbits(5) == rd.getrandbits(5)


def test_fivebit_offset_true_replays_submissions():
    rd = random.Random()
    rd.seed(21)
    rc = RandCracker()
    data = [rd.getrandbits(5) for _ in range(4984)]
    for num in data:
        rc.submit(num, 5)
    rc.check(offset=True)
    assert rc.rnd.getrandbits(5) == data[0]
    assert [rc.rnd.getrandbits(5) for _ in range(497)] == data[1:498]


def test_fivebit_below_information_bound_returns_false():
    rc = RandCracker()
    for _ in range(3987):            # 3987*5 = 19935 < 19937
        rc.submit(0, 5)
    assert rc.check() is False


def test_fivebit_one_call_short_is_rank_deficient():
    # 4983 calls = 24915 bits >= 19937, but the paper's rank curve gives
    # r_5(4983) = 19936, so the state is not yet uniquely determined
    rd = random.Random()
    rd.seed(22)
    rc = RandCracker()
    for _ in range(4983):
        rc.submit(rd.getrandbits(5), 5)
    with pytest.raises(ValueError):
        rc.check()


def test_carrier_a_readout():
    # a_i: invisible for 3738 calls, then eps3*(I+S^227)u with u = delta_i
    for i in (1, 227, 397, 623):
        outs = top5_stream(fivebit.carrier_a(i), 4984)
        assert all(o == 0 for o in outs[:3738])
        d1 = outs[3738:4361]
        exp = [0] * 623
        exp[i - 1] ^= 4
        if i + 227 <= 623:
            exp[i + 227 - 1] ^= 4
        assert d1 == exp


def test_carrier_h_readout():
    # h_i: invisible for 4361 calls, then diagonal eps3 at position i
    for i in (1, 227, 397, 623):
        outs = top5_stream(fivebit.carrier_h(i), 4984)
        assert all(o == 0 for o in outs[:4361])
        d2 = outs[4361:4984]
        exp = [0] * 623
        exp[i - 1] = 4
        assert d2 == exp


def test_extension_state_readout():
    outs = top5_stream(fivebit.extension_state(), 4984)
    assert all(o == 0 for o in outs[:3738])
    assert outs[3738:4361] == fivebit.E_SYMBOLS
    assert outs[4361:4984] == fivebit.F_SYMBOLS
