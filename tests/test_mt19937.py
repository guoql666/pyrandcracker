# -*- coding: utf-8 -*-
"""Low-level MT19937 word operations."""
import random

import numpy as np
from pyrandcracker import mt19937 as mt


def test_temper_untemper_roundtrip():
    rng = random.Random(1)
    for _ in range(10000):
        x = rng.getrandbits(32)
        assert mt.untemper(mt.temper(x)) == x


def test_temper_matches_known_values():
    # tempering identities from the paper (Theorem full-word-schedules)
    assert mt.temper(0x80000000) == 0x88102204
    assert mt.temper(0x40000000) == 0x44081102


def test_twist_matches_cpython():
    r = random.Random(2)
    words = [r.getrandbits(32) for _ in range(624)]
    rng = random.Random()
    mt.set_state(rng, words)
    expect = [rng.getrandbits(32) for _ in range(624)]
    got = [mt.temper(w) for w in mt.twist(list(words))]
    assert got == expect


def test_untwist_inverts_twist_on_reachable_states():
    r = random.Random(3)
    for _ in range(5):
        raw = [r.getrandbits(32) for _ in range(624)]
        post = mt.twist(list(raw))          # a genuine post-twist array
        back = mt.untwist(post)
        # word 0 low 31 bits are never recoverable (paper Proposition 1)
        assert back[1:] == raw[1:]
        assert (back[0] & 0x80000000) == (raw[0] & 0x80000000)


def test_canonical_pack_roundtrip():
    r = random.Random(4)
    words = [r.getrandbits(32) for _ in range(624)]
    x = mt.words_to_canonical(words)
    assert x.bit_length() <= mt.CANON_BITS
    w2 = mt.canonical_to_words(x)
    assert w2[0] == (words[0] & 0x80000000)
    assert w2[1:] == words[1:]


def test_batch_stream_matches_per_call():
    r = random.Random(5)
    words = [r.getrandbits(32) for _ in range(624)]
    rng = random.Random()
    mt.set_state(rng, words)
    big = mt.stream_words(rng, 1000)
    arr = mt.words_to_numpy(big, 1000)
    rng2 = random.Random()
    mt.set_state(rng2, words)
    percall = np.array([rng2.getrandbits(32) for _ in range(1000)],
                       dtype=np.uint32)
    assert (arr == percall).all()


def test_canonical_basis_span():
    # every canonical basis state is independent: coordinate words differ
    seen = set()
    for i in range(mt.CANON_BITS):
        w = mt.canonical_basis_words(i)
        key = (w[0], tuple(w[1:]))
        assert key not in seen
        seen.add(key)
