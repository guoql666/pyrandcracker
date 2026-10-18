# -*- coding: utf-8 -*-
"""Low-level MT19937 word operations (pure integers, no bit-list overhead).

This module follows the canonical prediction-state model of the paper
"State Recovery from Truncated Mersenne Twister Outputs":

    x_{t+n} = x_{t+m} XOR Psi(x_t & U_r, x_{t+1} & L_r),
    Psi(u, h) = (u >> 1) ^ (h >> 1) ^ ((h & 1) * a)

with MT19937 parameters n=624, m=397, w=32, r=31, a=0x9908B0DF.
The prediction state at time t is q_t = (x_t & U_r, x_{t+1}, ..., x_{t+n-1})
of dimension nw - r = 19937 (Proposition "Canonical prediction state"):
the low r bits of word 0 never influence the future output stream.

The twist-ready phase used by CPython is  rng.setstate((3, words + (624,), None));
its next call performs the in-place twist and emits temper(x_n).
"""
from __future__ import annotations

N = 624
M = 397
MATRIX_A = 0x9908B0DF
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7FFFFFFF
WORD_MASK = 0xFFFFFFFF
STATE_BITS = N * 32 - (32 - 1)  # 19968 raw bits, low 31 of word 0 irrelevant
CANON_BITS = N * 32 - 31        # 19937 canonical state bits (paper's bound)


# ---------------------------------------------------------------- tempering

def temper(y: int) -> int:
    """MT19937 tempering (matches CPython genrand_uint32 post-processing)."""
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y &= WORD_MASK
    y ^= y >> 18
    return y


def _inv_rshift(y: int, s: int) -> int:
    x = y
    for _ in range((32 + s - 1) // s):
        x = y ^ (x >> s)
    return x & WORD_MASK


def _inv_lshift(y: int, s: int, mask: int) -> int:
    x = y
    for _ in range((32 + s - 1) // s):
        x = y ^ ((x << s) & mask)
    return x & WORD_MASK


def untemper(y: int) -> int:
    """Exact inverse of :func:`temper`."""
    y = _inv_rshift(y, 18)
    y = _inv_lshift(y, 15, 0xEFC60000)
    y = _inv_lshift(y, 7, 0x9D2C5680)
    y = _inv_rshift(y, 11)
    return y


# ---------------------------------------------------------------- twist

def twist(mt: list) -> list:
    """One in-place MT19937 twist, C order; returns the list for chaining."""
    for kk in range(N - M):
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
        mt[kk] = mt[kk + M] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
    for kk in range(N - M, N - 1):
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)
        mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
    return mt


def untwist(mt: list) -> list:
    """Exact inverse of :func:`twist` (one block backwards)."""
    MT = list(mt)
    top = 1 << 31
    for i in range(N - 1, -1, -1):
        result = 0
        tmp = MT[i] ^ MT[(i + M) % N]
        if tmp & top:
            tmp ^= MATRIX_A
        result = (tmp << 1) & UPPER_MASK
        tmp = MT[(i - 1 + N) % N] ^ MT[(i + M - 1) % N]
        if tmp & top:
            tmp ^= MATRIX_A
            result |= 1
        result |= (tmp << 1) & LOWER_MASK
        MT[i] = result
    return MT


# ------------------------------------------------------- canonical state

def canonical_basis_words(index: int) -> list:
    """Raw word array of the `index`-th canonical basis state (0-based).

    Coordinates follow the paper: coordinate 0 is bit 31 (the only relevant
    bit) of word 0; coordinates 1..19936 enumerate words 1..623 from most
    significant to least significant bit.
    """
    words = [0] * N
    if index == 0:
        words[0] = UPPER_MASK
        return words
    j, b = divmod(index - 1, 32)
    words[j + 1] = 1 << (31 - b)
    return words


def words_to_canonical(words) -> int:
    """Pack the canonical coordinates of a word array into one integer."""
    x = (words[0] >> 31) & 1
    for j in range(1, N):
        x = (x << 32) | (words[j] & WORD_MASK)
    return x


def canonical_to_words(x: int) -> list:
    """Inverse of :func:`words_to_canonical`."""
    words = [0] * N
    words[0] = ((x >> (32 * 623)) & 1) * UPPER_MASK
    for j in range(1, N):
        words[j] = (x >> (32 * (623 - j))) & WORD_MASK
    return words


def canonical_words_from_bits(bits) -> list:
    """Pack a 19937-entry bool/0-1 vector of canonical coordinates into words."""
    words = [0] * N
    if bits[0]:
        words[0] = UPPER_MASK
    for c in range(1, len(bits)):
        if bits[c]:
            j, b = divmod(c - 1, 32)
            words[j + 1] |= 1 << (31 - b)
    return words


def set_state(rng, words, index: int = 624):
    """Put `rng` (a random.Random) into the twist-ready phase for `words`."""
    rng.setstate((3, tuple(int(w) & WORD_MASK for w in words) + (int(index),), None))


# ------------------------------------------------------- stream helpers

def stream_words(rng, count: int):
    """Return the next `count` raw 32-bit outputs of `rng` as one big integer.

    CPython fills the least-significant word first, so word t of the result
    (little-endian) is the t-th generated word.  A single getrandbits call
    generates the whole stream in C speed.
    """
    if count <= 0:
        return 0
    return rng.getrandbits(32 * count)


def words_to_numpy(big: int, count: int):
    """Little-endian big-int -> numpy uint32 array of generated words."""
    import numpy as np
    return np.frombuffer(big.to_bytes(4 * count, "little"), dtype="<u4")
