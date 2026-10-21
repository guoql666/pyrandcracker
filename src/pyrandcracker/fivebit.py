# -*- coding: utf-8 -*-
"""Constructive five-bit state recovery (the paper's Section "Five-Bit
Residual Structure and Recovery").

For uniform top-5 observations the minimum horizon is 4984 calls:
3738 prefix calls followed by two readout blocks of 623 calls each.
The residual space R_5 = ker O_{5,3738} (dimension 1247) has the explicit
basis {a_i, h_i : 1 <= i <= 623} u {e_*}, where

    a_i = eta_i(0x00004080),  h_i = eta_i(0x00008100),

eta_i(xi) places 2^k*xi at word position i - k*(m-1) (k >= 0), and e_* is
the tabulated extension state.  After solving for ANY state x0 compatible
with the 3738-call prefix, the remaining 1247 dimensions are recovered with
a linear number of scalar/word additions:

    c    = coefficient of 2^3 in y_1                      (y = D1 residual)
    b_j  = coefficient of 2^2 in y_j + c*E_j
    u    = (I + S^227 + S^454) b                          (S = shift on F^623)
    gamma_j = beta(z_j + c*F_j) + u_j + u_{j+396}         (z = D2 residual)
    x     = x0 + sum_j u_j a_j + sum_j gamma_j h_j + c e_*

All constants below are verified against the word recurrence and tempering
(see tests); they reproduce Appendix B of the paper exactly.
"""
from __future__ import annotations

import random

from . import gf2
from .mt19937 import N, canonical_words_from_bits, set_state

WIDTH = 5
PREFIX = 3738          # k(5)
BLOCK = 623            # length of each readout block
HORIZON = PREFIX + 2 * BLOCK   # n_min(5) = 4984
ALPHA = 227            # n - m

XI0 = 0x00004080       # carrier word of the a-family (visible at call 3739)
XI1 = 0x00008100       # carrier word of the h-family (visible at call 4361)

# ----------------------------------------------------------- carrier states


def _eta(i: int, xi: int):
    words = [0] * N
    k = 0
    while i - k * 396 >= 1:
        words[i - k * 396] = (1 << k) * xi
        k += 1
    return words


def carrier_a(i: int):
    """a_i = eta_i(0x4080): words 0x4080 at i and 0x8100 at i-396 (i>=397)."""
    return _eta(i, XI0)


def carrier_h(i: int):
    """h_i = eta_i(0x8100): words 0x8100 at i and 0x10200 at i-396 (i>=397)."""
    return _eta(i, XI1)


def extension_state():
    """The single five-bit extension state e_* (paper Appendix B)."""
    e = [0] * N
    e[1] = 0x74111731
    for j in range(2, 225):
        e[j] = 0x64031531
    e[225] = 0xD612340E
    e[226] = 0xD23B0C1D
    e[227] = 0x65EA0618
    e[228] = 0x10120200
    e[621] = 0x80000000
    e[622] = 0x5B1C2CD6
    e[623] = 0xD9FC394B
    return e


def _first_block_symbols():
    """E_j as 5-bit integers (D1 response of e_*)."""
    E = [0] * BLOCK
    E[0] = 8 + 4 + 2                                  # j=1: eps2+eps3+eps4
    E[0] = 8 + 4 + 2                                  # j=1: eps2+eps3+eps4
    for j in list(range(2, 225)) + list(range(227, 452)) + [454]:
        E[j - 1] = 4 + 2                              # eps3+eps4
    for j in (225, 226, 452, 453):
        E[j - 1] = 2                                  # eps4
    for j in (455, 623):
        E[j - 1] = 8                                  # eps2
    E[621] = 16 + 8 + 2                               # j=622: eps1+eps2+eps4
    return E


def _second_block_symbols():
    """F_j as 5-bit integers (D2 response of e_*)."""
    F = [0] * BLOCK
    F[0] = 2
    for j in range(2, 225):
        F[j - 1] = 23
    F[224] = 8
    F[225] = 18
    F[226] = 31
    F[227] = 21
    F[621] = 7
    F[622] = 27
    return F


E_SYMBOLS = _first_block_symbols()
F_SYMBOLS = _second_block_symbols()


# ---------------------------------------------------------------- solving


def _validate(submissions):
    if len(submissions) < HORIZON:
        raise ValueError("five-bit recovery needs at least %d submissions, "
                         "got %d" % (HORIZON, len(submissions)))
    for _, bits in submissions[:HORIZON]:
        if bits != WIDTH:
            raise ValueError("five-bit solver requires uniform width-5 "
                             "submissions")


def _residual_coefficients(y, z):
    """c, u, gamma from the two residual readout blocks (paper Sec. 5.4)."""
    c = (y[0] >> 3) & 1
    b = [((y[j] ^ (c * E_SYMBOLS[j])) >> 2) & 1 for j in range(BLOCK)]
    u = [0] * BLOCK
    for j in range(BLOCK):                      # u = (I + S^227 + S^454) b
        jj = j + 1                              # 1-based position
        uj = b[j]
        if jj - ALPHA >= 1:
            uj ^= b[jj - ALPHA - 1]
        if jj - 2 * ALPHA >= 1:
            uj ^= b[jj - 2 * ALPHA - 1]
        u[j] = uj
    gamma = [0] * BLOCK
    for j in range(BLOCK):
        jj = j + 1
        g = ((z[j] ^ (c * F_SYMBOLS[j])) >> 2) & 1
        g ^= u[j]
        if jj + 396 <= BLOCK:
            g ^= u[jj + 396 - 1]
        gamma[j] = g
    return c, u, gamma


def _apply_residual(words, c, u, gamma):
    """x = x0 + sum_j u_j a_j + sum_j gamma_j h_j + c e_*."""
    if c:
        e = extension_state()
        for i in range(N):
            words[i] ^= e[i]
    for j in range(BLOCK):
        if u[j]:
            w = carrier_a(j + 1)
            for i in range(N):
                if w[i]:
                    words[i] ^= w[i]
        if gamma[j]:
            w = carrier_h(j + 1)
            for i in range(N):
                if w[i]:
                    words[i] ^= w[i]
    return words


def solve_fivebit(submissions, trange=range):
    """Recover the raw word array from >= 4984 uniform 5-bit submissions.

    ``submissions`` is a list of (value, 5) pairs from consecutive
    getrandbits(5) calls.  Returns the 624-word state whose twist-ready
    phase reproduces them.  Raises ValueError if the submissions are not
    self-consistent.
    """
    _validate(submissions)

    # 1. any state matching the 3738-output prefix (free variables = 0)
    x0 = canonical_words_from_bits(
        gf2.solveobservations(submissions, first_n=PREFIX,
                              require_unique=False, trange=trange))

    # 2. residual observations: D1 = calls 3739..4360, D2 = 4361..4983
    rng = random.Random()
    set_state(rng, x0)
    stream = rng.getrandbits(32 * HORIZON).to_bytes(4 * HORIZON, "little")
    resp = [stream[4 * t + 3] >> 3 for t in range(HORIZON)]
    obs = [num for num, _ in submissions]
    y = [obs[PREFIX + j] ^ resp[PREFIX + j] for j in range(BLOCK)]
    z = [obs[PREFIX + BLOCK + j] ^ resp[PREFIX + BLOCK + j]
         for j in range(BLOCK)]

    # 3./4. coefficients and state assembly
    c, u, gamma = _residual_coefficients(y, z)
    words = _apply_residual(list(x0), c, u, gamma)

    # 5. verify against every submitted observation
    check = random.Random()
    set_state(check, words)
    for t, (num, _) in enumerate(submissions):
        if check.getrandbits(WIDTH) != num:
            raise ValueError(
                "five-bit observations inconsistent at submission %d; they "
                "do not come from one consecutive MT19937 stream" % t)
    return words
