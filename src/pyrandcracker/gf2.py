# -*- coding: utf-8 -*-
"""Fast bit-packed GF(2) linear algebra for MT19937 state recovery.

The observation model follows the paper: each observed output bit is a
linear form over the 19937-dimensional canonical state space, and the rows
are P_{v_t} T A^t for the submission schedule (v_0, ..., v_{L-1}).

Implementation notes
--------------------
* Rows are packed into numpy ``uint64`` words; unknown ``c`` lives at word
  ``c // 64``, bit ``c % 64`` (LSB-first), and the right-hand side occupies
  bit 0 of the final word.
* The observation matrix is generated column-wise: for each canonical basis
  state the whole submission schedule is replayed with ONE
  ``getrandbits(32 * total_words)`` call (CPython emits the chronological
  words in the little-endian limb order), so building all 19937 columns
  costs seconds instead of minutes.
* Elimination is a forward sweep to echelon form with vectorized row XOR,
  followed by vectorized back-substitution.
"""
from __future__ import annotations

import random

import numpy as np

from .mt19937 import CANON_BITS, canonical_basis_words, set_state

U64 = np.uint64


# ------------------------------------------------------------------ schedule

class Schedule:
    """Precomputed bit layout of a submission list [(num, bits), ...].

    Every getrandbits(v) call consumes ceil(v/32) raw 32-bit words; its value
    is those words' window interpreted with the LAST word most significant
    (CPython fills wordarray from the least significant limb and truncates
    the last word from the left).  Observed bit p of submission t (0-based,
    most significant first) therefore corresponds to value bit q = v-1-p
    (LSB numbering), which lives in word q//32 at bit q%32, plus the
    truncation offset 32-top on the last word, where top = v - 32*(nwords-1).
    """

    def __init__(self, submissions):
        self.submissions = list(submissions)
        lengths = np.array([bits for _, bits in self.submissions], dtype=np.int64)
        nwords = -(-lengths // 32)
        top = lengths - 32 * (nwords - 1)
        self.word_off = np.concatenate(([0], np.cumsum(nwords)[:-1]))
        self.bit_off = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        self.total_words = int(nwords.sum())
        self.total_bits = int(lengths.sum())
        sub_idx = np.repeat(np.arange(len(self.submissions)), lengths)
        p = np.arange(self.total_bits) - np.repeat(self.bit_off, lengths)
        q = np.repeat(lengths - 1, lengths) - p       # value bit index, LSB order
        j = q >> 5
        last = j == np.repeat(nwords - 1, lengths)
        shift = np.where(last, 32 - np.repeat(top, lengths) + (q & 31), q & 31)
        self.base_word = self.word_off[sub_idx] + j
        self.shift = shift.astype(np.int64)

    def observed_bits(self) -> np.ndarray:
        """Pack the submitted numbers into a bool array (global bit order)."""
        y = np.zeros(self.total_bits, dtype=bool)
        for t, (num, bits) in enumerate(self.submissions):
            off = int(self.bit_off[t])
            for p in range(bits):
                y[off + p] = (num >> (bits - 1 - p)) & 1
        return y


# ------------------------------------------------------------- matrix build

def build_columns(schedule: Schedule, generator_func=None, trange=range):
    """Columns of the observation matrix over the canonical basis.

    Returns a uint8 array of shape (19937, ceil(total_bits/8)); row i holds
    the packed observation bits produced by canonical basis state i.
    """
    B = schedule.total_bits
    nbytes = -(-B // 8)
    cols = np.zeros((CANON_BITS, nbytes), dtype=np.uint8)
    rng = random.Random()
    for i in trange(CANON_BITS):
        set_state(rng, canonical_basis_words(i))
        if generator_func is None:
            big = rng.getrandbits(32 * schedule.total_words)
            arr = np.frombuffer(big.to_bytes(4 * schedule.total_words, "little"),
                                dtype="<u4")
            bits = ((arr[schedule.base_word] >> schedule.shift) & 1).astype(bool)
        else:
            bits = np.asarray(generator_func(rng), dtype=bool)
            if bits.size != B:
                raise ValueError("generator_func returned %d bits, expected %d"
                                 % (bits.size, B))
        cols[i] = np.packbits(bits)
    return cols


def columns_to_rows(cols: np.ndarray, total_bits: int, chunk: int = 2048) -> np.ndarray:
    """Transpose packed columns (nc, B/8) into packed rows (B, ceil(nc/64)+1).

    Unknown c of each row sits at word c//64, bit c%64; bit 0 of the final
    word is reserved for the right-hand side.
    """
    B = total_bits
    nc = cols.shape[0]
    nrows_u64 = (nc + 63) // 64 + 1
    rows = np.zeros((B, nrows_u64), dtype=np.uint64)
    rows8 = rows.view(np.uint8).reshape(B, nrows_u64 * 8)
    assert chunk % 8 == 0
    for c0 in range(0, nc, chunk):
        c1 = min(c0 + chunk, nc)
        block = np.unpackbits(cols[c0:c1], axis=1, count=B)     # (nc', B)
        packed = np.packbits(block.T, axis=1, bitorder='little')  # (B, ceil(nc'/8))
        rows8[:, c0 // 8: c0 // 8 + -(-(c1 - c0) // 8)] = packed
    return rows


def set_rhs(rows: np.ndarray, y: np.ndarray):
    """Write the observed bits into the reserved rhs bit of each row."""
    rhs_word = rows.shape[1] - 1
    rows[y, rhs_word] |= U64(1)


# -------------------------------------------------------------- elimination

def gauss_jordan(rows: np.ndarray, n_unknowns: int = CANON_BITS,
                 require_unique: bool = True, trange=range):
    """Forward Gauss elimination to echelon form over GF(2), in place.

    Returns (pivot_row_of_col, rank).  ``pivot_row_of_col[c]`` is the row
    index holding the pivot for column c, or -1 for a free column.  Follow
    with :func:`read_solution` for back-substitution and consistency checks.
    """
    pivot_row = np.full(n_unknowns, -1, dtype=np.int64)
    B = rows.shape[0]
    top = 0
    for c in trange(n_unknowns):
        if top >= B:
            break
        wc, bc = c >> 6, c & 63
        col = (rows[top:, wc] >> U64(bc)) & U64(1)
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        p = top + int(nz[0])
        if p != top:
            rows[[p, top]] = rows[[top, p]]
        # eliminate this column from all rows below the pivot
        mask = np.zeros(B, dtype=bool)
        mask[top + 1:] = (rows[top + 1:, wc] >> U64(bc)) & U64(1)
        if mask.any():
            rows[mask] ^= rows[top]
        pivot_row[c] = top
        top += 1
    if require_unique and (pivot_row < 0).any():
        raise ValueError(
            "observations do not determine the state uniquely "
            "(rank %d < %d); submit more bits" % (int((pivot_row >= 0).sum()), n_unknowns))
    return pivot_row, top


def read_solution(rows: np.ndarray, pivot_row: np.ndarray,
                  n_unknowns: int = CANON_BITS) -> np.ndarray:
    """Back-substitution and consistency check after :func:`gauss_jordan`.

    Free variables are set to zero (a particular solution).  Raises
    ValueError if the system is inconsistent (0 = 1 row).
    """
    rhs_word = rows.shape[1] - 1
    x = np.zeros(rows.shape[1] - 1, dtype=np.uint64)
    pivots = np.flatnonzero(pivot_row >= 0)[::-1]
    for c in pivots:
        r = int(pivot_row[c])
        v = int(np.bitwise_xor.reduce(rows[r, :rhs_word] & x))
        v ^= v >> 32
        v ^= v >> 16
        v ^= v >> 8
        v ^= v >> 4
        v ^= v >> 2
        v ^= v >> 1
        bit = (v & 1) ^ (int(rows[r, rhs_word]) & 1)
        if bit:
            x[c >> 6] |= U64(1) << U64(c & 63)
    used = np.zeros(rows.shape[0], dtype=bool)
    used[pivot_row[pivot_row >= 0]] = True
    leftover = rows[~used]
    if leftover.shape[0]:
        coeff_any = leftover[:, :rhs_word].any(axis=1)
        rhs_set = leftover[:, rhs_word].astype(bool)
        if bool((~coeff_any & rhs_set).any()):
            raise ValueError(
                "observations are inconsistent: no MT19937 state produces "
                "them (check that all submissions come from one consecutive "
                "stream)")
    out = np.zeros(n_unknowns, dtype=bool)
    for c in pivots:
        out[c] = bool((x[c >> 6] >> U64(c & 63)) & U64(1))
    return out


# ------------------------------------------------------------------- facade

def build_system(submissions, first_n=None, generator_func=None, trange=range):
    """Build the packed augmented rows (B, ceil(n/64)+1) plus the rhs vector.

    Splitting build from elimination lets alternate solvers (e.g. SageMath's
    M4RI through ``sagemath_support``) operate on the same compact rows.
    """
    if first_n is not None:
        submissions = submissions[:first_n]
    sched = Schedule(submissions)
    y = sched.observed_bits()
    cols = build_columns(sched, generator_func=generator_func, trange=trange)
    rows = columns_to_rows(cols, sched.total_bits)
    del cols
    set_rhs(rows, y)
    return rows, y


def solve_rows(rows, n_unknowns=CANON_BITS, require_unique=True, trange=range):
    """Gaussian-eliminate already-built augmented rows; returns a bool vector."""
    pivot_row, _ = gauss_jordan(rows, n_unknowns=n_unknowns,
                                require_unique=require_unique, trange=trange)
    return read_solution(rows, pivot_row, n_unknowns=n_unknowns)


def solveobservations(submissions, first_n=None, generator_func=None,
                      require_unique: bool = True, trange=range):
    """Solve for the canonical state from submissions; returns a bool vector.

    Set ``require_unique=False`` to obtain a particular solution of an
    underdetermined system (free variables zero).
    """
    rows, _ = build_system(submissions, first_n=first_n,
                           generator_func=generator_func, trange=trange)
    return solve_rows(rows, require_unique=require_unique, trange=trange)
