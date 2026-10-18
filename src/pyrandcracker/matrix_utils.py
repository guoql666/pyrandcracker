# -*- coding: utf-8 -*-
"""Dense GF(2) matrix utilities (legacy public API, corrected).

These helpers operate on ordinary 0/1 matrices.  The state-recovery hot
path uses the bit-packed :mod:`pyrandcracker.gf2` core instead; this
module keeps the historical function signatures for compatibility and for
small hand-written systems.

Fixes relative to the original release:

* ``solve_right``/``solve_left`` now raise ``ValueError`` when the system
  is underdetermined instead of silently returning one (possibly wrong)
  particular solution;
* the elimination no longer mixes pivot selection with already-used rows.
"""
from __future__ import annotations

import numpy as np


def gf2_gauss_elimination(matrix, trange=range):
    """Reduced row echelon form over GF(2).

    Returns (rref_matrix, pivot_cols, pivot_rows) where ``pivot_rows`` are
    indices into the ORIGINAL matrix of the rows that became pivots.
    """
    mat = np.array(matrix, dtype=np.uint8) % 2
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    m, n = mat.shape
    original_rows = np.arange(m)
    pivot_cols = []
    pivot_rows = []
    top = 0
    for current_col in trange(n):
        if top >= m:
            break
        col = mat[top:, current_col]
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        p = top + int(nz[0])
        if p != top:
            mat[[p, top]] = mat[[top, p]]
            original_rows[[p, top]] = original_rows[[top, p]]
        rows_to_fix = np.flatnonzero(mat[:, current_col])
        rows_to_fix = rows_to_fix[rows_to_fix != top]
        if rows_to_fix.size:
            mat[rows_to_fix] ^= mat[top]
        pivot_cols.append(current_col)
        pivot_rows.append(int(original_rows[top]))
        top += 1
    return mat, pivot_cols, pivot_rows


def gf2_solve(A, B):
    """Solve A x = B over GF(2) for square A of full rank."""
    A = np.array(A, dtype=np.uint8) % 2
    B = np.array(B, dtype=np.uint8).reshape(-1, 1) % 2
    r = A.shape[0]
    if A.shape[1] != r:
        raise ValueError("A must be square")
    aug = np.hstack((A, B))
    for i in range(r):
        nz = np.flatnonzero(aug[i:, i])
        if nz.size == 0:
            raise ValueError("No solution")
        p = i + int(nz[0])
        if p != i:
            aug[[i, p]] = aug[[p, i]]
        rows_to_fix = np.flatnonzero(aug[:, i])
        rows_to_fix = rows_to_fix[rows_to_fix != i]
        if rows_to_fix.size:
            aug[rows_to_fix] ^= aug[i]
    if np.any(aug[:, :r].sum(axis=0) == 0):
        raise ValueError("No solution")
    x = aug[:, -1]
    if np.any((aug[:, :r] @ x) % 2 != aug[:, -1]):
        raise ValueError("No solution")
    return x


def solve_right(T, B, check=True, trange=range):
    """Solve T x = B over GF(2); raises ValueError if not uniquely solvable."""
    T = np.array(T, dtype=np.uint8) % 2
    B = np.array(B, dtype=np.uint8).reshape(-1) % 2
    b, a = T.shape
    if B.shape != (b,):
        raise ValueError("B must be a vector of length equal to the rows of T")
    aug = np.hstack((T, B.reshape(-1, 1)))
    pivot_cols = []
    top = 0
    for c in trange(a):
        if top >= b:
            break
        nz = np.flatnonzero(aug[top:, c])
        if nz.size == 0:
            continue
        p = top + int(nz[0])
        if p != top:
            aug[[p, top]] = aug[[top, p]]
        rows_to_fix = np.flatnonzero(aug[:, c])
        rows_to_fix = rows_to_fix[rows_to_fix != top]
        if rows_to_fix.size:
            aug[rows_to_fix] ^= aug[top]
        pivot_cols.append(c)
        top += 1
    if len(pivot_cols) < a:
        raise ValueError("system is underdetermined (rank %d < %d unknowns); "
                         "no unique solution" % (len(pivot_cols), a))
    leftover = aug[top:]
    if leftover[:, :a].any():
        raise ValueError("system is underdetermined")
    if leftover[:, a].any():
        raise ValueError("No solution")
    x = np.zeros(a, dtype=np.uint8)
    for i, c in enumerate(pivot_cols):
        x[c] = aug[i, a]
    if check and not np.array_equal((T @ x) % 2, B):
        raise ValueError("Matrix equation has no solutions")
    return x


def solve_left(T, B, check=True, trange=range):
    """Solve x T = B over GF(2) (equivalently T^T x^T = B^T)."""
    T = np.array(T, dtype=np.uint8) % 2
    B = np.array(B, dtype=np.uint8) % 2
    return solve_right(T.T.copy(), B.reshape(-1), check=check, trange=trange)
