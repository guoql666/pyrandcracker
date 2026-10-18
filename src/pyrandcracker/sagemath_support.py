# -*- coding: utf-8 -*-
"""SageMath acceleration hook for the GF(2) elimination step.

The state-recovery pipeline is split into

    build_columns / columns_to_rows   (packed numpy uint64 rows)
        -> eliminate + back-substitute

When SageMath is installed, :func:`solve_gf2_sage` replaces the second step
with Sage's M4RI elimination over GF(2).  Construction of the Sage matrix
costs O(rows * cols) Python objects, so for state-sized systems the numpy
core is usually faster; the Sage path is kept as the drop-in accelerator /
independent cross-check that ``force_sage=True`` selects, and it falls back
to numpy automatically when Sage is missing or the system is too large.

Row layout (shared with :mod:`pyrandcracker.gf2`): unknown ``c`` lives at
word ``c // 64``, bit ``c % 64``, and the right-hand side is bit 0 of the
final word.
"""
from __future__ import annotations

import numpy as np

#: Above this many matrix cells the Python-level Sage construction dominates
#: and we decline the Sage path rather than waste time.
SAGE_MAX_CELLS = 2_000_000


def have_sage() -> bool:
    try:
        import sage.all  # noqa: F401
        return True
    except Exception:
        return False


def solve_gf2_sage(rows, n_unknowns):
    """Solve ``M x = b`` over GF(2) with Sage for packed augmented ``rows``.

    ``rows`` is a numpy uint64 array of shape (B, ceil(n_unknowns/64)+1) as
    produced by :func:`pyrandcracker.gf2.build_system`.  Returns a bool
    vector of length ``n_unknowns``.

    Raises ValueError if Sage is unavailable, the system is too large for
    the Sage construction, or the state is not uniquely determined.
    """
    if not have_sage():
        raise ValueError("sagemath is not installed")
    B, W = rows.shape
    if B * (n_unknowns + 1) > SAGE_MAX_CELLS:
        raise ValueError("system too large for the Sage path (%d cells > %d)"
                         % (B * (n_unknowns + 1), SAGE_MAX_CELLS))
    from sage.all import GF, Matrix

    rhs_word = W - 1
    flat = []
    for i in range(B):
        row = rows[i]
        for c in range(n_unknowns):
            flat.append((int(row[c >> 6]) >> (c & 63)) & 1)
        flat.append(int(row[rhs_word]) & 1)

    aug = Matrix(GF(2), B, n_unknowns + 1, flat)
    A = aug[:, :n_unknowns]
    b = aug[:, n_unknowns]
    if A.rank() < n_unknowns:
        raise ValueError(
            "observations do not determine the state uniquely "
            "(rank %d < %d); submit more bits" % (int(A.rank()), n_unknowns))
    x = A.solve_right(b)
    return np.array([int(v) for v in x], dtype=bool)
