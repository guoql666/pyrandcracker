# -*- coding: utf-8 -*-
"""Legacy dense GF(2) matrix utilities (corrected semantics)."""
import numpy as np
import pytest

from pyrandcracker import matrix_utils as mu


def test_gauss_elimination():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    rref, pivots, rows = mu.gf2_gauss_elimination(M)
    assert pivots == [0, 1, 2]
    # reduced row echelon: each pivot column has a single 1
    for c in pivots:
        assert int(rref[:, c].sum()) == 1


def test_solve_right_unique():
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    x_true = np.array([1, 0, 1], dtype=np.uint8)
    b = (A @ x_true) % 2
    x = mu.solve_right(A, b)
    assert np.array_equal(x, x_true)


def test_solve_right_underdetermined_raises():
    # rank 2 < 3 unknowns -> must raise, not return a silent particular
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    b = np.array([1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError):
        mu.solve_right(A, b)


def test_solve_right_inconsistent_raises():
    A = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)
    with pytest.raises(ValueError):
        mu.solve_right(A, b)


def test_solve_left():
    # x A = B  <=>  A^T x^T = B^T
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    x_true = np.array([1, 0, 1], dtype=np.uint8)
    B = (x_true @ A) % 2
    x = mu.solve_left(A, B)
    assert np.array_equal(x, x_true)


def test_gf2_solve_square():
    A = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    x_true = np.array([1, 0], dtype=np.uint8)
    b = (A @ x_true) % 2
    x = mu.gf2_solve(A, b)
    assert np.array_equal(x, x_true)
