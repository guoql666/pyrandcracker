import numpy as np


def gf2_gauss_elimination(matrix):
    """Performs Gaussian elimination on a matrix over GF(2) and returns the modified matrix, pivot columns, and original pivot rows."""
    mat = np.array(matrix, dtype=int).copy()
    m, n = mat.shape
    original_rows = np.arange(m)
    pivot_cols = []
    pivot_rows = []

    current_row = 0
    for current_col in range(n):
        if current_row >= m:
            break
        # Find the first 1 in the current column from current_row downwards
        pivot = np.where(mat[current_row:, current_col] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot_rel = pivot[0]
        pivot_abs = current_row + pivot_rel
        # Swap rows
        if pivot_abs != current_row:
            mat[[current_row, pivot_abs]] = mat[[pivot_abs, current_row]]
            original_rows[[current_row, pivot_abs]] = original_rows[[pivot_abs, current_row]]
        # Record pivot column and original row
        pivot_cols.append(current_col)
        pivot_rows.append(original_rows[current_row])
        # Eliminate entries below and above
        for row in range(m):
            if row != current_row and mat[row, current_col] == 1:
                mat[row] ^= mat[current_row]
        current_row += 1
    return mat, pivot_cols, pivot_rows

def gf2_solve(A, B):
    """Solves the equation Ax = B over GF(2) where A is a square matrix."""
    A = np.array(A, dtype=int)
    B = np.array(B, dtype=int)
    r = A.shape[0]
    augmented = np.hstack((A, B.reshape(-1, 1)))
    # Forward elimination
    for i in range(r):
        # Find pivot in column i
        pivot = np.where(augmented[i:, i] == 1)[0]
        if len(pivot) == 0:
            raise ValueError("No solution")
        pivot_row = i + pivot[0]
        if pivot_row != i:
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
        # Eliminate other rows
        for row in range(r):
            if row != i and augmented[row, i] == 1:
                augmented[row] ^= augmented[i]
    # Extract solution
    x = augmented[:, -1]
    return x


def solve_left(T, B, check=True):
    '''
    solve XT=B
    input:
    T: matrix
    B: vector
    check: if True, check if the solution is valid
    output:
    x: vector
    '''


    '''
    (XT)^T = T^T * X^T = B^T
    cause numpy vector not same as sagemath
    so X need be transposed
    '''
    return solve_right(T.T, B, check=check).T



def solve_right(T, B, check=True):
    """Solves T @ x = B over GF(2) where T is a matrix and B is a vector."""
    T = np.array(T, dtype=int)
    B = np.array(B, dtype=int)
    b, a = T.shape
    if B.shape != (b,):
        raise ValueError("B must be a vector of length equal to the rows of T")
    
    # Step 1: Gaussian elimination on T
    _, pivot_cols, pivot_rows_T = gf2_gauss_elimination(T)
    r = len(pivot_cols)
    if r == 0:
        if np.any(B != 0):
            raise ValueError("No solution")
        return np.zeros(a, dtype=int)
    
    # Step 2: Construct matrix from pivot columns
    A = T[:, pivot_cols]
    
    # Step 3: Gaussian elimination on A to find pivot rows
    _, _, A_pivot_rows = gf2_gauss_elimination(A)
    if len(A_pivot_rows) != r:
        raise ValueError("Unexpected error in matrix rank")
    
    # Step 4: Construct subsystem
    A_prime = A[A_pivot_rows, :]
    B_prime = B[A_pivot_rows]
    
    # Step 5: Solve subsystem
    try:
        x_part = gf2_solve(A_prime, B_prime)
    except ValueError:
        if check:
            raise ValueError("No solution")
        x_part = np.zeros(r, dtype=int)  # Fallback, may be invalid
    
    # Step 6: Construct full solution vector
    x = np.zeros(a, dtype=int)
    x[pivot_cols] = x_part
    
    # Step 7: Verify solution
    if check:
        if not np.array_equal((T @ x) % 2, B):
            raise ValueError("Matrix equation has no solutions")
    return x