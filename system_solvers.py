# this file contains common direct methods for solving linear systems of equations

import numpy as np

# subtracts rows of matrix (or column vector)
def row_swap(mat : np.array , r1: int, r2: int):
    r1_copy = mat[r1].copy()
    mat[r1] = mat[r2]
    mat[r2] = r1_copy

# gives solution vector x to the equation mat * x = vec
def solver_gaussian_elimination(mat : np.array, vec : np.array) -> np.array:
    A = mat.copy() # coefficient matrix
    Q = vec.copy() # right vector
    size = A[0].size # num rows/columns in mat

    ### forward elimination
    for pivot_row in range(0,size-1):
        # if the pivot starts with a zero, swap it with the first lower row starting with a nonzero entry
        if A[pivot_row, pivot_row] == 0:
            skip_pivot = True
            #print("Zero in pivot row")
            for i in range(pivot_row + 1, size):
                if A[i,pivot_row] != 0:
                    row_swap(A, pivot_row, i)
                    row_swap(Q, pivot_row, i)
                    skip_pivot = False
                    break
            if skip_pivot:
                continue
                             
        # subtract pivot row from lower rows
        for i in range(pivot_row + 1, size):
            coeff = A[i, pivot_row]/A[pivot_row, pivot_row] # constant that pivot row is multiplied by before subtraction from lower rows
            for j in range(pivot_row, size):
                A[i,j] -= coeff * A[pivot_row,j]
            Q[i] -= coeff *  Q[pivot_row]

    if (not np.any(A[size-1])):
        print("Coefficient matrix is singular. No solution.")
        return 
    
    ### backward substitution
    x = np.zeros(size) # solution vector
    for i in range(size-1, -1, -1):
        sum = Q[i]
        for j in range(i+1,size):
            sum -= A[i,j] * x[j]
        x[i] = sum/A[i,i]
    return x

# This is just Gaussian elimination but with significant simiplifications possible due to the tridiagonal structure of the matrix
def solver_tridiagonal(diag : np.array, below : np.array, above : np.array, vec : np.array) -> np.array:
    d = diag.copy()
    a = below.copy()
    c = above.copy()
    Q = vec.copy()
    size = d.size

    ### forward elimination
    for pivot_row in range(0, size-1):
        coeff = a[pivot_row] / d[pivot_row] 
        #a[pivot_row] = 0 <- this is what happens in the elimination, but we don't need to keep track of the updated a values since they're all zero after the forward elimination phase
        d[pivot_row + 1] -= coeff * c[pivot_row]
        Q[pivot_row + 1] -= coeff * Q[pivot_row]

    ### backward substitution
    x = np.zeros(size) # solution vector
    x[size - 1] = Q[size - 1] / d[size - 1]
    for i in range(size-2,-1,-1):
        x[i] = (Q[i] - c[i] * x[i+1]) / d[i]
    return x



    
