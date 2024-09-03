import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

M=100
N=100

delta_x = 1.0/(N-1.0)
delta_y = 1.0/(M-1.0)

A = 1.0/(delta_x**2)
B = 1.0/(delta_y**2)
C = -2.0*A - 2.0*B

def is_boundary_index(k):
    return (k % N == 0 or k % N == N-1 or k>=(M-1)*N or k<N)

main_diagonal = np.ones(M*N)
for k in range(M*N):
    if is_boundary_index(k):
        main_diagonal[k] = 1
    else:
        main_diagonal[k] = C

a_diagonal_left = np.zeros(M*N-1)
for k in range(1,M*N):
    l = k-1 #index of entry in diagonal
    if not is_boundary_index(k):
        a_diagonal_left[l] = A

a_diagonal_right = np.zeros(M*N-1)
for k in range(0,M*N-1):
    l = k #index of entry in diagonal
    if not is_boundary_index(k):
        a_diagonal_right[l] = A

b_diagonal_left = np.zeros(M*N-N)
for k in range(N,M*N):
    l = k-N
    if not is_boundary_index(k):
        b_diagonal_left[l] = B

b_diagonal_right = np.zeros(M*N-N)
for k in range(0,M*N-N):
    l = k #index of entry in diagonal
    if not is_boundary_index(k):
        b_diagonal_right[l] = B


diags = [main_diagonal, a_diagonal_left, a_diagonal_right, b_diagonal_left, b_diagonal_right]
diag_positions = [0, -1, 1, -N, N]
Mat = scipy.sparse.diags(diags, diag_positions, format='csc')

y = np.ones(M*N)

#k % N == 0 or k % N == N-1 or k>=(M-1)*N or k<N

for k in range(M*N):
    if is_boundary_index(k):
        y[k] = 0.0
    else:
        y[k] = -1

x = scipy.sparse.linalg.spsolve(Mat, y)

print(x)

from matplotlib import colors
 
data = x.reshape(M,N)

x = np.arange(0, 1, delta_x)
y = np.arange(0, 1, delta_y)
X, Y = np.meshgrid(x, y)

cmap = plt.colormaps['jet']

fig, ax = plt.subplots()
pc = ax.pcolormesh(data, cmap=cmap)

plt.show()
    


