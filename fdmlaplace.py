# This program numerically solves the Laplace equation with a source in a rectangular region where each boundary has either Dirichlet or Neumann conditions

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

# M = # nodes in x direction, N = # nodes in y direction
M = 25
N = 25

delta_x = 1/M
delta_y = 1/N

A = 1/(delta_x ** 2)
B = 1/(delta_y ** 2)
C = -2 * A - 2 * B

def source(x,y):
    return 0

# returns coordinates of a node given its 1d index k
def get_coords_from_index(k):
    i = k % M
    j = k // M
    return i * delta_x, j * delta_y

# boundary condition types
boundary_conditions = {'left': 'dirichlet', 'right': 'dirichlet', 'top': 'dirichlet', 'bottom': 'neumann'}
prescribed_values = {'left': 0.0, 'right': 0.0, 'top':1.0, 'bottom': 0.0}

def get_node_type(k):
    # get 2d indices of node (i,j) from 1d index k 
    i = k % M
    j = k // M

    if i == 0:
        return 'left'
    elif i == M - 1:
        return 'right'
    elif j == 0:
        return 'bottom'
    elif j == N - 1:
        return 'top'
    
    return 'interior'
    
# generate matrix of system ========================================
main_diagonal = np.zeros(M*N)
for k in range(0,M*N):
    node_type = get_node_type(k)
    if node_type == 'interior':
        main_diagonal[k] = C
    elif boundary_conditions[node_type] == 'dirichlet':
        main_diagonal[k] = 1
    elif node_type == 'left' or node_type == 'right':
        main_diagonal[k] = -1/delta_x
    elif node_type == 'top' or node_type == 'bottom':
        main_diagonal[k] = -1/delta_y

plus_one_diagonal = np.zeros(M*N-1)
for k in range(0, M*N-1):
    node_type = get_node_type(k)
    if node_type == 'interior':
        plus_one_diagonal[k] = A
    if node_type == 'left' and boundary_conditions[node_type] == 'neumann':
        plus_one_diagonal[k] = 1/delta_x

minus_one_diagonal = np.zeros(M*N-1)
for k in range(1, M*N):
    node_type = get_node_type(k)
    if node_type == 'interior':
        minus_one_diagonal[k-1] = A
    if node_type == 'right' and boundary_conditions[node_type] == 'neumann':
        minus_one_diagonal[k-1] = 1/delta_x

plus_M_diagonal = np.zeros(M*N-M)
for k in range(0, M*N-M):
    node_type = get_node_type(k)
    if node_type == 'interior':
        plus_M_diagonal[k] = B
    if node_type == 'bottom' and boundary_conditions[node_type] == 'neumann':
        plus_M_diagonal[k] = 1/delta_y

minus_M_diagonal = np.zeros(M*N-M)
for k in range(M, M*N):
    node_type = get_node_type(k)
    if node_type == 'interior':
        minus_M_diagonal[k-M] = B
    if node_type == 'top' and boundary_conditions[node_type] == 'neumann':
        minus_M_diagonal[k-M] = 1/delta_y

y = np.zeros(M*N)

for k in range(M*N):
    node_type = get_node_type(k)
    if node_type == 'interior':
        x_coord,y_coord = get_coords_from_index(k)
        y[k] = source(x_coord,y_coord)
    else:
        y[k] = prescribed_values[node_type]

diags = [main_diagonal, plus_one_diagonal, minus_one_diagonal, plus_M_diagonal, minus_M_diagonal]
diag_positions = [0, 1, -1, M, -M]
Mat = scipy.sparse.diags(diags, diag_positions, format='csc')

# solve system Ax = y
x = scipy.sparse.linalg.spsolve(Mat, y)

# render the solution
from matplotlib import colors
 
data = x.reshape(M,N)

x = np.arange(0, 1, delta_x)
y = np.arange(0, 1, delta_y)
X, Y = np.meshgrid(x, y)

cmap = plt.colormaps['jet']

fig, ax = plt.subplots()
pc = ax.pcolormesh(data, cmap=cmap)

plt.show()