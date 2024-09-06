# This program uses the finite difference method to numerically 
# solve the Laplace equation with a source in the rectangle [0,1]x[0,1]
# where each boundary has either Dirichlet or Neumann conditions.

#
# d^2u/dx^2 + d^2u/dy^2 = source(x,y)
#

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

# M = # nodes in x direction, N = # nodes in y direction
M = 50
N = 50

delta_x = 1/(M-1)
delta_y = 1/(N-1)

def source(x,y):
    return -10.0 * x

# returns coordinates of a node given its 1d index k
def get_coords_from_index(k):
    i = k % M
    j = k // M
    return i * delta_x, j * delta_y

def get_2d_indices(k):
    i = k % M
    j = k // M
    return i,j

# boundary conditions
boundary_conditions = {'left': 'neumann', 'right': 'neumann', 'top': 'neumann', 'bottom': 'dirichlet'}
prescribed_values = {'left': 0.0, 'right': 1.0, 'top':1.0, 'bottom': 1.0}

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
    
# generate the system ========================================
mat_LIL = scipy.sparse.lil_matrix((M*N, M*N)) # the matrix is initially in CSR format for easy entry modification
right_column_vector = np.zeros(M*N)

# these values occur frequently in the system
A = 1/(delta_x ** 2)
B = 1/(delta_y ** 2)
C = -2 * A - 2 * B

for k in range(M*N):
    node_type = get_node_type(k)
    x_coord, y_coord = get_coords_from_index(k)
    if node_type == 'interior':
        mat_LIL[k,k + 1] = A
        mat_LIL[k,k - 1] = A
        mat_LIL[k,k + M] = B
        mat_LIL[k,k - M] = B
        mat_LIL[k,k] = C
    elif boundary_conditions[node_type] == 'dirichlet':
        mat_LIL[k, k] = 1
    elif boundary_conditions[node_type] == 'neumann':
        if node_type == 'left':
            mat_LIL[k,k+1] = 1/delta_x
            mat_LIL[k,k] = -1/delta_x
        elif node_type == 'right':
            mat_LIL[k,k-1] = 1/delta_x
            mat_LIL[k,k] = -1/delta_x
        elif node_type == 'top':
            mat_LIL[k,k-M] = 1/delta_y
            mat_LIL[k,k] = -1/delta_y
        elif node_type == 'bottom':
            mat_LIL[k,k+M] = 1/delta_y
            mat_LIL[k,k] = -1/delta_y
    if node_type == 'interior':
        right_column_vector[k] = source(x_coord, y_coord)
    else:
        right_column_vector[k] = prescribed_values[node_type]

# convert matrix to CSR format for faster calculations
mat_CSR = mat_LIL.tocsr() 

# solve system to get temperature at each node
node_temps = scipy.sparse.linalg.spsolve(mat_CSR, right_column_vector)

# render the solution =====================
from matplotlib import colors
 
data = node_temps.reshape(M,N)

x = np.arange(0, 1, delta_x)
y = np.arange(0, 1, delta_y)
X, Y = np.meshgrid(x, y)

cmap = plt.colormaps['jet']

fig, ax = plt.subplots()
pc = ax.pcolormesh(data, cmap=cmap)

plt.show()