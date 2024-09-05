# Goal: Simulate the heat equation in a rectangle where boundary conditions can be Dirichlet or Neumann
# u_t = ku_xx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt

M = 100
N = 100

delta_x = 1/M
delta_y = 1/N

nodes = np.zeros((M,N))
new_nodes = np.zeros((M,N))

def get_coords_from_indices(i,j):
    return i * delta_x, j * delta_y

def init_temp(x_coord,y_coord):
    return x_coord**2 + y_coord**2

for i in range(M):
    for j in range(N):
        x_coord, y_coord = get_coords_from_indices(i,j)
        nodes[i][j] = init_temp(x_coord, y_coord)

num_frames = 100

for frame in range(num_frames+1):
    # calculate second derivatives u_xx, u_yy
    for i in range(M):
        for j in range(N):
            

# render the solution
from matplotlib import colors
 
data = nodes

x = np.arange(0, 1, delta_x)
y = np.arange(0, 1, delta_y)
X, Y = np.meshgrid(x, y)

cmap = plt.colormaps['jet']

fig, ax = plt.subplots()
pc = ax.pcolormesh(data, cmap=cmap)

plt.show()