# This program simulates the heat equation in the rectangle [0,1]x[0,1]
# where the boundary conditions can be Dirichlet or Neumann.
# Heat equation: d(temp)/dt = d^2(temp)/dx^2 + d^2(temp)/dy^2 + source(x,y)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

M = 75 # number of nodes in x direction
N =  75 # number of nodes in y direction

# spacing between nodes
delta_x = 1/(M-1)
delta_y = 1/(N-1)

# time step, this has to be small enough otherwise instability will occur
delta_t = 0.001 
time = 0

# boundary conditions
def left_value(t):
    return 5 * np.sin(5 * t) + 5
def right_value(t):
    return 0
def top_value(t):
    return 0
def bottom_value(t):
    return 0




boundary_conditions = {'left': ('dirichlet', left_value), 'right': ('dirichlet',right_value), 'top': ('dirichlet', top_value), 'bottom': ('dirichlet',bottom_value)}

nodes = np.zeros((M,N))
new_nodes = np.zeros((M,N))

def init_temp(x_coord,y_coord):
    if x_coord > 0.35 and x_coord < 0.65 and y_coord > 0.35 and y_coord < 0.65:
        return 50
    return 0

def source(x_coord, y_coord):
    return 0

def get_coords_from_indices(i,j):
    return i * delta_x, j * delta_y

def get_node_type(i,j):
    if i == 0:
        return 'left'
    elif i == M - 1:
        return 'right'
    elif j == 0:
        return 'bottom'
    elif j == N - 1:
        return 'top'
    return 'interior'

# set initial conditions
for i in range(M):
    for j in range(N):
        x_coord, y_coord = get_coords_from_indices(i,j)
        nodes[i,j] = init_temp(x_coord, y_coord)

# update temps (forward time step)

x_ticks = np.arange(0, 1, delta_x)
y_ticks = np.arange(0, 1, delta_y)
grid_x, grid_y = np.meshgrid(x_ticks, y_ticks)

color_map = plt.colormaps['jet']
color_norm = colors.Normalize(vmin=0, vmax=10.0)

figure, axis = plt.subplots()

# render results of simulation
render_freq = 50 # number of updates/frames between renders
frame = 0 # frame counter
max_frames = 100000 # number of frames to be simulated
save_images = True
while frame < max_frames:
    if (frame % render_freq == 0):
        pc = axis.pcolormesh(nodes.transpose(),norm=color_norm, cmap=color_map) # need to take transpose of nodes matrices to render properly
        plt.pause(0.0001) # need this to make matplotlib actually render
        
        if (save_images):
            plt.savefig('frame_'+ str(frame)+'.png')

    frame += 1
    
    # update the temperature of nodes
    for i in range(M):
        for j in range(N):
            # first compute second derivatives of temp at node i,j at the current time step
            # d1 denotes d^2(temp)/dx^2
            # d2 denotes d^2(temp)/dy^2
            d1 = 0 
            d2 = 0
            node_type = get_node_type(i,j)
            if (node_type == 'interior'):
                d1 = (nodes[i+1,j] + nodes[i-1, j] - 2 * nodes[i,j]) / (delta_x**2)
                d2 = (nodes[i, j+1] + nodes[i, j-1] - 2 * nodes[i,j]) / (delta_y**2)
            elif (boundary_conditions[node_type][0] == "dirichlet"):
                new_nodes[i,j] = boundary_conditions[node_type][1](time)
                continue # since we've already found the new temp at this node
            elif (boundary_conditions[node_type][0] == "neumann"):
                prescribed_flux = boundary_conditions[node_type][1](time)
                # compute d1 first
                if (i == 0):
                    d1 = 2/(delta_x ** 2) * (nodes[i+1,j] - nodes[i,j] - prescribed_flux * delta_x)
                elif (i == M-1):
                    d1 = 2/(delta_x ** 2) * (nodes[i-1,j] - nodes[i,j] + prescribed_flux * delta_x)
                else:
                    d1 = (nodes[i+1,j] + nodes[i-1, j] - 2 * nodes[i,j]) / (delta_x**2)
                # now d2
                if (j == 0):
                    d2 = 2/(delta_y ** 2) * (nodes[i,j+1] - nodes[i,j] - prescribed_flux * delta_y)
                elif (j == N-1):
                    d2 = 2/(delta_y ** 2) * (nodes[i,j-1] - nodes[i,j] + prescribed_flux * delta_y)
                else:
                    d2 = (nodes[i, j+1] + nodes[i, j-1] - 2 * nodes[i,j])/ (delta_y**2)

            # now compute the temp at node i,j in new time step
            new_nodes[i,j] = nodes[i,j] + delta_t * 0.01 *(d1 + d2 + source(x_coord, y_coord))

    # update node temps
    nodes = new_nodes.copy()

    time += delta_t