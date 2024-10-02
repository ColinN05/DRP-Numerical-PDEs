# Simulation of diffusion + advection. This code is mostly the same as heatequation.py
# where the boundary conditions can be Dirichlet or Neumann.
# Advection-Diffusion Equation: du/dt = d^2u/dx^2 + d^2u/dy^2 - au_x - bu_y + source(x,y), (a,b) = velocity of advection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

M = 50 # number of nodes in x direction
N =  50 # number of nodes in y direction
a = 10/100 # x velocity of advection
b = 0/100 # y velocity of avection
diffusion_constant = 0.001 #0.01

# spacing between nodes
delta_x = 1/(M-1)
delta_y = 1/(N-1)

# time step, this has to be small enough otherwise instability will occur
delta_t = 0.001 
time = 0

# boundary conditions
def left_value(t):
    return 0
def right_value(t):
    return 0
def top_value(t):
    return 0
def bottom_value(t):
    return 0

boundary_conditions = {'left': ('neumann', left_value), 'right': ('neumann',right_value), 'top': ('neumann', top_value), 'bottom': ('neumann',bottom_value)}

nodes = np.zeros((M,N))
new_nodes = np.zeros((M,N))

def init_u(x_coord,y_coord):
    if x_coord > 0.40 and x_coord < 0.60 and y_coord > 0.40 and y_coord < 0.60:
        return 25
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
        nodes[i,j] = init_u(x_coord, y_coord)

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
save_images = False
while frame < max_frames:
    if (frame % render_freq == 0):
        pc = axis.pcolormesh(nodes.transpose(),norm=color_norm, cmap=color_map) # need to take transpose of nodes matrices to render properly
        plt.pause(0.0001) # need this to make matplotlib actually render
        
        if (save_images):
            plt.savefig('frame_'+ str(frame)+'.png')

    frame += 1
    
    # update the value of u at nodes
    for i in range(M):
        for j in range(N):
            # first compute derivatives of u at node i,j at the current time step
            d1 = 0 # d1 denotes d^2u/dx^2
            d2 = 0 # d2 denotes d^2u/dy^2
            d3 = 0 # du/dx
            d4 = 0 # du/dy
            node_type = get_node_type(i,j)
            if (node_type == 'interior'):
                d1 = (nodes[i+1,j] + nodes[i-1, j] - 2 * nodes[i,j]) / (delta_x**2)
                d2 = (nodes[i, j+1] + nodes[i, j-1] - 2 * nodes[i,j]) / (delta_y**2)
                d3 = (nodes[i+1,j] - nodes[i,j])/delta_x
                d4 = (nodes[i,j+1]-nodes[i,j])/delta_y
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

                # now d3/d4
                
                if (i == 0 or i == M-1):
                    d3 = prescribed_flux
                else:
                    d3 = (nodes[i + 1, j] - nodes[i,j]) / delta_x

                if (j == 0 or j == N-1):
                    d4 = prescribed_flux
                else:
                    d4 = (nodes[i,j+1]-nodes[i,j])/delta_y
            # now compute the temp at node i,j in new time step
            new_nodes[i,j] = nodes[i,j] + delta_t * (diffusion_constant * (d1 + d2) - a * d3 - b * d4 + source(x_coord, y_coord))

    # update node temps
    nodes = new_nodes.copy()

    time += delta_t