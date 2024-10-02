# solution to the wave equation in [0,1]x[0,1] using Euler's method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# wave velocity
c = 0.5

M = 30 # nodes in the x direction
N = 30 # nodes in the y direction

# node spacing
delta_x = 1.0/(M-1.0) 
delta_y = 1.0/(N-1.0)
delta_t = 0.02

# coefficients that appear in the udpate formula
alpha = 1.0/(delta_t**2)
beta = c**2/(delta_x**2)
gamma = c**2/(delta_y**2)

# values of nodes
nodes = np.zeros((M,N)) # at current time t
nodes_old1 = np.zeros((M,N)) # at t-delta_t
nodes_old2 = np.zeros((M,N)) # at t- 2 * delta_t

#initial conditions
def init_value(x,y):
    return 10.0 / (1.0 + (1.0 + (20.0*(x-0.5))**2 + (20.0*(y-0.5))**2)) # gaussian
def init_velocity(x,y):
    return 1.0
def coords(i,j,k):
    return i * delta_x, j * delta_y, k * delta_t

# set initial conditions
for i in range(M):
    for j in range(N):
        x,y,t=coords(i,j,0)
        nodes_old2[i,j] = init_value(x,y)
        nodes_old1[i,j] = init_value(x,y) + init_velocity(x,y) * delta_t

max_frames = 150 # number of frames to rendered
frame = 0
render_freq = 1
save_images = False

boundary_conditions = { "left": ("dirichlet", 0.0), "right": ("dirichlet", 0.0), "top": ("dirichlet", 0.0), "bottom": ("dirichlet", 0.0) }

def get_node_type(i,j):
    if i == 0:
        return "left"
    elif i == M-1:
        return "right"
    elif j == 0:
        return "bottom"
    elif j == N-1:
        return "top"
    return "interior"

def interior_node_value(i,j): # calculates value of interior node using values at previous time steps
    return 2.0 * nodes_old1[i,j] - nodes_old2[i,j] + beta/alpha * nodes_old1[i+1,j] - 2 * beta/alpha * nodes_old1[i,j] + beta/alpha * nodes_old1[i-1,j] + gamma/alpha * nodes_old1[i,j+1] - 2 * gamma/alpha * nodes_old1[i,j] + gamma/alpha * nodes_old1[i,j-1]

# setup plot
x_ticks = np.arange(0, 1, delta_x)
y_ticks = np.arange(0, 1, delta_y)
grid_x, grid_y = np.meshgrid(x_ticks, y_ticks)

color_map = plt.colormaps['jet']
color_norm = colors.Normalize(vmin=0.0, vmax=1.0)

figure, axis = plt.subplots()

# simulation main loop
while frame < max_frames:
    # calculate new node values
    for i in range(M):
        for j in range(N):
            node_type = get_node_type(i,j)
            if node_type == "interior":
                nodes[i,j] = interior_node_value(i,j)
            else:
                condition_type = boundary_conditions[node_type][0]
                condition_value = boundary_conditions[node_type][1]
                if condition_type == "dirichlet":
                    nodes[i,j] = condition_value
                elif condition_type == "neumann":
                    if node_type == "left":
                        #nodes[i,j] = interior_node_value(i+1,j) + condition_value * (-delta_x)
                        nodes[i,j] = nodes_old1[i+1,j] + condition_value * (-delta_x)
                    elif node_type == "right":
                        #nodes[i,j] = interior_node_value(i-1,j) + condition_value * delta_x
                        nodes[i,j] = nodes_old1[i-1,j] + condition_value * (delta_x)
                    elif node_type == "top":
                        #nodes[i,j] = interior_node_value(i,j-1) + condition_value * delta_y
                        nodes[i,j] = nodes_old1[i,j-1] + condition_value * (delta_y)
                    elif node_type == "bottom":
                        #nodes[i,j] = interior_node_value(i,j+1) + condition_value * (-delta_y)
                        nodes[i,j] = nodes_old1[i,j+1] + condition_value * (-delta_y)
    nodes_old2 = nodes_old1.copy()
    nodes_old1 = nodes.copy()

    # render and save image
    if (frame % render_freq == 0):
        pc = axis.pcolormesh(nodes.transpose(),norm=color_norm, cmap=color_map)
        plt.pause(0.0001)
        
        if (save_images):
            plt.savefig('frame_'+ str(frame)+'.png')

    frame += 1



