# The general solver is not complete . . .

# Goal: The general solver should be able to solve any problem of the following form:
### Equation: Au_xx + Bu_xy + Cu_yy + Du_x + Eu_y = F where A,B,C,D,E,F are functions of x,y
### Domain: a general axis-aligned rectangle [a,b]x[c,d]
### Boundary conditions: general Robin condition au + b du/dn = c, a,b,c are functions

# The current plan is to use the finite difference method with an ADI iterative solver

import numpy as np
from system_solvers import *

class general_solver_params:
    def printMeshResolution(self):
        print(self.nodes_x, self.nodes_y)

def general_solver(params: general_solver_params) -> np.array:
    nodes = np.array((params.nodes_x, params.nodes_y))
    ### calculations . . .
    return nodes

############ Example for testing 
a = 0.0
b = 1.0
c = 0.0
d = 1.0

# equation coefficients
def A(x: np.double,y: np.double) -> np.double:
    return 1.0
def B(x: np.double,y: np.double) -> np.double:
    return 0.0
def C(x: np.double,y: np.double) -> np.double:
    return 1.0
def D(x: np.double,y: np.double) -> np.double:
    return 0.0
def E(x: np.double,y: np.double) -> np.double:
    return 0.0
def F(x: np.double,y: np.double) -> np.double:
    return 0.0

M = 30
N = 30
delta_x = (b-a)/(M-1)
delta_y = (d-c)/(N-1)

nodes = np.zeros((M,N))
nodes_old = np.zeros((M,N))

# work in progress . . .



