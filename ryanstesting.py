import numpy as np
from matplotlib import pyplot

np.set_printoptions(precision=3)

L = 1 # length of space grid
N = 100 # number of points in space grid
dx = L / (N - 1)
x_grid = np.array([j*dx for j in range(N)])

T = 50 # length of time grid
Nt = 100 # number of points in time grid
dt = T / (Nt - 1)
t_grid = np.array([i*dt for i in range(Nt)])

a = 1 # diffusion parameter

u = np.zeros((N, Nt))

u[0,:] = 0
u[(N-1),:] = 0

u[int((N-1)/2),0] = 1