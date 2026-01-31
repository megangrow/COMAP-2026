import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(precision=3)

L = 1 # length of space grid
N = 10 # number of points in space grid
dx = L / (N - 1)
x_grid = np.array([j*dx for j in range(N)])

T = 10 # length of time grid
Nt = 1000 # number of points in time grid
dt = T / (Nt - 1)
t_grid = np.array([i*dt for i in range(Nt)])

a = 0.25 # diffusion parameter
s = (a * dt)/(dx**2)

print(s)

u = np.full(N, 0.25)

u[int((N-1)/2)] = 0.5

F = np.zeros(N)
F[0] = 0
F[N-1] = 0

A = np.diagflat([(-s/2) for i in range(N-1)], -1) +\
    np.diagflat([(1+s) for i in range(N)]) +\
    np.diagflat([(-s/2) for i in range(N-1)], 1)

B = np.diagflat([(s/2) for i in range(N-1)], -1) +\
    np.diagflat([(1-s) for i in range(N)]) +\
    np.diagflat([(s/2) for i in range(N-1)], 1)

u_record = []
u_record.append(u)

for n in range(1, Nt):
    u_new = np.linalg.solve(A, B @ u + F)
    u = u_new
    u_record.append(u)

fig = plt.figure()
ax = plt.axes(xlim=(0,1), ylim=(-1,1))

line, = ax.plot([], [])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x_grid, u_record[i])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=10, blit=True)

plt.show()
