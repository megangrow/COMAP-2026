import numpy
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D, axes3d  ##library for 3d projection plots

###variable declarations
nx = 16
ny = 16
nz = 16
nt = 25
nu = .05 # a
v = 2
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dz = 2 / (nz - 1)
sigma = .25
dt = sigma * dx * dy * dz / nu

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
z = numpy.linspace(0, 2, nz)

u = numpy.ones((nx, ny, nz))  # create a 1xn vector of 1's
un = numpy.ones((nx, ny, nz))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dx):int(1 / dx + 1), int(.5 / dy):int(1 / dy + 1), int(.5 / dz):int(1 / dz + 1)] = 4

###Run through nt timesteps
def diffuse(nt):
    u[int(.5 / dx):int(1 / dx + 1), int(.5 / dy):int(1 / dy + 1), int(.5 / dz):int(1 / dz + 1)] = 4

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = numpy.meshgrid(x, y, z)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    for n in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] +
                         nu * dt / dx ** 2 *
                         (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1]) +
                         nu * dt / dy ** 2 *
                         (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1]) +
                         nu * dt / dz ** 2 *
                         (un[1: -1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]) -
                         v * dt / dx * (un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1]) -
                         v * dt / dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) -
                         v * dt / dz * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]))
        u[0, :, :] = 1
        u[-1, :, :] = 1
        u[:, 0, :] = 1
        u[:, -1, :] = 1
        u[:, :, 0] = 1
        u[:, :, -1] = 1


        ax.cla()  # clear it each time + reset
        ax.scatter(X, Y, Z, c=u, alpha=0.12)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        pyplot.pause(0.1)

    pyplot.show()

diffuse(100)
