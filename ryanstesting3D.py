import numpy
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D, axes3d  ##library for 3d projection plots

###variable declarations
nx = 10
ny = 10
nz = 10
length = 60
width = 24
height = 6
nu = 0.078 # thermal diffusivity
b = 0.00343 # thermal expansion
dx = length / (nx - 1)
dy = width / (ny - 1)
dz = height / (nz - 1)
sigma = .01
dt = sigma * dx * dy * dz / nu

x = numpy.linspace(0, 60, nx)
y = numpy.linspace(0, 24, ny)
z = numpy.linspace(0, 6, nz)

u = numpy.ones((nx, ny, nz))  # create a 1xn vector of 1's
un = numpy.ones((nx, ny, nz))

###Assign initial conditions
# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[:, :, :] = 21

###Run through nt timesteps
def diffuse(nt):
    u[:, :, :] = 21

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = numpy.meshgrid(x, y, z)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 24)
    ax.set_zlim(0, 6)
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
                         b * (dt / dx * ((un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1]) +
                                         (un[1:-1, 1:-1, 1:-1] - un[2:, 1:-1, 1:-1])) +
                         dt / dy * ((un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) +
                                    (un[1:-1, 1:-1, 1:-1] - un[1:-1, 2:, 1:-1])) +
                         dt / dz * ((un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]) +
                                    (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 2:]))))
        u[0, :, :] = 26
        u[-1, :, :] = 23
        u[:, 0, :] = 23
        u[:, -1, :] = 23
        u[:, :, 0] = 23
        u[:, :, -1] = 23


        ax.cla()  # clear it each time + reset
        mask = u > 0
        ax.scatter(
            X[mask],
            Y[mask],
            Z[mask],
            c=u[mask],
            cmap='plasma',
            alpha=0.6,
            vmin=22,
            vmax=26,
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 24)
        ax.set_zlim(0, 6)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        pyplot.pause(0.5)

    pyplot.show()

diffuse(72)
