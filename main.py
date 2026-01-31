# location for main code
import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D, axes3d  ##library for 3d projection plots

###variable declarations
nx = 10
ny = 10
nz = 10
length = 60
width = 24
height = 6
nu = 1.872 # thermal diffusivity (0.078 for hourly, 1.872 for daily)
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

T = lambda t: 0.69585135778619 * np.sin(0.00137029999867194 * t - 2432.77084285537) +\
            12.2251580183288 * np.cos(0.017201860290628 * t + 36557989.8870071) + 3.13224850133946

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

    avg_room_temps = []

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

        ambient = T(n)
        h_temp = 24
        c_temp = 20

        R = 1.5

        u[0, :, :] = ambient + (u[1, :, :] - ambient)*(R - 1)
        u[-1, :, :] = ambient + (u[-2, :, :] - ambient)*(R - 1)
        u[:, 0, :] = ambient + (u[:, 1, :] - ambient)*(R - 1)
        u[:, -1, :] = ambient + (u[:, -2, :] - ambient)*(R - 1)
        u[:, :, 0] = ambient + (u[:, :, 1] - ambient)*(R - 1)
        u[:, :, -1] = ambient + (u[:, :, -2] - ambient)*(R - 1)

        bmask_x1 = (X >= 0) & (X <= 30)
        bmask_x2 = (X >= 30) & (X <= 60)
        bmask_y1 = (Y >= 0) & (Y <= 6)
        bmask_y2 = (Y >= 18) & (Y <= 24)
        bmask_z1 = (Z == 0)
        bmask_z2 = (Z == 6)

        if ambient <= 3.13:
            u[bmask_x1 & bmask_y1 & bmask_z1] = h_temp
            u[bmask_x2 & bmask_y1 & bmask_z1] = h_temp
            u[bmask_x1 & bmask_y2 & bmask_z1] = h_temp
            u[bmask_x2 & bmask_y2 & bmask_z1] = h_temp

        elif ambient > 3.13:
            u[bmask_x1 & bmask_y1 & bmask_z2] = c_temp
            u[bmask_x2 & bmask_y1 & bmask_z2] = c_temp
            u[bmask_x1 & bmask_y2 & bmask_z2] = c_temp
            u[bmask_x2 & bmask_y2 & bmask_z2] = c_temp

        avg_room_temps.append(np.average(u[1:-2, 1:-2, 1:-2]))

        ax.cla()  # clear it each time + reset
        mask = u > -30
        ax.scatter(
            X[mask],
            Y[mask],
            Z[mask],
            c=u[mask],
            cmap='plasma',
            alpha=0.6,
            vmin=-30,
            vmax=30
        )
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 24)
        ax.set_zlim(0, 6)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        pyplot.pause(0.1)
        print(f"DAY: {n}")

    pyplot.show()

    return(avg_room_temps)

results = diffuse(365)

pyplot.plot(np.linspace(0, 366, 366), results)
pyplot.show()
