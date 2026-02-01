# location for main code
import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D, axes3d  ##library for 3d projection plots

###variable declarations
location = "ANCHORAGE"
# location = "MIAMI"
nx = 60
ny = 24
nz = 6
length = 60
width = 24
height = 6
nu = 1.872 # thermal diffusivity (0.078 for hourly, 1.872 for daily)
dx = (length - 1) / (nx - 1)
dy = (width - 1) / (ny - 1)
dz = (height - 1) / (nz - 1)
sigma = 0.5
dt = sigma* ((dx**2) / (6*nu)) # ASSUMES dx = dy = dz

x = numpy.linspace(0, 60, nx + 1)
y = numpy.linspace(0, 24, ny + 1)
z = numpy.linspace(0, 6, nz + 1)

u = numpy.ones((nx+1, ny+1, nz+1))  # create a 1xn vector of 1's
un = numpy.ones((nx+1, ny+1, nz+1))

###Assign initial conditions
def createwalls(u):
    y1 = int(0.275 * length)
    y2 = int(0.725 * length) # 45% of south wall = window
    y3 = int(0.35 * length)
    y4 = int(0.65 * length) # 30% of new walls = window
    x1 = int(0.35 * width)
    x2 = int(0.65 * width)

    south_window = np.zeros_like(u, dtype=bool)
    south_brick = np.zeros_like(u, dtype=bool)
    north_window = np.zeros_like(u, dtype=bool)
    north_brick = np.zeros_like(u, dtype=bool)
    east_window = np.zeros_like(u, dtype=bool)
    east_brick = np.zeros_like(u, dtype=bool)
    west_window = np.zeros_like(u, dtype=bool)
    west_brick = np.zeros_like(u, dtype=bool)

    south_brick[0, :, :] = True
    north_brick[-1, :, :] = True
    east_brick[:, -1, :] = True
    west_brick[:, 0, :] = True

    south_window[0, y1:y2, :] = True
    north_window[-1, y3:y4, :] = True
    west_window[x1:x2, 0, :] = True
    east_window[x1:x2, -1, :] = True

    south_brick[south_window] = False
    west_brick[west_window] = False
    north_brick[north_window] = False
    east_brick[east_window] = False

    brick_mask = (north_brick | south_brick | west_brick | east_brick)
    window_mask = (north_window | south_window | west_window | east_window)
    return brick_mask, window_mask
brick_mask, window_mask = createwalls(u)

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[:, :, :] = 21

if (location == "MIAMI"):
    T = lambda t: 4.9188111757142 * np.sin( 0.017200792322545 * t - 1193.71990384336) +\
            0.582462690124397 * np.cos( 760.266156033683 * t + 2095.71540499492) + 24.5960040916041
else:
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
                         (un[1: -1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]))

        ambient = T(n)
        h_temp = 24
        c_temp = 20

        R_brick = 0.872792
        R_window = 0.340659

        u[brick_mask] = ambient + (u[brick_mask] - ambient) * (R_brick)
        u[window_mask] = ambient + (u[window_mask] - ambient) * (R_window)

        ac_x1 = int(0.25 * width)
        ac_x2 = int(0.75 * width)
        ac_y1 = int((1/6) * length)
        ac_y2 = int((2/6) * length)
        ac_y3 = int((4/6) * length)
        ac_y4 = int((5/6) * length)

        if ambient <= 21:
            u[:ac_x1, ac_y1:ac_y2, -1] = h_temp
            u[ac_x2:, ac_y1:ac_y2, -1] = h_temp
            u[:ac_x1, ac_y3:ac_y4, -1] = h_temp
            u[ac_x2:, ac_y3:ac_y4, -1] = h_temp

        elif ambient > 21:
            u[:ac_x1, ac_y1:ac_y2, 0] = c_temp
            u[ac_x2:, ac_y1:ac_y2, 0] = c_temp
            u[:ac_x1, ac_y3:ac_y4, 0] = c_temp
            u[ac_x2:, ac_y3:ac_y4, 0] = c_temp

        avg_room_temps.append(np.average(u[1:-2, 1:-2, 1:-2]))

        ax.cla()  # clear it each time + reset
        ax.scatter(
            X,
            Y,
            Z,
            c=u,
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
