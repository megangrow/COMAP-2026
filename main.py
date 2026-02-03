# location for main code
import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D, axes3d  ##library for 3d projection plots
import pvlib
from pygments.lexers import ambient

###variable declarations
#location = "ANCHORAGE"
location = "MIAMI"
nx = 60
ny = 24
nz = 6
length = 60
width = 24
height = 6
nu = 1.872 # thermal diffusivity (0.078 for hourly, 1.872 for daily)
k_air = 100 # heat transfer coefficient of air
dx = (length - 1) / (nx - 1)
dy = (width - 1) / (ny - 1)
dz = (height - 1) / (nz - 1)
sigma = 0.05
dt = sigma * ((dx**2) / (6*nu)) # ASSUMES dx = dy = dz

x = numpy.linspace(0, nx, nx + 1)
y = numpy.linspace(0, ny, ny + 1)
z = numpy.linspace(0, nz, nz + 1)

u = numpy.ones((nx+1, ny+1, nz+1))  # create a 1xn vector of 1's
un = numpy.ones((nx+1, ny+1, nz+1))

print(dt)

###Assign initial conditions
def createwalls(u):
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

    s = int(0.45 * nx)
    n = int(0.3 * nx)
    o = int(0.3 * ny)

    x1 = (nx - s) // 2
    x2 = nx - x1
    x3 = (nx - n) // 2
    x4 = nx - x3
    y1 = (ny - o) // 2
    y2 = ny - y1

    south_window[x1:x2, 0, :] = True
    north_window[x3:x4, -1, :] = True
    west_window[0, y1:y2, :] = True
    east_window[-1, y1:y2, :] = True

    south_brick[south_window] = False
    west_brick[west_window] = False
    north_brick[north_window] = False
    east_brick[east_window] = False

    brick_mask = (north_brick | south_brick | west_brick | east_brick)
    window_mask = (north_window | south_window | west_window | east_window)
    return brick_mask, window_mask
brick_mask, window_mask = createwalls(u)

u[:, :, :] = 21

if (location == "MIAMI"):
    T = lambda t: 4.9188111757142 * np.sin( 0.017200792322545 * t - 1193.71990384336) +\
            0.582462690124397 * np.cos( 760.266156033683 * t + 2095.71540499492) + 24.5960040916041

    ###Compute Angle of the SUN
    def sun_angle(d):
        return ((np.pi / 2) - ((25.774 * np.pi) / 180) + pvlib.solarposition.declination_cooper69(d)) * (180 / np.pi)

    ths = sun_angle(172)
    thw = sun_angle(354)

elif (location == "ANCHORAGE"):
    T = lambda t: 0.69585135778619 * np.sin(0.00137029999867194 * t - 2432.77084285537) +\
            12.2251580183288 * np.cos(0.017201860290628 * t + 36557989.8870071) + 3.13224850133946

    ###Compute Angle of the SUN
    def sun_angle(d):
        return ((np.pi / 2) - ((61.21806 * np.pi) / 180) + pvlib.solarposition.declination_cooper69(d)) * (180 / np.pi)

    ths = sun_angle(172)
    thw = sun_angle(354)

else:
    raise RuntimeError("\n Please check location and variables!!! \n")

###Run through nt timesteps
def diffuse(nt):
    u[:, :, :] = 21

    fig = pyplot.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = numpy.meshgrid(x, y, z, indexing='ij')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 24)
    ax.set_zlim(0, 6)
    ax.set_box_aspect([10, 4, 1])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    avg_room_temps = []

    t = np.arange(0, nt, dt)

    for n in t:
        un = u.copy()
        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] + dt * (
                         nu / dx ** 2 *
                         (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1]) +
                         nu / dy ** 2 *
                         (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1]) +
                         nu / dz ** 2 *
                         (un[1: -1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2])))
        ambient = T(n)
        h_temp = 24
        c_temp = 20

        R_brick = 2.45
        R_window = 0.366

        u[brick_mask] = (ambient+10) + (u[brick_mask] - (ambient+10)) * (R_brick)

        R_as = 0.36
        R_v = 0.003
        if (location == "ANCHORAGE"):
            thd = sun_angle(n + 59.25)
        elif (location == "MIAMI"):
            thd = sun_angle(n + 48)
        else:
            raise RuntimeError("Not valid location")

        p11 = (ambient+5-u[1, :, :])/R_brick
        p12 = (ambient+5-u[-2, :, :])/R_brick
        p21 = (ambient+0-u[:, 1, :])/R_brick
        p22 = (ambient+10-u[:, -2, :])/R_brick
        p31 = (ambient+0-u[:, :, 1])/(R_brick+1)
        p32 = (ambient+10-u[:, :, -2])/R_brick

        u[0, :, :] = u[1, :, :] + p11 * dz
        u[-1, :, :] = u[-2, :, :] + p12 * dz
        u[:, 0, :] = u[:, 1, :] + p21 * dz
        u[:, -1, :] = u[:, -2, :] + p22 * dz
        u[:, :, -1] = u[:, :, 1] + p31 * dz
        u[:, :, -1] = u[:, :, -2] + p32 * dz

        on_or_off = True
        if on_or_off:
            shading = (ths - thd) / (ths - thw)
        else:
            shading = 1

        r11 = ((ambient + 5*shading) - u[1, 8:16, :])/R_window
        r12 = ((ambient + 5*shading) - u[-2, 8:16, :]) / R_window
        r21 = ((ambient + 0*shading) - u[21:39, 1, :]) / R_window
        r22 = ((ambient + 10*shading) - u[16:44, -2, :]) / R_window

        u[0, 8:16, :] = u[1, 8:16, :] + r11 * dz
        u[-1, 8:16, :] = u[-2, 8:16, :] + r12 * dz
        u[21:39, 0, :] = u[21:39, 1, :] + r21 * dz
        u[16:44, -1, :] = u[16:44, -2, :] + r22 * dz

        # CONCRETE MASS
        # u[1:3, 8:16, 1:3] = 21
        # u[-4:-2, 8:16, 1:3] = 21
        # u[21:39, 1:3, 1:3] = 21
        # u[16:44, -4:-2, 1:3] = 21

        # Tg = (ambient + 10 * ((thd - thw / ths - thw))) + (
        #             (u[window_mask] - (ambient + 10 * (thd - thw / ths - thw))) * (R_window))
        #
        # u[window_mask] = Tg + (u[window_mask] - Tg) * (R_window)

        if (ambient < 20):
            vent_temp = 30
        else:
            vent_temp = 15

        vent_temp = 21
        u[10:12, 4:20, -1] = u[10:12, 4:20, -2] + k_air * (vent_temp - u[10:12, 4:20, -2]) * dz
        u[20:22, 4:20, -1] = u[20:22, 4:20, -2] + k_air * (vent_temp - u[20:22, 4:20, -2]) * dz
        u[30:32, 4:20, -1] = u[30:32, 4:20, -2] + k_air * (vent_temp - u[30:32, 4:20, -2]) * dz
        u[40:42, 4:20, -1] = u[40:42, 4:20, -2] + k_air * (vent_temp - u[40:42, 4:20, -2]) * dz
        u[50:52, 4:20, -1] = u[50:52, 4:20, -2] + k_air * (vent_temp - u[50:52, 4:20, -2]) * dz
        # u[10:12, 4:20, -2] = 21
        # u[20:22, 4:20, -2] = 21
        # u[30:32, 4:20, -2] = 21
        # u[40:42, 4:20, -2] = 21
        # u[50:52, 4:20, -2] = 21


        # u[10:12, 4:20, -3] = vent_temp #u[10:12, 4:20, -4] + k_air * (vent_temp - u[10:12, 4:20, -4]) * dz
        # u[20:22, 4:20, -3] = vent_temp #u[20:22, 4:20, -4] + k_air * (vent_temp - u[20:22, 4:20, -4]) * dz
        # u[30:32, 4:20, -3] = vent_temp #u[30:32, 4:20, -4] + k_air * (vent_temp - u[30:32, 4:20, -4]) * dz
        # u[40:42, 4:20, -3] = vent_temp #u[40:42, 4:20, -4] + k_air * (vent_temp - u[40:42, 4:20, -4]) * dz
        # u[50:52, 4:20, -3] = vent_temp #u[50:52, 4:20, -4] + k_air * (vent_temp - u[50:52, 4:20, -4]) * dz
        #
        # u[10:12, 4:20, -6] = vent_temp  # u[10:12, 4:20, -2] + k_air * (vent_temp - u[10:12, 4:20, -2]) * dz
        # u[20:22, 4:20, -6] = vent_temp  # u[20:22, 4:20, -2] + k_air * (vent_temp - u[20:22, 4:20, -2]) * dz
        # u[30:32, 4:20, -6] = vent_temp  # u[30:32, 4:20, -2] + k_air * (vent_temp - u[30:32, 4:20, -2]) * dz
        # u[40:42, 4:20, -6] = vent_temp  # u[40:42, 4:20, -2] + k_air * (vent_temp - u[40:42, 4:20, -2]) * dz
        # u[50:52, 4:20, -6] = vent_temp

        avg_room_temps.append(np.average(u[1:-2, 1:-2, 1:-2]))

        if location == "MIAMI":
            if int(n) == 90 or int(n) == -2 or int(n) == -1 :
                ax.cla()  # clear it each time + reset
                sc = ax.scatter(
                    X,
                    Y,
                    Z,
                    c=u,
                    cmap='plasma',
                    alpha=0.1,
                    vmin=-30,
                    vmax=30
                )
                ax.set_xlim(0, 60)
                ax.set_ylim(0, 24)
                ax.set_zlim(0, 6)
                ax.set_zticks([3, 6])
                ax.set_xlabel('$x$', labelpad=50, fontsize=35)
                ax.set_ylabel('$y$', labelpad=15, fontsize=35)
                ax.set_zlabel('$z$', labelpad=10, fontsize=35)
                ax.tick_params(axis='both', which='major', labelsize=17)
                ax.set_box_aspect([10, 4, 1])
                ax.view_init(elev=30, azim=120)
                # ax.set_title(f"Temperature Heat Map for Day {int(n)} in {location}")
                fig.text(0.5, 0.80, f"Temperature Heat Map in {location}", ha='center', va='top', fontsize=20)
                pyplot.pause(0.1)

        else:
            if int(n) == 255 or int(n) == -3 or int(n) == -1 :
                ax.cla()  # clear it each time + reset
                sc = ax.scatter(
                    X,
                    Y,
                    Z,
                    c=u,
                    cmap='plasma',
                    alpha=0.1,
                    vmin=-30,
                    vmax=30
                )
                ax.set_xlim(0, 60)
                ax.set_ylim(0, 24)
                ax.set_zlim(0, 6)
                ax.set_zticks([3, 6])
                ax.set_xlabel('$x$', labelpad=50, fontsize=35)
                ax.set_ylabel('$y$', labelpad=15, fontsize=35)
                ax.set_zlabel('$z$', labelpad=10, fontsize=35)
                ax.tick_params(axis='both', which='major', labelsize=17)
                ax.set_box_aspect([10, 4, 1])
                ax.view_init(elev=30, azim=120)
                # ax.set_title(f"Temperature Heat Map for Day {int(n)} in {location}")
                fig.text(0.5, 0.80, f"Temperature Heat Map in {location}", ha='center', va='top', fontsize=20)
                pyplot.pause(0.1)

        print(f"DAY: {n}")

    pyplot.show()

    return(X, Y, Z, t, avg_room_temps)

X, Y, Z, t, results = diffuse(365)

fig = pyplot.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.plot(t, results)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Temp (C)')
ax.set_title(f'{location} w/ Solar Shading')
pyplot.tight_layout()
pyplot.grid()
pyplot.show()
