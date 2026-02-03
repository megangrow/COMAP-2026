#TODO change offset for data
#TODO make sure to include pvlib
if (location == "MIAMI"):
    T = lambda t: 4.9188111757142 * np.sin( 0.017200792322545 * t - 1193.71990384336) +\
            0.582462690124397 * np.cos( 760.266156033683 * t + 2095.71540499492) + 24.5960040916041

    ###Compute Angle of the SUN
    def sun_angle(d):
        return ((np.pi / 2) - ((25.774 * np.pi) / 180) + pvlib.solarposition.declination_cooper69(d)) * (180 / np.pi)

elif (location == "ANCHORAGE"):
    T = lambda t: 0.69585135778619 * np.sin(0.00137029999867194 * t - 2432.77084285537) +\
            12.2251580183288 * np.cos(0.017201860290628 * t + 36557989.8870071) + 3.13224850133946

    ###Compute Angle of the SUN
    def sun_angle(d):
        return ((np.pi / 2) - ((61.21806 * np.pi) / 180) + pvlib.solarposition.declination_cooper69(d)) * (180 / np.pi)

else:
    raise RuntimeError("\n Please check location and variables!!! \n")