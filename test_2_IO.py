import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.ticker as ptick
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from MyPlotRecipe.SharedX import ShareXaxis
from MyPlotRecipe.UniversalColor import UniversalColor
from MyPlotRecipe.legend_shadow import legend_shadow
import time

import spiceypy as spice
import JupiterMag as jm
from CSField import CSField

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'
F.set_default()

jm.Internal.Config(Model='jrm33',  Degree=18,
                   CartesianIn=True, CartesianOut=True)
jm.Con2020.Config(equation_type='integral')

spice.furnsh(
    '/Users/shin/Documents/Research/Jupiter/Codes/HST/kernel/cassMetaK.txt')
radii = spice.bodvrd("JUPITER", "RADII", 3)[1]
RJ_km = radii[0]
c = radii[2]
f = (RJ_km - c) / RJ_km


# ===========================================================
# CONSTANTS
# ===========================================================
MU0 = 4*np.pi*1E-7       # 真空中の透磁率 [H m-1]
AMU2KG = 1.66E-27        # [kg]
RJ = 71492.0E+3          # JUPITER RADIUS [m]


# ===========================================================
# CONFIGURE CSFIELD
# ===========================================================
csfield = CSField()
csfield.config(
    I_rho0=16.7,
    i_phi0=4.35E-4,
    D=3.6,          # 3.6 / 2.5
    c1=12.6,
    c2=22.0,
    c3=25.5,
    w1=-0.13,
    w2=2.18,
    w3=-2.11
)


# ===========================================================
# FIELD LINE TRACE TEST
# ===========================================================
ns = -1                          # -1: north; 1: south
r0 = 15.0*RJ                     # Radial distance [m]
theta0 = np.radians(89.9)        # Colatitude [rad]
phi0 = np.radians(360.0-112.0)   # East longitude [rad]

x0 = r0*np.sin(theta0)*np.cos(phi0)
y0 = r0*np.sin(theta0)*np.sin(phi0)
z0 = r0*np.cos(theta0)

Niter = int(70000)

ds0 = 3000000.0   # [m]
p = 3.0

x, y, z = x0, y0, z0
start = time.time()
for i in range(Niter):
    # Community codes
    Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)       # [nT]
    Bx1, By1, Bz1 = csfield.magnetic_field(x/RJ, y/RJ, z/RJ)  # [T]
    Bx = (Bx0*1E-9)+Bx1     # [T]
    By = (By0*1E-9)+By1     # [T]
    Bz = (Bz0*1E-9)+Bz1     # [T]
    B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

    ds = ds0/(np.log10(B0*1E+9))**p      # [m]

    # 座標更新 (x, y, z)
    x += (ds*Bx[0]/B0)*ns
    y += (ds*By[0]/B0)*ns
    z += (ds*Bz[0]/B0)*ns

    # if i % 200 == 0:
    #     print('ds [km]:', ds/1000.0)

    # 座標更新 (r, theta, phi)
    rs = math.sqrt(x**2 + y**2 + z**2)
    thetRJ_km = math.acos(z/rs)
    phi = math.atan2(y, x)

    # 終了条件
    if rs < (1.0*RJ+2000.0E+3):
        lon_gr, lat_gr, alt_gr = spice.recpgr('JUPITER',
                                              np.array([x/1000.0,
                                                        y/1000.0,
                                                        z/1000.0]),
                                              RJ_km,
                                              f)
        if abs(alt_gr-900.0)*1000.0 <= 0.5*ds:
            print('End altitude [km]:', round(alt_gr, 2))
            break
print('Time [sec]:', time.time()-start)


# ===========================================================
# CONNERNEY+2022 REFERENCE
# ===========================================================
ref_data_N = np.loadtxt(
    '/Users/shin/Documents/Research/Juno/UVS/Codes2/data/JRM33/satellite_foot_N.txt', skiprows=3)
ref_data_S = np.loadtxt(
    '/Users/shin/Documents/Research/Juno/UVS/Codes2/data/JRM33/satellite_foot_S.txt', skiprows=3)

eq_wlon_ref = ref_data_N[:, 0]          # [deg]
ifp_ref_pos_N = ref_data_N[:, 3:5]      # (lat, wlon) [deg]
ifp_ref_pos_S = ref_data_S[:, 3:5]      # (lat, wlon) [deg]
efp_ref_pos_N = ref_data_N[:, 5:7]      # (lat, wlon) [deg]
efp_ref_pos_S = ref_data_S[:, 5:7]      # (lat, wlon) [deg]
gfp_ref_pos_N = ref_data_N[:, 7:9]      # (lat, wlon) [deg]
gfp_ref_pos_S = ref_data_S[:, 7:9]      # (lat, wlon) [deg]


# ===========================================================
# IO FOOTPATH REFERENCE
# ===========================================================
r0 = 5.9*RJ                      # Radial distance [m]
theta0 = np.radians(89.9)        # Colatitude [rad]
phi0 = np.radians(360.0-np.linspace(0.0, 359.9, 120))   # East longitude [rad]

x0 = r0*np.sin(theta0)*np.cos(phi0)
y0 = r0*np.sin(theta0)*np.sin(phi0)
z0 = r0*np.cos(theta0)*np.ones(phi0.size)

Niter = int(70000)

ds0 = 3000000.0   # [m]
p = 3.0

# 北半球
ns = -1     # -1: north; 1: south
result_arr = np.zeros((phi0.size, 4))
start = time.time()
for j in range(x0.size):
    x, y, z = x0[j], y0[j], z0[j]
    for i in range(Niter):
        # Community codes
        Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)       # [nT]
        Bx1, By1, Bz1 = csfield.magnetic_field(x/RJ, y/RJ, z/RJ)  # [T]
        Bx = (Bx0*1E-9)+Bx1     # [T]
        By = (By0*1E-9)+By1     # [T]
        Bz = (Bz0*1E-9)+Bz1     # [T]
        B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

        ds = ds0/(np.log10(B0*1E+9))**p      # [m]

        # 座標更新 (x, y, z)
        x += (ds*Bx[0]/B0)*ns
        y += (ds*By[0]/B0)*ns
        z += (ds*Bz[0]/B0)*ns

        # if i % 200 == 0:
        #     print('ds [km]:', ds/1000.0)

        # 座標更新 (r, theta, phi)
        rs = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z/rs)
        phi = math.atan2(y, x)

        # 終了条件
        if rs < (1.0*RJ+2000.0E+3):
            # ds = 5000.0     # [m]
            lon_gr, lat_gr, alt_gr = spice.recpgr('JUPITER',
                                                  np.array([x/1000.0,
                                                            y/1000.0,
                                                            z/1000.0]),
                                                  RJ_km,
                                                  f)
            if abs(alt_gr-1.0)*1000.0 <= 0.5*ds:
                print('End altitude [km]:', round(alt_gr, 2))
                break
    """print(round(360.0-np.degrees(phi0[j]), 2),
          round(rs/RJ, 2),
          round(np.mod(np.degrees(0.5*np.pi-theta), 360.0), 2),
          round(np.mod(np.degrees(2*np.pi-phi), 360.0), 2)
          )"""
    result_arr[j, :] = np.array([
        round(360.0-np.degrees(phi0[j]), 3),
        round(rs/RJ, 3),
        round(np.mod(np.degrees(0.5*np.pi-theta), 360.0), 3),
        round(np.mod(np.degrees(2*np.pi-phi), 360.0), 3)
    ]
    )

print(result_arr)
print('Time [sec]:', time.time()-start)
np.savetxt('results/FP_IO_NORTH_FASTER.txt', result_arr)

# 南半球
ns = 1     # -1: north; 1: south
result_arr = np.zeros((phi0.size, 4))
start = time.time()
for j in range(x0.size):
    x, y, z = x0[j], y0[j], z0[j]
    for i in range(Niter):
        # Community codes
        Bx0, By0, Bz0 = jm.Internal.Field(x/RJ, y/RJ, z/RJ)       # [nT]
        Bx1, By1, Bz1 = csfield.magnetic_field(x/RJ, y/RJ, z/RJ)  # [T]
        Bx = (Bx0*1E-9)+Bx1     # [T]
        By = (By0*1E-9)+By1     # [T]
        Bz = (Bz0*1E-9)+Bz1     # [T]
        B0 = math.sqrt(Bx[0]**2+By[0]**2+Bz[0]**2)      # [T]

        ds = ds0/(np.log10(B0*1E+9))**p      # [m]

        # 座標更新 (x, y, z)
        x += (ds*Bx[0]/B0)*ns
        y += (ds*By[0]/B0)*ns
        z += (ds*Bz[0]/B0)*ns

        # if i % 200 == 0:
        #     print('ds [km]:', ds/1000.0)

        # 座標更新 (r, theta, phi)
        rs = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z/rs)
        phi = math.atan2(y, x)

        # 終了条件
        if rs < (1.0*RJ+2000.0E+3):
            # ds = 5000.0     # [m]
            lon_gr, lat_gr, alt_gr = spice.recpgr('JUPITER',
                                                  np.array([x/1000.0,
                                                            y/1000.0,
                                                            z/1000.0]),
                                                  RJ_km,
                                                  f)
            if abs(alt_gr-1.0)*1000.0 <= 0.5*ds:
                print('End altitude [km]:', round(alt_gr, 2))
                break
    """print(round(360.0-np.degrees(phi0[j]), 2),
          round(rs/RJ, 2),
          round(np.mod(np.degrees(0.5*np.pi-theta), 360.0), 2),
          round(np.mod(np.degrees(2*np.pi-phi), 360.0), 2)
          )"""
    result_arr[j, :] = np.array([
        round(360.0-np.degrees(phi0[j]), 3),
        round(rs/RJ, 3),
        round(np.mod(np.degrees(0.5*np.pi-theta), 360.0), 3),
        round(np.mod(np.degrees(2*np.pi-phi), 360.0), 3)
    ]
    )

print('Time [sec]:', time.time()-start)
np.savetxt('results/FP_IO_SOUTH_FASTER.txt', result_arr)

F = ShareXaxis()
F.fontsize = 19
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=2, figsize=(5.5, 9.5), dpi='XL')
F.initialize()

F.panelname = [' a. North ', ' b. South ']

xticks = np.arange(0, 360+1, 45)
xticklabels = np.arange(0, 360+1, 45, dtype=int)

F.set_xaxis(label=r'$X$ [$R_{\rm J}$]',
            min=-0.7, max=0.7,
            ticks=np.linspace(-6, 6, 7)/10,
            ticklabels=np.linspace(-6, 6, 7)/10,
            minor_num=2)
F.set_yaxis(ax_idx=0, label=r'$Y$ [$R_{\rm J}$]',
            min=-0.7, max=0.7,
            ticks=np.linspace(-6, 6, 7)/10,
            ticklabels=np.linspace(-6, 6, 7)/10,
            minor_num=2)
F.set_yaxis(ax_idx=1, label=r'$Y$ [$R_{\rm J}$]',
            min=-0.7, max=0.7,
            ticks=np.linspace(-6, 6, 7)/10,
            ticklabels=np.linspace(-6, 6, 7)/10,
            minor_num=2)

# FASTER footpath at 900 km
result_arr = np.loadtxt('results/FP_IO_NORTH_FASTER.txt')
theta_modelfp = np.radians(90.0-result_arr[:, 2])
phi_modelfp = np.radians(360.0-result_arr[:, 3])
x_modelfp_polar = np.sin(theta_modelfp)*np.cos(phi_modelfp)
y_modelfp_polar = np.sin(theta_modelfp)*np.sin(phi_modelfp)
F.ax[0].plot(x_modelfp_polar, y_modelfp_polar, c=UC.pink,
             linewidth=2.0, linestyle='-', zorder=0.95)

# FASTER footpath at 900 km
result_arr = np.loadtxt('results/FP_IO_SOUTH_FASTER.txt')
theta_modelfp = np.radians(90.0-result_arr[:, 2])
phi_modelfp = np.radians(360.0-result_arr[:, 3])
x_modelfp_polar = np.sin(theta_modelfp)*np.cos(phi_modelfp)
y_modelfp_polar = np.sin(theta_modelfp)*np.sin(phi_modelfp)
F.ax[1].plot(x_modelfp_polar, y_modelfp_polar, c=UC.pink,
             linewidth=2.0, linestyle='-', zorder=0.95)

# Connerney+ 2022 surface reference 0 km
theta_modelfp = np.radians(90.0-ifp_ref_pos_N[:, 0])
phi_modelfp = np.radians(360.0-ifp_ref_pos_N[:, 1])
x_modelfp_polar = np.sin(theta_modelfp)*np.cos(phi_modelfp)
y_modelfp_polar = np.sin(theta_modelfp)*np.sin(phi_modelfp)
F.ax[0].plot(x_modelfp_polar, y_modelfp_polar, c='k',
             linewidth=1.6, linestyle=(0, (4, 7)), zorder=0.95)

# Connerney+ 2022 surface reference 0 km
theta_modelfp = np.radians(90.0-ifp_ref_pos_S[:, 0])
phi_modelfp = np.radians(360.0-ifp_ref_pos_S[:, 1])
x_modelfp_polar = np.sin(theta_modelfp)*np.cos(phi_modelfp)
y_modelfp_polar = np.sin(theta_modelfp)*np.sin(phi_modelfp)
F.ax[1].plot(x_modelfp_polar, y_modelfp_polar, c='k',
             linewidth=1.6, linestyle=(0, (4, 7)), zorder=0.95)

# Longitudinal grid
s3wlon_grid = np.linspace(0, 360, 9)
phi_grid = np.radians(360-s3wlon_grid)
for i in range(phi_grid.size):
    for j in range(2):
        F.ax[j].plot((0, np.cos(phi_grid[i])),
                     (0, np.sin(phi_grid[i])),
                     color=UC.lightgray, linestyle='--',
                     linewidth=1.0, zorder=0.5)
        if i in [7, phi_grid.size-1]:
            continue
        F.textbox(ax_idx=j,
                  x=0.63*np.cos(phi_grid[i]),
                  y=0.63*np.sin(phi_grid[i]),
                  text=str(int(s3wlon_grid[i]))+'˚W',
                  fontsize=F.fontsize*0.6,
                  horizontalalignment='center',
                  textshadow=False,
                  textcolor='k',
                  facealpha=0.0,
                  edgecolor=(0, 0, 0, 0), )

# Latitudinal grid
lat_grid = np.arange(0, 90+1, 15)
for i in range(lat_grid.size):
    for j in range(2):
        circle = plt.Circle(xy=(0, 0),
                            radius=math.cos(math.radians(90.0-lat_grid[i])),
                            fill=False, ec=UC.lightgray, linewidth=1,
                            linestyle='--', zorder=0.5)
        F.ax[j].add_patch(circle)
    if i in [0, 1, lat_grid.size-2, lat_grid.size-1]:
        continue
    F.textbox(ax_idx=0,
              x=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
              y=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
              text=str(int(90-lat_grid[i]))+'˚N',
              fontsize=F.fontsize*0.6,
              horizontalalignment='center',
              textshadow=False,
              textcolor='k',
              facealpha=0.0,
              edgecolor=(0, 0, 0, 0), )
    F.textbox(ax_idx=1,
              x=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
              y=np.cos(math.radians(90.0-lat_grid[i]))/1.4142,
              text=str(int(90-lat_grid[i]))+'˚S',
              fontsize=F.fontsize*0.6,
              horizontalalignment='center',
              textshadow=False,
              textcolor='k',
              facealpha=0.0,
              edgecolor=(0, 0, 0, 0), )

plt.savefig('img/FP_IO_POLAR_FASTER.png', bbox_inches='tight')
