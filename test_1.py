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

import spiceypy as spice
import JupiterMag as jm
from CSField import CSField

UC = UniversalColor()
UC.set_palette()

F = ShareXaxis()
F.fontsize = 23
F.fontname = 'Liberation Sans Narrow'
F.set_default()

jm.Internal.Config(Model='jrm33', CartesianIn=True,
                   CartesianOut=True)
jm.Con2020.Config(equation_type='integral')


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
    i_phi0=5.2E-5,
    D=2.5,          # 2.5
    c1=5.0,         # 10.8
    c2=14.0,        # 14.6
    c3=46.0,        # 46.0
    c4=5.0,         # 5.0
    w1=-0.06,       # -0.29
    w2=1.39,        # 1.39
    w3=-0.90,       # -0.90
    w4=0,           # -0.025
    rho_cs0=4
)


# ===========================================================
# AZIMUTHAL CURRENT DENSITY
# ===========================================================
# FASTER
rho_cs = np.linspace(0.0, 100.0, 300)*RJ     # [m]
i_phi_new = csfield.i_phi_profile(
    MLT=0.0,
    rho_cs=rho_cs/RJ,
    each_term=True,
)   # [A m-1]
J_phi_new = (i_phi_new/(2*csfield.D*RJ))*(1E-6)*(RJ**2)  # [MA RJ-2]
I_rho_new = csfield.I_rho_profile(rho_cs=rho_cs/RJ, each_term=True)
print('i_phi_0 [MA RJ-1]:', csfield.i_phi0*1E-6*RJ)
print('np.min(J_phi_new):', np.min(J_phi_new))

# Wang+2022
J_phi_w22_6h = np.zeros(rho_cs.size)   # [MA RJ-2]
J_phi_w22_12h = np.zeros(rho_cs.size)  # [MA RJ-2]
J_phi_w22_18h = np.zeros(rho_cs.size)  # [MA RJ-2]
I_phi_w22_6h = np.zeros(rho_cs.size)   # [MA RJ-1]
I_phi_w22_18h = np.zeros(rho_cs.size)  # [MA RJ-1]
for i in range(rho_cs.size):
    J_phi_w22_6h[i] = csfield.J_phi_Wang22(rho_cs[i]/RJ, z=0, MLT=6.0)
    J_phi_w22_12h[i] = csfield.J_phi_Wang22(rho_cs[i]/RJ, z=0, MLT=12.0)
    J_phi_w22_18h[i] = csfield.J_phi_Wang22(rho_cs[i]/RJ, z=0, MLT=18.0)
    I_phi_w22_6h[i] = csfield.i_phi_Wang22(rho_cs[i]/RJ, MLT=6.0)
    I_phi_w22_18h[i] = csfield.i_phi_Wang22(rho_cs[i]/RJ, MLT=18.0)

# Con2020
rho_cs_con20 = np.linspace(7.8, 51.4, 100)*RJ         # [m]
i_phi_con20 = csfield.i_phi_Con20(rho_cs_con20/RJ)    # [A m-1]
J_phi_con20 = (i_phi_con20/(2*3.6*RJ))*(1E-6)*RJ*RJ   # [MA RJ-2]


# ===========================================================
# 面電流密度(方位角電流)の動径分布を描いてみる
# ===========================================================
F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.5, 4.7), dpi='XL')
F.hspace = 0.15
F.initialize()

xticks = np.arange(0, 65+1, 5)
xticklabels = np.arange(0, 65+1, 5, dtype=int)

F.set_xaxis(label=r'$\rho$ [$R_{\rm J}$]',
            min=0.0, max=35.0,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'$i_{\varphi}(R)$ [MA $R_{\rm J}^{-1}$]',
            min=-0.2, max=18.0,
            ticks=np.linspace(0, 18, 10),
            ticklabels=np.linspace(0, 18, 10, dtype=int),
            minor_num=2,)

F.ax.plot(rho_cs/RJ, i_phi_new*(1E-6)*RJ,
          color=UC.red, linewidth=3.0,
          label='New model')
F.ax.plot(rho_cs_con20/RJ, i_phi_con20*(1E-6)*RJ,
          color='k',
          label='Con2020')
F.ax.plot(rho_cs/RJ, I_phi_w22_6h,
          color=UC.blue,
          label='Wang+2022')
F.ax.plot(rho_cs/RJ, I_phi_w22_18h,
          color=UC.blue)
F.ax.fill_between(x=rho_cs/RJ,
                  y1=I_phi_w22_6h,
                  y2=I_phi_w22_18h,
                  ec=None,
                  fc=UC.lighterblue,
                  alpha=0.4, zorder=0.9,)

F.ax.set_title(r'Current density', weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=1, markerscale=1.0,
                  loc='upper right',
                  handlelength=1.6,
                  textcolor=False,
                  fontsize_scale=0.65,
                  handletextpad=0.4)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

plt.savefig('i_phi.png', bbox_inches='tight')


# ===========================================================
# 面電流密度(方位角電流)の動径分布を描いてみる (項ごとに)
# ===========================================================
F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.5, 4.7), dpi='XL')
F.hspace = 0.15
F.initialize()

xticks = np.arange(0, 65+1, 5)
xticklabels = np.arange(0, 65+1, 5, dtype=int)

F.set_xaxis(label=r'$\rho$ [$R_{\rm J}$]',
            min=0.0, max=35.0,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'$i_{\varphi}(R)$ [MA $R_{\rm J}^{-1}$]',
            min=-10.0, max=20.0,
            ticks=np.linspace(-10, 20, 7),
            ticklabels=np.linspace(-10, 20, 7, dtype=int),
            minor_num=5,)

F.ax.plot(rho_cs/RJ,
          csfield.w1*csfield.f1*(1E-6)*RJ,
          color='k', linestyle='-',
          label=r'$w_1 f_1$')
F.ax.plot(rho_cs/RJ,
          csfield.w2*csfield.f2*(1E-6)*RJ,
          color='k', linestyle='--',
          label=r'$w_2 f_2$')
F.ax.plot(rho_cs/RJ,
          csfield.w3*csfield.f3*(1E-6)*RJ,
          color='k', linestyle='-.',
          label=r'$w_3 f_3$')
if csfield.w4 != 0.0:
    F.ax.plot(rho_cs/RJ,
              csfield.w4*csfield.f4*(1E-6)*RJ,
              color='k', linestyle=(2, (2, 2)),
              label=r'$w_4 f_4$')

F.ax.plot(rho_cs/RJ, i_phi_new*(1E-6)*RJ,
          color=UC.red, linewidth=3.0,
          label='New model (total)')
F.ax.fill_between([0, 100], [-200, 0],
                  color=UC.lightgray,
                  alpha=0.4, zorder=-5)

F.ax.set_title(r'Current density', weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=2, markerscale=1.0,
                  loc='upper right',
                  handlelength=1.6,
                  textcolor=False,
                  fontsize_scale=0.65,
                  handletextpad=0.4)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

plt.savefig('i_phi_terms.png', bbox_inches='tight')


# ===========================================================
# 面電流密度(動径方向電流)の動径分布を描いてみる (項ごとに)
# ===========================================================
w11, w12, w13 = 0.0, 0.0, 50
F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.5, 4.7), dpi='XL')
F.hspace = 0.15
F.initialize()

xticks = np.arange(0, 65+1, 5)
xticklabels = np.arange(0, 65+1, 5, dtype=int)

F.set_xaxis(label=r'$\rho$ [$R_{\rm J}$]',
            min=0.0, max=35.0,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'$I_{\rho}(R)$ [MA]',
            min=-10.0, max=20.0,
            ticks=np.linspace(-10, 20, 7),
            ticklabels=np.linspace(0-10, 20, 7, dtype=int),
            minor_num=5,)

F.ax.plot(rho_cs/RJ,
          (w11*csfield.f11+w12*csfield.f12+w13*csfield.f13)*(1E+3),
          color=UC.red, linestyle='-',
          linewidth=2.0,
          label=r'New model')
F.ax.plot(rho_cs/RJ,
          (w11*csfield.f11)*(1E+3),
          color='k', linestyle='--',
          linewidth=1.0,
          label=r'$w_{11}f_{11}$')
F.ax.plot(rho_cs/RJ,
          (w12*csfield.f12)*(1E+3),
          color='k', linestyle='-.',
          linewidth=1.0,
          label=r'$w_{12}f_{12}$')
F.ax.plot(rho_cs/RJ,
          (w13*csfield.f13)*(1E+3),
          color='k', linestyle=(2, (2, 2)),
          linewidth=1.0,
          label=r'$w_{13}f_{13}$')
F.ax.plot(rho_cs/RJ,
          16.7*np.ones(rho_cs.size),
          color='k', linestyle='-',
          label=r'Con2020')

F.ax.set_title(r'Current density', weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=2, markerscale=1.0,
                  loc='upper right',
                  handlelength=1.6,
                  textcolor=False,
                  fontsize_scale=0.65,
                  handletextpad=0.4)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

plt.savefig('I_rho_terms.png', bbox_inches='tight')


# ===========================================================
# B1を計算してみる
# ===========================================================
r0 = 15.0*RJ                    # Radial distance [m]
theta0 = np.radians(89.9)       # Colatitude [rad]
phi0 = np.radians(360.0-112.0)   # East longitude [rad]

x0 = r0*np.sin(theta0)*np.cos(phi0)
y0 = r0*np.sin(theta0)*np.sin(phi0)
z0 = r0*np.cos(theta0)
# print('x0/RJ, y0/RJ, z0/RJ:', x0/RJ, y0/RJ, z0/RJ)

Bx1, By1, Bz1 = csfield.magnetic_field(x0/RJ, y0/RJ, z0/RJ)  # [T]
print('B_norm (New) [nT]:', math.sqrt(Bx1**2+By1**2+Bz1**2)*1E+9)

Bx1_c, By1_c, Bz1_c = jm.Con2020.Field(x0/RJ, y0/RJ, z0/RJ)  # [nT]
print('B_norm (Con2020) [nT]:', math.sqrt(Bx1_c**2+By1_c**2+Bz1_c**2))


# ===========================================================
# B1を計算してみる (動径方向に)
# ===========================================================
r0_arr = np.linspace(2, 40, 20)*RJ  # Radial discenta [m]
theta0 = np.radians(89.5)       # Colatitude [rad]
phi0 = np.radians(360.0-112.0)   # East longitude [rad]

x0 = r0_arr*np.sin(theta0)*np.cos(phi0)
y0 = r0_arr*np.sin(theta0)*np.sin(phi0)
z0 = r0_arr*np.cos(theta0)

Bx1 = np.zeros(x0.size)
By1 = np.zeros(x0.size)
Bz1 = np.zeros(x0.size)
Bphi1 = np.zeros(x0.size)
for i in range(x0.size):
    Bx1[i], By1[i], Bz1[i] = csfield.magnetic_field(x0[i]/RJ,
                                                    y0[i]/RJ,
                                                    z0[i]/RJ,)
Brho1 = Bx1*np.cos(phi0) + By1*np.sin(phi0)
Bphi1 = -Bx1*np.sin(phi0) + By1*np.cos(phi0)

Bx1_c, By1_c, Bz1_c = jm.Con2020.Field(x0/RJ, y0/RJ, z0/RJ)  # [nT]
Brho1_c = Bx1_c*np.cos(phi0) + By1_c*np.sin(phi0)
Bphi1_c = -Bx1_c*np.sin(phi0) + By1_c*np.cos(phi0)


# ===========================================================
# |B1|の動径分布を描いてみる
# ===========================================================
F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.5, 4.7), dpi='XL')
F.hspace = 0.15
F.initialize()

xticks = np.arange(5, 45+1, 5)
xticklabels = np.arange(5, 45+1, 5, dtype=int)

F.set_xaxis(label=r'$\rho$ [$R_{\rm J}$]',
            min=2.0, max=35.0,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'[nT]',
            min=0, max=200.0,
            ticks=np.linspace(-50, 200, 6),
            ticklabels=np.linspace(-50, 200.0, 6),
            minor_num=5,)

F.ax.plot(r0_arr/RJ,
          np.sqrt(Bx1**2+By1**2+Bz1**2)*1E+9,
          color=UC.red,
          linewidth=1.75,
          label=r'New model $|B|$')
F.ax.plot(r0_arr/RJ,
          Brho1*1E+9,
          color=UC.red,
          linestyle='--',
          linewidth=1.75,
          label=r'New model $B_\rho$')
F.ax.plot(r0_arr/RJ,
          Bphi1*1E+9,
          color=UC.red,
          linestyle='-.',
          linewidth=1.75,
          label=r'New model $B_\varphi$')
F.ax.plot(r0_arr/RJ,
          Bz1*1E+9,
          color=UC.red,
          linestyle=(2, (2, 2)),
          linewidth=1.75,
          label=r'New model $B_z$')
F.ax.plot(r0_arr/RJ,
          np.sqrt(Bx1_c**2+By1_c**2+Bz1_c**2),
          color='k',
          label=r'Con2020 $|B|$')
F.ax.plot(r0_arr/RJ,
          Brho1_c,
          color='k',
          linestyle='--',
          label=r'Con2020 $B_\rho$')
F.ax.plot(r0_arr/RJ,
          Bphi1_c,
          color='k',
          linestyle='-.',
          label=r'Con2020 $B_\varphi$')
F.ax.plot(r0_arr/RJ,
          Bz1_c,
          color='k',
          linestyle=(2, (2, 2)),
          label=r'Con2020 $B_z$')

F.ax.fill_between([0, 100], [-200, 0],
                  color=UC.lightgray,
                  alpha=0.4, zorder=-5)

F.ax.set_title(r'Magnetic field intensity', weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=1, markerscale=1.0,
                  loc='upper right',
                  handlelength=1.8,
                  textcolor=False,
                  fontsize_scale=0.65,
                  handletextpad=0.4)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

plt.savefig('B1_norm.png', bbox_inches='tight')


# ===========================================================
# 体積電流密度の動径分布を描いてみる
# ===========================================================
F = ShareXaxis()
F.fontsize = 21
F.fontname = 'Liberation Sans Narrow'

F.set_figparams(nrows=1, figsize=(5.5, 4.7), dpi='XL')
F.hspace = 0.15
F.initialize()

xticks = np.arange(0, 65+1, 5)
xticklabels = np.arange(0, 65+1, 5, dtype=int)

F.set_xaxis(label=r'$\rho$ [$R_{\rm J}$]',
            min=0.0, max=35.0,
            ticks=xticks,
            ticklabels=xticklabels,
            minor_num=5)
F.set_yaxis(ax_idx=0,
            label=r'$J_{\varphi}(R)$ [MA $R_{\rm J}^{-2}$]',
            min=0.0, max=2.5,
            ticks=np.linspace(0, 2.5, 6),
            ticklabels=np.linspace(0, 2.5, 6),
            minor_num=5,)

F.ax.plot(rho_cs/RJ, J_phi_new,
          color=UC.red, linewidth=3.0,
          label='New model')
F.ax.plot(rho_cs_con20/RJ, J_phi_con20,
          color='k',
          label='Con2020')
F.ax.plot(rho_cs/RJ, J_phi_w22_6h,
          color=UC.blue,
          label='Wang+2022')
F.ax.plot(rho_cs/RJ, J_phi_w22_18h,
          color=UC.blue,)
F.ax.fill_between(x=rho_cs/RJ,
                  y1=J_phi_w22_6h,
                  y2=J_phi_w22_18h,
                  ec=None,
                  fc=UC.lighterblue,
                  alpha=0.4, zorder=0.9,)

F.ax.set_title(r'Current density', weight='bold')

legend = F.legend(ax_idx=0,
                  ncol=1, markerscale=1.0,
                  loc='upper right',
                  handlelength=1.6,
                  textcolor=False,
                  fontsize_scale=0.65,
                  handletextpad=0.4)
legend_shadow(legend=legend, fig=F.fig, ax=F.ax, d=0.7)

plt.savefig('J_phi.png', bbox_inches='tight')
