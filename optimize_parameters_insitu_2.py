import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from matplotlib.ticker import FixedLocator
from MyPlotRecipe.SharedX import ShareXaxis
from MyPlotRecipe.UniversalColor import UniversalColor
from MyPlotRecipe.legend_shadow import legend_shadow
import time

import spiceypy as spice
import JupiterMag as jm
from CSField import CSField

from multiprocessing import Pool
from functools import partial

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
# IMPORT Juno-MAG DATA
# ===========================================================
def load_MAGdata(PJ_num):
    dir = 'data/FGM/PJ'+str(PJ_num).zfill(2)+'/'
    time_arr = np.loadtxt(
        dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_time.txt')
    Bx_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_Bx.txt')
    By_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_By.txt')
    Bz_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_Bz.txt')
    rx_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_rx.txt')
    ry_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_ry.txt')
    rz_pc = np.loadtxt(dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_rz.txt')
    JunoMLT = np.loadtxt(
        dir+'fgm_PJ'+str(PJ_num).zfill(2)+'_pc_2min_JunoMLT.txt')

    rx_pc *= 1/71492.0  # [RJ]
    ry_pc *= 1/71492.0  # [RJ]
    rz_pc *= 1/71492.0  # [RJ]

    return rx_pc, ry_pc, rz_pc, time_arr, Bx_pc, By_pc, Bz_pc, JunoMLT


# ===========================================================
# CALCULATE THE INTERNAL FIELD WITH JRM33
# ===========================================================
def B_internal(n_time_arr, rx_pc, ry_pc, rz_pc):
    Bx0_pc = np.zeros(n_time_arr)
    By0_pc = np.zeros(n_time_arr)
    Bz0_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        Bx0_pc[i], By0_pc[i], Bz0_pc[i] = jm.Internal.Field(rx_pc[i],
                                                            ry_pc[i],
                                                            rz_pc[i])  # [nT]
    return Bx0_pc, By0_pc, Bz0_pc


def dB_obs(
        time_arr,
        rx_pc, ry_pc, rz_pc,
        Bx_pc, By_pc, Bz_pc,
        cos_theta, sin_theta,
        cos_phi, sin_phi):
    Bx0_pc, By0_pc, Bz0_pc = B_internal(time_arr.size, rx_pc, ry_pc, rz_pc)
    dBx = Bx_pc - Bx0_pc
    dBy = By_pc - By0_pc
    dBz = Bz_pc - Bz0_pc
    dBr = dBx*sin_theta*cos_phi + dBy*sin_theta*sin_phi + dBz*cos_theta
    dBtheta = dBx*cos_theta*cos_phi + dBy*cos_theta*sin_phi - dBz*sin_theta
    dBphi = -dBx*sin_phi + dBy*cos_phi

    return Bx0_pc, By0_pc, Bz0_pc, dBx, dBy, dBz, dBr, dBtheta, dBphi


# ===========================================================
# PLOT THE BEST FIT
# ===========================================================
def plot_best(
        time_arr, r_pc, rho_cs, z_cs,
        dBr, dBtheta, dBphi,
        dBr2, dBtheta2, dBphi2,
        dBphics, dBphics2, JunoMLT,
):
    if PJ_num in [17]:
        r_ticks_ref = np.arange(8, 30+1, 1)
    else:
        r_ticks_ref = np.arange(5, 30+1, 1)
    r_ticks = np.zeros((r_ticks_ref.size*2, 2))
    for i in range(r_ticks_ref.size):
        r_i = r_ticks_ref[-1]-i
        time_i = time_arr[np.where(r_pc < r_i)[0][0]]
        r_ticks[i, 0] = r_i
        r_ticks[i, 1] = time_i

        time_i = time_arr[np.where(r_pc < r_i)[0][-1]]
        r_ticks[-i-1, 0] = r_i
        r_ticks[-i-1, 1] = time_i

    F = ShareXaxis()
    F.fontsize = 21
    F.fontname = 'Liberation Sans Narrow'

    F.set_figparams(nrows=6, figsize=(11.0, 16.0), dpi='XL')
    F.hspace = 0.2
    F.initialize()

    xticks = r_ticks[::3, 1]
    xticklabels = r_ticks[::3, 0].astype(int)

    F.set_xaxis(label=r'Distance from Jupiter [$R_{\rm J}$]',
                min=int(time_arr[0]),
                max=int(time_arr[-1])+1,
                ticks=xticks,
                ticklabels=xticklabels,
                minor_num=None)
    for i in range(5):
        F.ax[i].xaxis.set_minor_locator(FixedLocator(r_ticks[:, 1]))
    for i in range(4):
        i += 1
        ax_upper = F.ax[i].twiny()
        ax_upper.set_xlim(int(time_arr[0]), int(time_arr[-1])+1)
        ax_upper.set_xticks(xticks)
        ax_upper.xaxis.set_minor_locator(FixedLocator(r_ticks[:, 1]))
        ax_upper.tick_params(labeltop=False)

    F.set_yaxis(ax_idx=0,
                label=r'$\delta B_{r}$ [nT]',
                min=-100, max=100,
                ticks=np.linspace(-100, 100, 5),
                ticklabels=np.linspace(-100, 100, 5, dtype=int),
                minor_num=4)
    F.set_yaxis(ax_idx=1,
                label=r'$\delta B_{\theta}$ [nT]',
                min=-150, max=30,
                ticks=np.linspace(-150, 30, 7),
                ticklabels=np.linspace(-150, 30, 7, dtype=int),
                minor_num=4)
    F.set_yaxis(ax_idx=2,
                label=r'$\delta B_{\varphi}$ [nT]',
                min=-20, max=20,
                ticks=np.linspace(-20, 20, 5),
                ticklabels=np.linspace(-20, 20, 5, dtype=int),
                minor_num=4)
    F.set_yaxis(ax_idx=3,
                label=r'$\delta B_{\varphi_{\rm cs}}$ [nT]',
                min=-20, max=20,
                ticks=np.linspace(-20, 20, 5),
                ticklabels=np.linspace(-20, 20, 5, dtype=int),
                minor_num=4)
    F.set_yaxis(ax_idx=4,
                label=r'$z_{\rm cs}$ [$R_{\rm J}$]',
                min=-30, max=20,
                ticks=np.linspace(-30, 20, 6),
                ticklabels=np.linspace(-30, 20, 6, dtype=int),
                minor_num=5)
    F.set_yaxis(ax_idx=5,
                label=r'$\rho_{\rm cs}$ [$R_{\rm J}$]',
                min=0, max=50,
                ticks=np.linspace(0, 50, 6),
                ticklabels=np.linspace(0, 50, 6, dtype=int),
                minor_num=10)
    F.ax[0].plot(time_arr, dBr, color='k',
                 linewidth=1.0, label='Observations', zorder=2.0)
    F.ax[0].plot(time_arr, dBr2, color=UC.red,
                 linewidth=1.9, label='FASTER', zorder=1.0)
    F.ax[1].plot(time_arr, dBtheta, color='k',
                 linewidth=1.0, label='Observations', zorder=2.0)
    F.ax[1].plot(time_arr, dBtheta2, color=UC.red,
                 linewidth=1.9, label='FASTER', zorder=1.0)
    F.ax[2].plot(time_arr, dBphi, color='k',
                 linewidth=1.0, label='Observations', zorder=2.0)
    F.ax[2].plot(time_arr, dBphi2, color=UC.red,
                 linewidth=1.9, label='FASTER', zorder=1.0)
    F.ax[3].plot(time_arr, dBphics, color='k',
                 linewidth=1.0, label='Observations', zorder=2.0)
    F.ax[3].plot(time_arr, dBphics2, color=UC.red,
                 linewidth=1.9, label='FASTER', zorder=1.0)
    F.ax[4].plot(time_arr, z_cs, color='k',
                 linewidth=1.0, label='Juno trajectory', zorder=1.0)
    F.ax[5].plot(time_arr, rho_cs, color='k',
                 linewidth=1.0, label='Juno trajectory', zorder=1.0)

    # z_cs current sheet shade
    rng = np.random.default_rng(42)
    noise = rng.random((500, 1200))
    im = F.ax[4].imshow(
        noise,
        extent=[int(time_arr[0]), int(time_arr[-1])+1, -2.5, 2.5],
        origin="lower",
        cmap="Blues",
        alpha=0.20,
        interpolation="nearest",
        aspect="auto",
    )

    # Shades near Jupiter (< 5.0 RJ)
    inner_idx = np.where((rho_cs < np.min(r_ticks_ref)))[0]
    for i in range(F.nrows):
        F.ax[i].axvspan(time_arr[inner_idx[0]],
                        time_arr[inner_idx[-1]],
                        fc=UC.lightgray, ec=None, zorder=2.5)

    # Shades in the outer region
    outer_idx = np.where((r_pc < np.max(r_ticks_ref)))[0]
    for i in range(F.nrows):
        F.ax[i].axvspan(time_arr[0],
                        time_arr[outer_idx[0]],
                        fc=UC.lightgray, alpha=0.5, ec=None, zorder=2.5)
        F.ax[i].axvspan(time_arr[outer_idx[-1]],
                        time_arr[-1],
                        fc=UC.lightgray, alpha=0.5, ec=None, zorder=2.5)

    # y = 0 line
    for i in range(F.nrows):
        F.ax[i].axhline(y=0.0, color=UC.gray, linewidth=0.7, zorder=0.9)

    F.ax[0].text(0.99, 0.9, 'PJ'+str(PJ_num).zfill(2),
                 color='k',
                 fontweight='bold',
                 fontsize=F.fontsize,
                 verticalalignment='top',
                 horizontalalignment='right',
                 transform=F.ax[0].transAxes)

    if PJ_num in [47, 71]:
        legend_loc = 'lower left'
    else:
        legend_loc = 'lower right'
    legend = F.legend(ax_idx=1,
                      ncol=1, markerscale=1.0,
                      loc=legend_loc,
                      handlelength=0.7,
                      textcolor=False,
                      fontsize_scale=0.85,
                      handletextpad=0.4)
    legend_shadow(legend=legend, fig=F.fig, ax=F.ax[0], d=0.7)

    plt.savefig('img/'+savedir[8:-8]+'/PJ'+str(PJ_num).zfill(2)+'.png',
                bbox_inches='tight')

    F.close()
    plt.close()

    return None


# ===========================================================
# CALCULATE THE EXTERNAL FIELD WITH FASTER (POLOIDAL)
# ===========================================================
def calc_poloidal(
        params,
        n_time_arr,
        rx_pc, ry_pc, rz_pc, r_pc,
        cos_theta, sin_theta, cos_phi, sin_phi,
        dBx, dBy, dBz,
        dBr, dBtheta, dBphi,
        MLT, PJ_num,
        a0=0.0, MLT0=0.0, b1=0.0):
    # ===========================================================
    # CONFIGURE CSFIELD
    # ===========================================================
    i_phi0 = params
    csfield = CSField()

    dBx2_pc = np.zeros(n_time_arr)
    dBy2_pc = np.zeros(n_time_arr)
    dBz2_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        csfield.config(
            i_phi0=i_phi0,
            D=2.5,
            a0=a0,
            MLT0=MLT0,
            b1=b1,
        )
        dBx2_pc[i], dBy2_pc[i], dBz2_pc[i] = csfield.magnetic_field(rx_pc[i],
                                                                    ry_pc[i],
                                                                    rz_pc[i],
                                                                    MLT[i])  # [T]

    dBx2_pc *= 1E+9    # [nT]
    dBy2_pc *= 1E+9    # [nT]
    dBz2_pc *= 1E+9    # [nT]
    dBr2_pc = dBx2_pc*sin_theta*cos_phi + dBy2_pc * \
        sin_theta*sin_phi + dBz2_pc*cos_theta
    dBtheta2_pc = dBx2_pc*cos_theta*cos_phi + \
        dBy2_pc*cos_theta*sin_phi - dBz2_pc*sin_theta
    dBphi2_pc = -dBx2_pc*sin_phi + dBy2_pc*cos_phi

    # ===========================================================
    # RMSE（Root Mean Squared Error）
    # ===========================================================
    x_cs, y_cs, _ = csfield._sys3_2_cs(rx_pc, ry_pc, rz_pc)
    phi_cs = np.arctan2(y_cs, x_cs)
    rho_cs = np.sqrt(x_cs**2 + y_cs**2)
    in_30rj = np.where(r_pc < 30.0)[0][0]
    if PJ_num in [17]:
        in_5rj = np.where(rho_cs < 8.0)[0][0]
    else:
        in_5rj = np.where(rho_cs < 5.0)[0][0]
    in_num = in_5rj-in_30rj-1
    out_30rj = np.where(r_pc < 30.0)[0][-1]
    if PJ_num in [17]:
        out_5rj = np.where(rho_cs < 8.0)[0][-1]
    else:
        out_5rj = np.where(rho_cs < 5.0)[0][-1]
    out_num = out_30rj-out_5rj-1

    # ===========================================================
    # Sys3 (x, y, z) -> CS (rho, phi, z)
    # ===========================================================
    dBx_cs, dBy_cs, dBz_cs = csfield._sys3_2_cs(dBx, dBy, dBz)
    dBrho_cs = dBx_cs*np.cos(phi_cs) + dBy_cs*np.sin(phi_cs)
    dBphi_cs = -dBx_cs*np.sin(phi_cs) + dBy_cs*np.cos(phi_cs)

    dBx2_cs, dBy2_cs, dBz2_cs = csfield._sys3_2_cs(dBx2_pc, dBy2_pc, dBz2_pc)
    dBrho2_cs = dBx2_cs*np.cos(phi_cs) + dBy2_cs*np.sin(phi_cs)
    dBphi2_cs = -dBx2_cs*np.sin(phi_cs) + dBy2_cs*np.cos(phi_cs)

    # ===========================================================
    # Total RMSE
    # ===========================================================
    RMS_poloidal_cs = math.sqrt(
        (
            np.sum(
                (dBrho_cs[in_30rj:in_5rj]-dBrho2_cs[in_30rj:in_5rj])**2 +
                (dBz_cs[in_30rj:in_5rj]-dBz2_cs[in_30rj:in_5rj])**2
            ) + np.sum(
                (dBrho_cs[out_5rj:out_30rj]-dBrho2_cs[out_5rj:out_30rj])**2 +
                (dBz_cs[out_5rj:out_30rj]-dBz2_cs[out_5rj:out_30rj])**2
            )
        )/(2*in_num+2*out_num)
    )

    return RMS_poloidal_cs


# ===========================================================
# CALCULATE THE EXTERNAL FIELD WITH FASTER (TOROIDAL)
# ===========================================================
def calc_toroidal(
        params,
        n_time_arr,
        rx_pc, ry_pc, rz_pc, r_pc,
        cos_theta, sin_theta, cos_phi, sin_phi,
        dBx, dBy, dBz,
        dBr, dBtheta, dBphi,
        MLT, PJ_num,
        a0=0.0, MLT0=0.0, b1=0.0):
    # ===========================================================
    # CONFIGURE CSFIELD
    # ===========================================================
    I_rho0 = params
    csfield = CSField()

    dBx2_pc = np.zeros(n_time_arr)
    dBy2_pc = np.zeros(n_time_arr)
    dBz2_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        csfield.config(
            I_rho0=I_rho0,
            D=2.5,
            a0=a0,
            MLT0=MLT0,
            b1=b1,
        )
        dBx2_pc[i], dBy2_pc[i], dBz2_pc[i] = csfield.magnetic_field(rx_pc[i],
                                                                    ry_pc[i],
                                                                    rz_pc[i],
                                                                    MLT[i])  # [T]

    dBx2_pc *= 1E+9    # [nT]
    dBy2_pc *= 1E+9    # [nT]
    dBz2_pc *= 1E+9    # [nT]
    dBr2_pc = dBx2_pc*sin_theta*cos_phi + dBy2_pc * \
        sin_theta*sin_phi + dBz2_pc*cos_theta
    dBtheta2_pc = dBx2_pc*cos_theta*cos_phi + \
        dBy2_pc*cos_theta*sin_phi - dBz2_pc*sin_theta
    dBphi2_pc = -dBx2_pc*sin_phi + dBy2_pc*cos_phi

    # ===========================================================
    # RMSE（Root Mean Squared Error）
    # ===========================================================
    x_cs, y_cs, _ = csfield._sys3_2_cs(rx_pc, ry_pc, rz_pc)
    phi_cs = np.arctan2(y_cs, x_cs)
    rho_cs = np.sqrt(x_cs**2 + y_cs**2)
    in_30rj = np.where(r_pc < 30.0)[0][0]
    if PJ_num in [17]:
        in_5rj = np.where(rho_cs < 8.0)[0][0]
    else:
        in_5rj = np.where(rho_cs < 5.0)[0][0]
    in_num = in_5rj-in_30rj-1
    out_30rj = np.where(r_pc < 30.0)[0][-1]
    if PJ_num in [17]:
        out_5rj = np.where(rho_cs < 8.0)[0][-1]
    else:
        out_5rj = np.where(rho_cs < 5.0)[0][-1]
    out_num = out_30rj-out_5rj-1

    # ===========================================================
    # Sys3 (x, y, z) -> CS (rho, phi, z)
    # ===========================================================
    dBx_cs, dBy_cs, dBz_cs = csfield._sys3_2_cs(dBx, dBy, dBz)
    dBrho_cs = dBx_cs*np.cos(phi_cs) + dBy_cs*np.sin(phi_cs)
    dBphi_cs = -dBx_cs*np.sin(phi_cs) + dBy_cs*np.cos(phi_cs)

    dBx2_cs, dBy2_cs, dBz2_cs = csfield._sys3_2_cs(dBx2_pc, dBy2_pc, dBz2_pc)
    dBrho2_cs = dBx2_cs*np.cos(phi_cs) + dBy2_cs*np.sin(phi_cs)
    dBphi2_cs = -dBx2_cs*np.sin(phi_cs) + dBy2_cs*np.cos(phi_cs)

    # ===========================================================
    # Total RMSE
    # ===========================================================
    RMS_toroidal_cs = math.sqrt(
        (
            np.sum(
                (dBphi_cs[in_30rj:in_5rj]-dBphi2_cs[in_30rj:in_5rj])**2
            ) + np.sum(
                (dBphi_cs[out_5rj:out_30rj]-dBphi2_cs[out_5rj:out_30rj])**2
            )
        )/(in_num+out_num)
    )

    return RMS_toroidal_cs


def main():
    rx_pc, ry_pc, rz_pc, time_arr, Bx_pc, By_pc, Bz_pc, JunoMLT = load_MAGdata(
        PJ_num
    )
    r_pc = np.sqrt(rx_pc**2+ry_pc**2+rz_pc**2)  # [RJ]
    cos_phi = rx_pc/r_pc                        # [rad]
    sin_phi = ry_pc/r_pc                        # [rad]
    cos_theta = rz_pc/r_pc
    sin_theta = np.sqrt(rx_pc**2+ry_pc**2)/r_pc

    _, _, _, dBx, dBy, dBz, dBr, dBtheta, dBphi = dB_obs(
        time_arr, rx_pc, ry_pc, rz_pc,
        Bx_pc, By_pc, Bz_pc,
        cos_theta, sin_theta, cos_phi, sin_phi
    )

    start = time.time()
    # ===========================================================
    # FIT POLOIDAL MAGNETIC FIELD
    # ===========================================================
    args = i_phi0_arr
    worker = partial(
        calc_poloidal,
        n_time_arr=time_arr.size,
        rx_pc=rx_pc,
        ry_pc=ry_pc,
        rz_pc=rz_pc,
        r_pc=r_pc,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        dBx=dBx,
        dBy=dBy,
        dBz=dBz,
        dBr=dBr,
        dBtheta=dBtheta,
        dBphi=dBphi,
        MLT=JunoMLT,
        PJ_num=PJ_num,
        a0=a0,
        MLT0=MLT0,
        b1=b1_arr,
    )
    with Pool(processes=parallel) as pool:
        results = pool.map(worker, args)
    RMS_poloidal_cs = np.array(results)
    min_idx = np.argmin(RMS_poloidal_cs)
    fit_err_idx = np.where(
        abs(RMS_poloidal_cs-np.min(RMS_poloidal_cs))
        < 0.05*np.min(RMS_poloidal_cs)
    )
    i_phi0_min, i_phi0_max = i_phi0_arr[fit_err_idx][0], i_phi0_arr[fit_err_idx][-1]
    i_phi0_best = i_phi0_arr[min_idx]
    i_phi0_err = 0.5*(i_phi0_max-i_phi0_min)

    # ===========================================================
    # FIT TOROIDAL MAGNETIC FIELD
    # ===========================================================
    args = I_rho0_arr
    worker = partial(
        calc_toroidal,
        n_time_arr=time_arr.size,
        rx_pc=rx_pc,
        ry_pc=ry_pc,
        rz_pc=rz_pc,
        r_pc=r_pc,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        dBx=dBx,
        dBy=dBy,
        dBz=dBz,
        dBr=dBr,
        dBtheta=dBtheta,
        dBphi=dBphi,
        MLT=JunoMLT,
        PJ_num=PJ_num,
        a0=a0,
        MLT0=MLT0,
        b1=b1_arr,
    )
    with Pool(processes=parallel) as pool:
        results = pool.map(worker, args)
    RMS_toroidal_cs = np.array(results)
    min_idx = np.argmin(RMS_toroidal_cs)
    fit_err_idx = np.where(
        abs(RMS_toroidal_cs-np.min(RMS_toroidal_cs))
        < 0.05*np.min(RMS_toroidal_cs)
    )
    I_rho0_min, I_rho0_max = I_rho0_arr[fit_err_idx][0], I_rho0_arr[fit_err_idx][-1]
    I_rho0_best = I_rho0_arr[min_idx]
    I_rho0_err = 0.5*(I_rho0_max-I_rho0_min)
    print('i_phi0 best fit:', i_phi0_min, i_phi0_best, i_phi0_max)
    print('I_rho0 best fit:', I_rho0_min, I_rho0_best, I_rho0_max)
    print(f'RMS (poloidal): {np.min(RMS_poloidal_cs):.4f}')
    print(f'RMS (toroidal): {np.min(RMS_toroidal_cs):.4f}')

    # ===========================================================
    # BEST FIT MAGNETIC FIELD & PLOT
    # ===========================================================
    csfield = CSField()
    csfield.config(
        I_rho0=I_rho0_best,
        i_phi0=i_phi0_best,
        D=2.5,
        a0=a0,
        MLT0=MLT0,
        b1=b1_arr,
    )

    dBx2 = np.zeros(time_arr.size)
    dBy2 = np.zeros(time_arr.size)
    dBz2 = np.zeros(time_arr.size)
    for i in range(time_arr.size):
        dBx2[i], dBy2[i], dBz2[i] = csfield.magnetic_field(rx_pc[i],
                                                           ry_pc[i],
                                                           rz_pc[i],
                                                           JunoMLT[i])  # [T]
    dBx2 *= 1E+9    # [nT]
    dBy2 *= 1E+9    # [nT]
    dBz2 *= 1E+9    # [nT]
    dBr2 = dBx2*sin_theta*cos_phi + dBy2 * \
        sin_theta*sin_phi + dBz2*cos_theta
    dBtheta2 = dBx2*cos_theta*cos_phi + \
        dBy2*cos_theta*sin_phi - dBz2*sin_theta
    dBphi2 = -dBx2*sin_phi + dBy2*cos_phi
    dB2 = np.sqrt(dBx2**2+dBy2**2+dBz2**2)

    # ===========================================================
    # Sys3 (x, y, z) -> CS (rho, phi, z)
    # ===========================================================
    x_cs, y_cs, z_cs = csfield._sys3_2_cs(rx_pc, ry_pc, rz_pc)
    phi_cs = np.arctan2(y_cs, x_cs)
    rho_cs = np.sqrt(x_cs**2 + y_cs**2)
    dBx_cs, dBy_cs, dBz_cs = csfield._sys3_2_cs(dBx, dBy, dBz)
    dBrho_cs = dBx_cs*np.cos(phi_cs) + dBy_cs*np.sin(phi_cs)
    dBphi_cs = -dBx_cs*np.sin(phi_cs) + dBy_cs*np.cos(phi_cs)

    dBx2_cs, dBy2_cs, dBz2_cs = csfield._sys3_2_cs(dBx2, dBy2, dBz2)
    dBrho2_cs = dBx2_cs*np.cos(phi_cs) + dBy2_cs*np.sin(phi_cs)
    dBphi2_cs = -dBx2_cs*np.sin(phi_cs) + dBy2_cs*np.cos(phi_cs)
    dB2_cs = np.sqrt(dBrho2_cs**2+dBphi2_cs**2+dBz2_cs**2)

    plot_best(time_arr, r_pc, rho_cs, z_cs, dBr,
              dBtheta, dBphi, dBr2, dBtheta2, dBphi2,
              dBphi_cs, dBphi2_cs, JunoMLT)
    # print(dBphi2_cs[np.where((rho_cs > 5.0) & (rho_cs < 7.0))])
    # print('dB2    [nT]:', dB2[np.where((rho_cs > 5.0) & (rho_cs < 7.0))])
    # print('dB2_cs [nT]:', dB2_cs[np.where((rho_cs > 5.0) & (rho_cs < 7.0))])

    # ===========================================================
    # RMSE（Root Mean Squared Error）
    # ===========================================================
    in_30rj = np.where(r_pc < 30.0)[0][0]
    if PJ_num in [17]:
        in_5rj = np.where(rho_cs < 8.0)[0][0]
    else:
        in_5rj = np.where(rho_cs < 5.0)[0][0]
    in_num = in_5rj-in_30rj-1
    out_30rj = np.where(r_pc < 30.0)[0][-1]
    if PJ_num in [17]:
        out_5rj = np.where(rho_cs < 8.0)[0][-1]
    else:
        out_5rj = np.where(rho_cs < 5.0)[0][-1]
    out_num = out_30rj-out_5rj-1
    RMS_total = math.sqrt(
        (
            np.sum(
                (dBr[in_30rj:in_5rj]-dBr2[in_30rj:in_5rj])**2 +
                (dBtheta[in_30rj:in_5rj]-dBtheta2[in_30rj:in_5rj])**2 +
                (dBphi[in_30rj:in_5rj]-dBphi2[in_30rj:in_5rj])**2
            ) + np.sum(
                (dBr[out_5rj:out_30rj]-dBr2[out_5rj:out_30rj])**2 +
                (dBtheta[out_5rj:out_30rj]-dBtheta2[out_5rj:out_30rj])**2 +
                (dBphi[out_5rj:out_30rj]-dBphi2[out_5rj:out_30rj])**2
            )
        )/(3*in_num+3*out_num))
    print('RMS total:', RMS_total)
    print('Time [sec]:', round(time.time()-start, 2))

    # JunoMLT 代表値
    in_30rj = np.where(r_pc < 30.0)[0][0]
    in_5rj = np.where(rho_cs < 5.0)[0][0]
    JunoMLT_1 = np.median(JunoMLT[in_30rj:in_5rj])
    return np.array([PJ_num, i_phi0_best, i_phi0_err, I_rho0_best, I_rho0_err, RMS_total, JunoMLT_1])


if __name__ == '__main__':
    PJ_list = [1, 3, 4, 5,
               6, 7, 8, 9, 10,
               11, 12, 13, 14, 15,
               16, 17, 18, 19, 20,
               21, 22, 23, 24, 25,
               26, 27, 28, 29, 30,
               31, 32, 33, 34, 35,
               36, 37, 38, 39, 40,
               41, 42, 43, 44, 45,
               46, 47, 48, 49, 50,
               51, 52, 53, 54, 55,
               56, 57, 58, 59, 60,
               61, 62, 63, 64, 65,
               66, 67, 68, 69, 70,
               71, 72, 73, 74, 75,
               76, 77, 78, 79]
    parallel = 5

    a0 = 0.0
    MLT0 = 0.0
    b1_arr = 0.0    # 3.64 [nT] (Provan+2024)
    i_phi0_arr = 5.2E-5*np.linspace(0.75, 1.3, 40)
    s = 16
    I_rho0_arr = s*np.sinh(
        np.linspace(np.arcsinh(-30/s), np.arcsinh(45/s), 46)
    )
    # I_rho0_arr *= 0.00001
    print('i_phi0 [10^-5]:', i_phi0_arr[0]*1E+5, i_phi0_arr[-1]*1E+5)
    print('I_rho0:', I_rho0_arr[0], I_rho0_arr[-1])

    savedir = 'results/insitu_fit/result_'
    save_arr = np.zeros((len(PJ_list), 7))
    start = time.time()
    for i in range(len(PJ_list)):
        PJ_num = PJ_list[i]
        print('PJ:', PJ_num)
        save_arr[i, :] = main()
    print('Total time [sec]:', round(time.time()-start, 2))

    print(save_arr)
    print('RMS min, average:',
          np.min(save_arr[:, 5]),
          np.average(save_arr[:, 5]))

    fname = savedir + 'PJ'+str(PJ_list[0]).zfill(2)+'_'
    fname += 'PJ'+str(PJ_list[-1]).zfill(2)
    fname += '.txt'
    np.savetxt(fname, save_arr)
