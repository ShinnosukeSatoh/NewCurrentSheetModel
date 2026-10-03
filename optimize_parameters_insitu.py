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
def B_internal(time_arr, rx_pc, ry_pc, rz_pc):
    Bx0_pc = np.zeros(time_arr.size)
    By0_pc = np.zeros(time_arr.size)
    Bz0_pc = np.zeros(time_arr.size)
    for i in range(time_arr.size):
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
    Bx0_pc, By0_pc, Bz0_pc = B_internal(time_arr, rx_pc, ry_pc, rz_pc)
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
        time_arr, r_pc,
        dBr, dBtheta, dBphi,
        dBr2, dBtheta2, dBphi2,
):
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
    F.fontsize = 19
    F.fontname = 'Liberation Sans Narrow'

    F.set_figparams(nrows=3, figsize=(10.0, 7.5), dpi='XL')
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
    for i in range(3):
        F.ax[i].xaxis.set_minor_locator(FixedLocator(r_ticks[:, 1]))
    for i in range(2):
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
                min=-100, max=20,
                ticks=np.linspace(-100, 20, 7),
                ticklabels=np.linspace(-100, 20, 7, dtype=int),
                minor_num=4)
    F.set_yaxis(ax_idx=2,
                label=r'$\delta B_{\varphi}$ [nT]',
                min=-20, max=20,
                ticks=np.linspace(-20, 20, 5),
                ticklabels=np.linspace(-20, 20, 5, dtype=int),
                minor_num=4)
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

    # Shades near Jupiter (< 5.0 RJ)
    inner_idx = np.where((r_pc < 5.0))[0]
    for i in range(F.nrows):
        F.ax[i].axvspan(time_arr[inner_idx[0]],
                        time_arr[inner_idx[-1]],
                        fc=UC.lightgray, ec=None, zorder=2.5)

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

    legend = F.legend(ax_idx=1,
                      ncol=1, markerscale=1.0,
                      loc='lower right',
                      handlelength=0.7,
                      textcolor=False,
                      fontsize_scale=0.85,
                      handletextpad=0.4)
    legend_shadow(legend=legend, fig=F.fig, ax=F.ax[0], d=0.7)

    plt.savefig('img/insitu_fit/PJ'+str(PJ_num).zfill(2)+'.png',
                bbox_inches='tight')
    return None


# ===========================================================
# CALCULATE THE EXTERNAL FIELD WITH FASTER
# ===========================================================
def calc(
        params,
        n_time_arr,
        rx_pc, ry_pc, rz_pc, r_pc,
        cos_theta, sin_theta, cos_phi, sin_phi,
        dBr, dBtheta, dBphi, MLT):

    # ===========================================================
    # CONFIGURE CSFIELD
    # ===========================================================
    I_phi0, I_rho = params
    csfield = CSField()

    dBx2_pc = np.zeros(n_time_arr)
    dBy2_pc = np.zeros(n_time_arr)
    dBz2_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        csfield.config(
            I_rho=I_rho,
            I_phi=I_phi0,
            D=2.5,
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
    in_30rj = np.where(r_pc < 30.0)[0][0]
    in_5rj = np.where(r_pc < 5.0)[0][0]
    in_num = in_5rj-in_30rj-1
    out_30rj = np.where(r_pc < 30.0)[0][-1]
    out_5rj = np.where(r_pc < 5.0)[0][-1]
    out_num = out_30rj-out_5rj-1

    RMS = math.sqrt(
        (np.sum(
            (dBr[in_30rj:in_5rj]-dBr2_pc[in_30rj:in_5rj])**2 +
            (dBtheta[in_30rj:in_5rj]-dBtheta2_pc[in_30rj:in_5rj])**2 +
            (dBphi[in_30rj:in_5rj]-dBphi2_pc[in_30rj:in_5rj])**2
        ) + np.sum(
            (dBr[out_5rj:out_30rj]-dBr2_pc[out_5rj:out_30rj])**2 +
            (dBtheta[out_5rj:out_30rj]-dBtheta2_pc[out_5rj:out_30rj])**2 +
            (dBphi[out_5rj:out_30rj]-dBphi2_pc[out_5rj:out_30rj])**2
        ))/(3*in_num+3*out_num))

    # print('RMS [nT]:', RMS)

    return RMS


def main():
    date_list = ['2017DOY137', '2017DOY138',
                 '2017DOY139', '2017DOY140',
                 '2017DOY141',]

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

    # Create arg mesh
    a0_mesh, a1_mesh = np.meshgrid(I_phi0_arr, I_rho_arr)
    # -> shape is like (a1.size, a0.size, a2.size)

    args = list(zip(
        a0_mesh.ravel(),
        a1_mesh.ravel(),
    ))

    # 観測データを固定した関数を作成
    worker = partial(
        calc,
        n_time_arr=time_arr.size,
        rx_pc=rx_pc,
        ry_pc=ry_pc,
        rz_pc=rz_pc,
        r_pc=r_pc,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        cos_phi=cos_phi,
        sin_phi=sin_phi,
        dBr=dBr,
        dBtheta=dBtheta,
        dBphi=dBphi,
        MLT=JunoMLT
    )
    start = time.time()
    with Pool(processes=parallel) as pool:
        results = pool.map(worker, args)
    print('Time [sec]:', round(time.time()-start, 2))

    # 最小 RMS とそのインデックスを抽出
    min_idx = np.argmin(results)
    best_a, best_b = args[min_idx]
    min_rms = results[min_idx]

    print('Best fit:', best_a, best_b)
    print(f'RMS: {min_rms:.4f}')

    csfield = CSField()
    csfield.config(
        I_rho=best_b,
        I_phi=best_a,
        D=2.5,
    )

    dBx2 = np.zeros(time_arr.size)
    dBy2 = np.zeros(time_arr.size)
    dBz2 = np.zeros(time_arr.size)
    for i in range(time_arr.size):
        dBx2[i], dBy2[i], dBz2[i] = csfield.magnetic_field(rx_pc[i],
                                                           ry_pc[i],
                                                           rz_pc[i])  # [T]
    dBx2 *= 1E+9    # [nT]
    dBy2 *= 1E+9    # [nT]
    dBz2 *= 1E+9    # [nT]
    dBr2 = dBx2*sin_theta*cos_phi + dBy2 * \
        sin_theta*sin_phi + dBz2*cos_theta
    dBtheta2 = dBx2*cos_theta*cos_phi + \
        dBy2*cos_theta*sin_phi - dBz2*sin_theta
    dBphi2 = -dBx2*sin_phi + dBy2*cos_phi

    plot_best(time_arr, r_pc, dBr,
              dBtheta, dBphi, dBr2, dBtheta2, dBphi2)

    return None


if __name__ == '__main__':
    PJ_num = 12
    parallel = 4

    I_phi0_arr = 8.0E-5*np.linspace(0.75, 1.25, 20)
    I_rho_arr = 16.7*np.linspace(0.5, 1.6, 16)

    main()
