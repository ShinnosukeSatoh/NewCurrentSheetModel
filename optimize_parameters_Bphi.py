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
    inner_idx = np.where((r_pc < np.min(r_ticks_ref)))[0]
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

    F.close()
    plt.close()

    return None


# ===========================================================
# CALCULATE THE EXTERNAL FIELD WITH FASTER
# ===========================================================
def calc(
        params,
        n_time_arr,
        rx_pc, ry_pc, rz_pc, r_pc,
        cos_theta, sin_theta, cos_phi, sin_phi,
        dBr, dBtheta, dBphi, MLT, PJ_num):
    # ===========================================================
    # CONFIGURE CSFIELD
    # ===========================================================
    c11, c12, w11, w12 = params
    # print(params)
    csfield = CSField()

    dBx2_pc = np.zeros(n_time_arr)
    dBy2_pc = np.zeros(n_time_arr)
    dBz2_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        csfield.config(
            D=2.5,
            c11=c11,
            c12=c12,
            w11=w11,
            w12=w12,
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
    RMS = math.sqrt(np.sum((dBphi-dBphi2_pc)**2)/dBphi.size)
    print('RMS [nT] and parms:', RMS, params)

    return RMS


def main():
    r_pc = np.zeros(3)
    rx_pc = np.zeros(3)
    ry_pc = np.zeros(3)
    rz_pc = np.zeros(3)
    time_arr = np.zeros(3)
    Bx_pc = np.zeros(3)
    By_pc = np.zeros(3)
    Bz_pc = np.zeros(3)
    JunoMLT = np.zeros(3)

    for i in range(len(PJ_list)):
        rx_pc_i, ry_pc_i, rz_pc_i, time_arr_i, Bx_pc_i, By_pc_i, Bz_pc_i, JunoMLT_i = load_MAGdata(
            PJ_list[i]
        )
        r_pc_i = np.sqrt(rx_pc_i**2+ry_pc_i**2+rz_pc_i**2)

        in_30rj = np.where(r_pc_i < 30.0)[0][0]
        if PJ_list[i] in [17]:
            in_rj = np.where(r_pc_i < 8.0)[0][0]
        else:
            in_rj = np.where(r_pc_i < 5.0)[0][0]
        out_30rj = np.where(r_pc_i < 30.0)[0][-1]
        if PJ_list[i] in [17]:
            out_rj = np.where(r_pc_i < 8.0)[0][-1]
        else:
            out_rj = np.where(r_pc_i < 5.0)[0][-1]

        r_pc = np.append(
            r_pc,
            np.append(
                r_pc_i[in_30rj:in_rj], r_pc_i[out_rj:out_30rj]
            )
        )
        rx_pc = np.append(
            rx_pc,
            np.append(
                rx_pc_i[in_30rj:in_rj], rx_pc_i[out_rj:out_30rj]
            )
        )
        ry_pc = np.append(
            ry_pc,
            np.append(
                ry_pc_i[in_30rj:in_rj], ry_pc_i[out_rj:out_30rj]
            )
        )
        rz_pc = np.append(
            rz_pc,
            np.append(
                rz_pc_i[in_30rj:in_rj], rz_pc_i[out_rj:out_30rj]
            )
        )
        time_arr = np.append(
            time_arr,
            np.append(
                time_arr_i[in_30rj:in_rj], time_arr_i[out_rj:out_30rj]
            )
        )
        Bx_pc = np.append(
            Bx_pc,
            np.append(
                Bx_pc_i[in_30rj:in_rj], Bx_pc_i[out_rj:out_30rj]
            )
        )
        By_pc = np.append(
            By_pc,
            np.append(
                By_pc_i[in_30rj:in_rj], By_pc_i[out_rj:out_30rj]
            )
        )
        Bz_pc = np.append(
            Bz_pc,
            np.append(
                Bz_pc_i[in_30rj:in_rj], Bz_pc_i[out_rj:out_30rj]
            )
        )
        JunoMLT = np.append(
            JunoMLT,
            np.append(
                JunoMLT_i[in_30rj:in_rj], JunoMLT_i[out_rj:out_30rj]
            )
        )

    r_pc = r_pc[3:]
    rx_pc = rx_pc[3:]
    ry_pc = ry_pc[3:]
    rz_pc = rz_pc[3:]
    time_arr = time_arr[3:]
    Bx_pc = Bx_pc[3:]
    By_pc = By_pc[3:]
    Bz_pc = Bz_pc[3:]
    JunoMLT = JunoMLT[3:]

    cos_phi = rx_pc/r_pc                        # [rad]
    sin_phi = ry_pc/r_pc                        # [rad]
    cos_theta = rz_pc/r_pc
    sin_theta = np.sqrt(rx_pc**2+ry_pc**2)/r_pc

    _, _, _, dBx, dBy, dBz, dBr, dBtheta, dBphi = dB_obs(
        time_arr, rx_pc, ry_pc, rz_pc,
        Bx_pc, By_pc, Bz_pc,
        cos_theta, sin_theta, cos_phi, sin_phi
    )

    #
    #
    # ここから下は手付かず!
    #
    #
    # Create arg mesh
    a0_mesh, a1_mesh, a2_mesh, a3_mesh = np.meshgrid(c11_arr,
                                                     c12_arr,
                                                     w11_arr,
                                                     w12_arr)
    # -> shape is like (a1.size, a0.size, a2.size, a3.size)

    args = list(zip(
        a0_mesh.ravel(),
        a1_mesh.ravel(),
        a2_mesh.ravel(),
        a3_mesh.ravel(),
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
        MLT=JunoMLT,
        PJ_num=0.0,
    )
    start = time.time()
    with Pool(processes=parallel) as pool:
        results = pool.map(worker, args)
    print('Time [sec]:', round(time.time()-start, 2))

    # 最小 RMS とそのインデックスを抽出
    min_idx = np.argmin(results)
    best_a, best_b, best_c, best_d = args[min_idx]
    min_rms = results[min_idx]

    print('Best fit:', best_a, best_b, best_c, best_d,)
    print(f'RMS: {min_rms:.4f}')

    return None


if __name__ == '__main__':
    PJ_list = [1, 3, 4, 5,
               6, 7, 8, 9, 10,
               11, 12, 13, 14, 15,
               16, 17, 18, 19, 20,
               21, 22, 23, 24, 25,
               26, 27, 28, 29, 30,
               32, 33, 34, 35,
               36, 37, 38, 39, 40,
               41, 42, 43, 44, 45,
               46, 48, 49, 50,
               51, 52, 53, 54, 55,
               56, 57, 58, 59, 60]
    parallel = 6

    c11_arr = np.linspace(1.0, 5.0, 6)
    c12_arr = np.linspace(15.0, 30.0, 6)
    w11_arr = np.linspace(-1.0, 1.0, 8)
    w12_arr = np.linspace(-1.0, 1.0, 8)

    main()
