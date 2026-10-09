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
        time_arr, r_pc, z_cs,
        dBr, dBtheta, dBphi,
        dBr2, dBtheta2, dBphi2,
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
    F.fontsize = 19
    F.fontname = 'Liberation Sans Narrow'

    F.set_figparams(nrows=4, figsize=(10.0, 10.5), dpi='XL')
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
    for i in range(4):
        F.ax[i].xaxis.set_minor_locator(FixedLocator(r_ticks[:, 1]))
    for i in range(3):
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
                label=r'$z_{\rm cs}$ [$R_{\rm J}$]',
                min=-30, max=20,
                ticks=np.linspace(-30, 20, 6),
                ticklabels=np.linspace(-30, 20, 6, dtype=int),
                minor_num=5)
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
    F.ax[3].plot(time_arr, z_cs, color='k',
                 linewidth=1.0, label='Juno trajectory', zorder=1.0)

    # z_cs current sheet shade
    rng = np.random.default_rng(42)
    noise = rng.random((500, 1200))
    im = F.ax[3].imshow(
        noise,
        extent=[int(time_arr[0]), int(time_arr[-1])+1, -2.5, 2.5],
        origin="lower",
        cmap="Blues",
        alpha=0.20,
        interpolation="nearest",
        aspect="auto",
    )

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
    i_phi0, I_rho0 = params
    csfield = CSField()

    dBx2_pc = np.zeros(n_time_arr)
    dBy2_pc = np.zeros(n_time_arr)
    dBz2_pc = np.zeros(n_time_arr)
    for i in range(n_time_arr):
        csfield.config(
            I_rho0=I_rho0,
            i_phi0=i_phi0,
            D=2.5,
            # c11=1.0, c12=23.571,
            # w11=0.1667, w12=1.5,
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
    if PJ_num in [17]:
        in_5rj = np.where(r_pc < 8.0)[0][0]
    else:
        in_5rj = np.where(r_pc < 5.0)[0][0]
    in_num = in_5rj-in_30rj-1
    out_30rj = np.where(r_pc < 30.0)[0][-1]
    if PJ_num in [17]:
        out_5rj = np.where(r_pc < 8.0)[0][-1]
    else:
        out_5rj = np.where(r_pc < 5.0)[0][-1]
    out_num = out_30rj-out_5rj-1

    # B_phiもフルで使う
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

    # B_phiのinwardはフィッティングに使わない
    RMS = math.sqrt(
        (np.sum(
            (dBr[in_30rj:in_5rj]-dBr2_pc[in_30rj:in_5rj])**2 +
            (dBtheta[in_30rj:in_5rj]-dBtheta2_pc[in_30rj:in_5rj])**2
        ) + np.sum(
            (dBr[out_5rj:out_30rj]-dBr2_pc[out_5rj:out_30rj])**2 +
            (dBtheta[out_5rj:out_30rj]-dBtheta2_pc[out_5rj:out_30rj])**2 +
            (dBphi[out_5rj:out_30rj]-dBphi2_pc[out_5rj:out_30rj])**2
        ))/(2*in_num+3*out_num))

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
    a0_mesh, a1_mesh = np.meshgrid(i_phi0_arr, I_rho0_arr)
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
        MLT=JunoMLT,
        PJ_num=PJ_num
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
        I_rho0=best_b,
        i_phi0=best_a,
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

    _, _, z_cs, _, _ = csfield._sys3_2_cs(rx_pc, ry_pc, rz_pc)

    plot_best(time_arr, r_pc, z_cs, dBr,
              dBtheta, dBphi, dBr2, dBtheta2, dBphi2)

    return np.array([PJ_num, best_a, best_b, min_rms])


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
               46, 48, 49, 50,
               51, 52, 53, 54, 55,
               56, 57, 58, 59, 60,
               61, 62, 63, 64, 65,
               66, 67, 68]
    PJ_list = [57, 58]
    parallel = 5

    i_phi0_arr = 5.2E-5*np.linspace(0.75, 1.3, 30)
    I_rho0_arr = 16.7*np.linspace(0.15, 2.0, 21)
    s = 10
    I_rho0_arr = s*np.sinh(
        np.linspace(np.arcsinh(-10/s), np.arcsinh(40/s), 27)
    )
    I_rho0_arr = np.array([-30, -20,])
    print('i_phi0 [10^-5]:', i_phi0_arr[0]*1E+5, i_phi0_arr[-1]*1E+5)
    print('I_rho0:', I_rho0_arr[0], I_rho0_arr[-1])

    save_arr = np.zeros((len(PJ_list), 4))
    start = time.time()
    for i in range(len(PJ_list)):
        PJ_num = PJ_list[i]
        print('PJ:', PJ_num)
        save_arr[i, :] = main()
    print('Total time [sec]:', round(time.time()-start, 2))

    print(save_arr)
    print('RMS min, average:',
          np.min(save_arr[:, 3]),
          np.average(save_arr[:, 3]))

    fname = 'results/insitu_fit_Bphi_outonly/result_'
    fname += 'PJ'+str(PJ_list[0]).zfill(2)+'_'
    fname += 'PJ'+str(PJ_list[-1]).zfill(2)
    fname += '.txt'
    np.savetxt(fname, save_arr)
