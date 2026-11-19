"""
lag_correlation.py
==================
HisakiのM_dot観測データをdtだけ時間シフトして、
JunoペリジョーブごとのI_rho0との相関を最大にするdt（時間遅れ）を探す。

【物理的な意味】
  Hisakiがイオ軌道(5-8 RJ)でM_dotの変化を観測してから、
  動径輸送によって~13-16 RJで共回転遅れが増幅され、
  JunoのI_rho0に現れるまでの時間遅れ = best dt

【dtシフトの定義】
  shifted_date = original_hisaki_date + dt  (Hisakiの日付をdt日進める)
  → 各PJの近木点日に対して、dt日前のHisakiデータが対応する
  → dtが大きいほど「HisakiがJunoより長く先行している」

【入力データ形式】
  hisaki_dict: {'2017-06-30': value, '2017-07-01': NaN, ...}
               (欠損はNaN or None, 24時間平均値を想定)
  pj_df      : pd.DataFrame with columns ['pj', 'date', 'I_rho0']
               date は '2017-08-27' 形式の近木点通過日

【使い方】
  import lag_correlation as lc
  hisaki_series = lc.make_hisaki_series(hisaki_dict)
  result_df = lc.scan_lag(hisaki_series, pj_df, dt_min=0, dt_max=200)
  best = lc.find_best_dt(result_df)
  fig1 = lc.plot_lag_scan(result_df, best_dt=best['dt'])
  fig2 = lc.plot_scatter_at_best_dt(hisaki_series, pj_df, int(best['dt']))
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================
# Step 1: データ準備
# ============================================================

def make_hisaki_series(dates, values=None):
    """
    以下の呼び出し形式をすべてサポートする:

      (A) dict形式（旧来）
          make_hisaki_series({'2017-06-30': 1.2, '2017-07-01': np.nan, ...})

      (B) 日付列 + 値列を直接渡す（CSVから読んだ場合に便利）
          make_hisaki_series(df['datetime'], df['R_comp_dusk'])
          make_hisaki_series(datetime_array, values_array)

    Args:
        dates  : dict / pd.Series / array-like of date strings or Timestamps
        values : array-like of float, or None（dict渡しのときはNone）

    Returns:
        pd.Series (index=DatetimeIndex, sorted, NaN保持)
    """
    if values is None:
        # (A) dict
        s = pd.Series(dates, dtype=float)
    else:
        # (B) 日付列 + 値列
        s = pd.Series(values, index=dates, dtype=float)

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


# ============================================================
# Step 2: 窓平均
# ============================================================

def window_mean(hisaki_series, center_date, half_window_days=3, min_pts=2):
    """
    center_date ± half_window_days の範囲でHisakiデータを平均する。

    Args:
        hisaki_series    : pd.Series (DatetimeIndex)
        center_date      : str or pd.Timestamp  (PJの近木点日)
        half_window_days : int  (デフォルト3 → ±3日 = 6日窓)
        min_pts          : int  (窓内に必要な最小データ点数。未満ならNaN返す)

    Returns:
        mean  : float  (有効データがmin_pts未満の場合はNaN)
        n     : int    (窓内の有効データ点数)
    """
    # str / datetime.datetime / pd.Timestamp いずれも受け取れる
    center = pd.Timestamp(center_date)
    t_start = center - pd.Timedelta(days=half_window_days)
    t_end = center + pd.Timedelta(days=half_window_days)

    mask = (hisaki_series.index >= t_start) & (hisaki_series.index <= t_end)
    window_data = hisaki_series[mask].dropna()

    if len(window_data) < min_pts:
        return np.nan, len(window_data)
    return window_data.mean(), len(window_data)


# ============================================================
# Step 3: 1つのdtで相関計算
# ============================================================

def compute_correlation_at_dt(
        hisaki_series,
        pj_df,
        dt_days,
        half_window_days=3,
        min_pts=2,
        min_valid_pj=5,
):
    """
    HisakiデータをdtだけシフトしてI_rho0との相関を計算する。

    Args:
        hisaki_series  : pd.Series (DatetimeIndex, 生のHisakiデータ)
        pj_df          : pd.DataFrame columns=['pj', 'date', 'I_rho0']
        dt_days        : int  (正=Hisakiがdt日先行)
        half_window_days : int
        min_pts        : int  (窓内に必要な最小Hisakiデータ点数)
        min_valid_pj   : 相関計算に必要な最小有効PJ数

    Returns:
        r            : float  Pearsonの相関係数 (不足時はNaN)
        p            : float  p値
        n_valid      : int    有効PJ数
        hisaki_vals  : np.array  各PJに対応するHisaki窓平均値
    """
    # Hisakiの日付をdt日進める
    shifted = hisaki_series.copy()
    shifted.index = hisaki_series.index + pd.Timedelta(days=dt_days)

    hisaki_vals = []
    irho_vals = []

    for _, row in pj_df.iterrows():
        h_mean, _ = window_mean(
            shifted, row['date'], half_window_days, min_pts)
        hisaki_vals.append(h_mean)
        irho_vals.append(row['I_rho0'])

    hisaki_vals = np.array(hisaki_vals, dtype=float)
    irho_vals = np.array(irho_vals,   dtype=float)

    # 両方有効な点のみ使用
    valid = np.isfinite(hisaki_vals) & np.isfinite(irho_vals)
    n_valid = int(valid.sum())

    if n_valid < min_valid_pj:
        return np.nan, np.nan, n_valid, hisaki_vals

    r, p = stats.pearsonr(hisaki_vals[valid], irho_vals[valid])
    return r, p, n_valid, hisaki_vals


# ============================================================
# Step 4: dtスキャン
# ============================================================

def scan_lag(
        hisaki_series,
        pj_df,
        dt_min=0,
        dt_max=200,
        dt_step=1,
        half_window_days=3,
        min_pts=2,
        min_valid_pj=5,
        verbose=True,
):
    """
    dt_min から dt_max まで dt_step 刻みでスキャンし、
    各dtの相関係数・p値・有効PJ数を返す。

    Args:
        min_pts      : 窓内に必要な最小Hisakiデータ点数（デフォルト2）
        min_valid_pj : 相関計算に必要な最小有効PJ数（デフォルト5）

    Returns:
        result_df : pd.DataFrame  columns=['dt', 'r', 'p', 'n_valid']
    """
    dt_range = range(dt_min, dt_max + 1, dt_step)
    records = []

    for dt in dt_range:
        r, p, n, _ = compute_correlation_at_dt(
            hisaki_series, pj_df, dt,
            half_window_days=half_window_days,
            min_pts=min_pts,
            min_valid_pj=min_valid_pj,
        )
        records.append({'dt': dt, 'r': r, 'p': p, 'n_valid': n})

    result_df = pd.DataFrame(records)

    if verbose:
        print(f"Scan complete: dt = {dt_min} to {dt_max} days")
        valid_r = result_df.dropna(subset=['r'])
        if len(valid_r) == 0:
            print("  Warning: 有効な相関係数が1つもありません。min_valid_pjを下げるか、データ範囲を確認してください。")
        else:
            best = valid_r.loc[valid_r['r'].idxmax()]
            print(f"  Best dt = {best['dt']:.0f} days  "
                  f"(r={best['r']:.4f}, p={best['p']:.3e}, N={best['n_valid']:.0f})")

    return result_df


# ============================================================
# Step 5: 最良dtの抽出
# ============================================================

def find_best_dt(result_df, p_threshold=0.05):
    """
    相関係数が最大になるdtを返す。
    p_threshold を指定すると統計的有意なdtのみを対象にする。

    Returns:
        pd.Series: best row with keys ['dt', 'r', 'p', 'n_valid']
    """
    df = result_df.dropna(subset=['r'])

    df_sig = df[df['p'] <= p_threshold] if p_threshold else df

    if len(df_sig) == 0:
        print(f"Warning: p < {p_threshold} を満たすdtがありません。フィルタなしで探します。")
        df_sig = df

    best = df_sig.loc[df_sig['r'].idxmax()]
    return best


# ============================================================
# Step 6: プロット
# ============================================================

def plot_lag_scan(result_df, best_dt=None, title='Lag scan: Hisaki vs I_rho0',
                  savefig=None):
    """
    dtスキャン結果を3パネルでプロット。
      Panel 1: Pearson r vs dt
      Panel 2: p-value vs dt (対数スケール)
      Panel 3: 有効PJ数 vs dt
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    dt = result_df['dt']
    r = result_df['r']
    p = result_df['p']
    n = result_df['n_valid']

    # Panel 1: 相関係数
    axes[0].plot(dt, r, 'b-', lw=1.5, label='Pearson r')
    axes[0].fill_between(dt, 0, r, where=(r > 0), alpha=0.15, color='blue')
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    if best_dt is not None:
        axes[0].axvline(best_dt, color='red', lw=1.5, ls='--',
                        label=f'best dt = {best_dt:.0f} d')
    axes[0].set_ylabel('Pearson r', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].set_title(title, fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Panel 2: p値（対数）
    axes[1].semilogy(dt, p, 'g-', lw=1.5)
    axes[1].axhline(0.05, color='red',    lw=1.2, ls='--', label='p = 0.05')
    axes[1].axhline(0.01, color='orange', lw=1.2, ls='--', label='p = 0.01')
    if best_dt is not None:
        axes[1].axvline(best_dt, color='red', lw=1.5, ls='--')
    axes[1].set_ylabel('p-value', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: 有効PJ数
    axes[2].plot(dt, n, 'k-', lw=1.5)
    axes[2].fill_between(dt, 0, n, alpha=0.15, color='black')
    if best_dt is not None:
        axes[2].axvline(best_dt, color='red', lw=1.5, ls='--')
    axes[2].set_ylabel('N valid PJs', fontsize=12)
    axes[2].set_xlabel('Time lag  dt [days]  (Hisaki leads I_rho0 by dt days)',
                       fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches='tight')
        print(f"Saved: {savefig}")
    return fig


def plot_scatter_at_best_dt(
        hisaki_series,
        pj_df,
        best_dt,
        half_window_days=3,
        hisaki_label='Hisaki (window mean)',
        savefig=None,
):
    """
    指定したdtでのHisaki vs I_rho0 散布図。
    点の色 = PJ番号。
    """
    r, p, n, hisaki_vals = compute_correlation_at_dt(
        hisaki_series, pj_df, best_dt, half_window_days
    )

    irho_vals = pj_df['I_rho0'].values
    pj_nums = pj_df['pj'].values
    valid = np.isfinite(hisaki_vals) & np.isfinite(irho_vals)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        hisaki_vals[valid], irho_vals[valid],
        c=pj_nums[valid], cmap='viridis',
        s=70, zorder=3, edgecolors='k', linewidths=0.4
    )
    plt.colorbar(sc, ax=ax, label='PJ number')

    # 回帰直線
    sl, ic, *_ = stats.linregress(hisaki_vals[valid], irho_vals[valid])
    xf = np.linspace(np.nanmin(hisaki_vals[valid]),
                     np.nanmax(hisaki_vals[valid]), 100)
    ax.plot(xf, sl*xf + ic, 'r--', lw=1.5, label=f'y = {sl:.2f}x + {ic:.2f}')

    ax.set_xlabel(f'{hisaki_label}\n(dt = {best_dt} days shift)', fontsize=11)
    ax.set_ylabel(r'$I_{\rho 0}$ [MA]', fontsize=11)
    ax.set_title(f'dt = {best_dt} d  |  r = {r:.3f},  p = {p:.3e},  N = {n}',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches='tight')
        print(f"Saved: {savefig}")
    return fig


# ============================================================
# 動作確認用ダミーデータ
# ============================================================
if __name__ == '__main__':

    np.random.seed(42)
    TRUE_LAG = 60   # 確認用の真のlag

    # --- ダミーHisakiデータ（毎日、30%欠損） ---
    dates_all = pd.date_range('2017-01-01', '2024-12-31', freq='D')
    true_signal = (np.sin(2*np.pi*np.arange(len(dates_all))/365)
                   + 0.3*np.random.randn(len(dates_all)))
    mask_nan = np.random.rand(len(dates_all)) < 0.3
    true_signal[mask_nan] = np.nan
    # (B)形式でも動くことを確認
    hisaki_series = make_hisaki_series(dates_all, true_signal)

    # --- ダミーPJデータ ---
    pj_dates = pd.date_range('2017-08-27', periods=40, freq='53D')
    irho_list = []
    for d in pj_dates:
        past = d - pd.Timedelta(days=TRUE_LAG)
        idx = np.argmin(np.abs(dates_all - past))
        irho_list.append(true_signal[idx] * 3 + 10 + 0.5*np.random.randn())

    pj_df = pd.DataFrame({
        'pj':     range(1, len(pj_dates)+1),
        'date':   [str(d.date()) for d in pj_dates],
        'I_rho0': irho_list,
    })

    # --- スキャン ---
    result_df = scan_lag(hisaki_series, pj_df, dt_min=0, dt_max=200)
    best = find_best_dt(result_df, p_threshold=0.05)
    print(f"\nBest dt = {best['dt']:.0f} days  "
          f"(true lag = {TRUE_LAG} days)")

    # --- プロット ---
    plot_lag_scan(
        result_df, best_dt=best['dt'],
        savefig='/mnt/user-data/outputs/lag_scan.png'
    )
    plot_scatter_at_best_dt(
        hisaki_series, pj_df, int(best['dt']),
        savefig='/mnt/user-data/outputs/scatter_best_dt.png'
    )
