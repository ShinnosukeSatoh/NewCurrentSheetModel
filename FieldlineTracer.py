""" FieldlineTracer.py

Created on Nov 18, 2026
@author: Shin Satoh

Description:
    Magnetic field line tracer for Jupiter's magnetosphere.
    Combines JRM33 internal field (via JupiterMag) and
    CSField current sheet field.

    Coordinate system: SIII right-hand (cartesian), unit = RJ

Usage:
    import JupiterMag as jm
    from CSField import CSField
    from FieldlineTracer import FieldlineTracer

    cs = CSField()
    cs.config()

    jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)

    tracer = FieldlineTracer(cs_field=cs)

    # 楕円体で止める（デフォルト）
    result = tracer.trace(x0=10.0, y0=0.0, z0=0.0, MLT=0.0)

    # 半径 1.0 RJ の球で止める
    result = tracer.trace(x0=10.0, y0=0.0, z0=0.0, MLT=0.0,
                          planet_shape='sphere', r_planet=1.0)

    # 電離圏高度（500 km）の楕円体で止める
    result = tracer.trace(x0=10.0, y0=0.0, z0=0.0, MLT=0.0,
                          planet_shape='ellipsoid', h_stop_km=500.0)

    print(result['x'], result['y'], result['z'])
    print(result['stop_fwd'], result['stop_bwd'])

Version
0.1.0 (Nov 18, 2026)
0.2.0 (Nov 18, 2026) : planet_shape パラメータ追加（楕円体/球の切り替え）
"""

import numpy as np
import JupiterMag as jm
from scipy.integrate import solve_ivp
from CSField import CSField


# ======================================================
# CONSTANTS
# ======================================================
REQ_RJ = 1.0                    # 木星赤道半径 [RJ]
RPOL_RJ = 66854.0 / 71492.0      # 木星極半径  [RJ] ≈ 0.9353
RJ_KM = 71492.0                # 1 RJ in km


# ======================================================
# FIELDLINE TRACER
# ======================================================
class FieldlineTracer:
    def __init__(
            self,
            cs_field: CSField,
    ):
        """
        Parameters
        ----------
        cs_field : CSField
            config() 済みの CSField インスタンス。
            JRM33 は JupiterMag.Internal をグローバルに使うため、
            呼び出し前に以下を済ませておくこと:
                jm.Internal.Config(Model='jrm33',
                                   CartesianIn=True, CartesianOut=True)
        """
        self.cs = cs_field

    # --------------------------------------------------
    # 合成磁場
    # --------------------------------------------------
    def _total_B_nT(self, x, y, z, MLT):
        """
        JRM33 (内部) + CSField (電流シート) の合成磁場を返す。

        Parameters
        ----------
        x, y, z : float  SIII右手系 [RJ]
        MLT     : float  磁気地方時 [hr]

        Returns
        -------
        Bx, By, Bz : float  [nT]
        """
        # --- JRM33 ---
        # CartesianIn=True, CartesianOut=True で設定済みを前提とする
        # Field() は [RJ] を受け取り [nT] を返す
        Bx_jrm, By_jrm, Bz_jrm = jm.Internal.Field(
            np.array([x]),
            np.array([y]),
            np.array([z]),
        )
        Bx_jrm = float(Bx_jrm)
        By_jrm = float(By_jrm)
        Bz_jrm = float(Bz_jrm)

        # --- CSField (電流シート) ---
        # magnetic_field_v2 は [T] を返すので nT へ変換
        Bx_cs, By_cs, Bz_cs = self.cs.magnetic_field_v2(
            x, y, z, MLT=MLT, output_coords='cartesian'
        )
        Bx_cs *= 1.0e9
        By_cs *= 1.0e9
        Bz_cs *= 1.0e9

        return (Bx_jrm + Bx_cs,
                By_jrm + By_cs,
                Bz_jrm + Bz_cs)

    # --------------------------------------------------
    # ODE 右辺（弧長 s でパラメータ化）
    # --------------------------------------------------
    def _rhs(self, s, pos, MLT, sign):
        """
        ds/ds = sign * B̂  （弧長パラメータ積分）

        sign = +1 : B方向にトレース
        sign = -1 : B逆方向にトレース
        """
        x, y, z = pos
        Bx, By, Bz = self._total_B_nT(x, y, z, MLT)
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

        if B_mag == 0.0:
            return [0.0, 0.0, 0.0]

        return [sign * Bx / B_mag,
                sign * By / B_mag,
                sign * Bz / B_mag]

    # --------------------------------------------------
    # イベント関数
    # --------------------------------------------------
    @staticmethod
    def _make_event_ellipsoid(h_stop_RJ):
        """
        楕円体表面（+h_stop_RJ）への到達を検出するイベント関数を返す。

        f(x,y,z) = x²/a² + y²/a² + z²/c² - 1
          > 0 : 楕円体外側
          < 0 : 楕円体内側
        terminal かつ 外→内 方向のみ検出。

        Parameters
        ----------
        h_stop_RJ : float  表面からの停止高度 [RJ]
        """
        a = REQ_RJ + h_stop_RJ   # 赤道方向の停止半径 [RJ]
        c = RPOL_RJ + h_stop_RJ   # 極方向の停止半径   [RJ]

        def event_ellipsoid(s, pos, MLT, sign):
            x, y, z = pos
            return x**2 / a**2 + y**2 / a**2 + z**2 / c**2 - 1.0

        event_ellipsoid.terminal = True
        event_ellipsoid.direction = -1   # 外→内のみ
        return event_ellipsoid

    @staticmethod
    def _make_event_sphere(r_planet_RJ):
        """
        球表面（半径 r_planet_RJ）への到達を検出するイベント関数を返す。

        f(x,y,z) = x² + y² + z² - r²
          > 0 : 球外側
          < 0 : 球内側
        terminal かつ 外→内 方向のみ検出。

        Parameters
        ----------
        r_planet_RJ : float  球の停止半径 [RJ]
        """
        r2 = r_planet_RJ**2

        def event_sphere(s, pos, MLT, sign):
            x, y, z = pos
            return x**2 + y**2 + z**2 - r2

        event_sphere.terminal = True
        event_sphere.direction = -1   # 外→内のみ
        return event_sphere

    @staticmethod
    def _make_event_r_max(r_max):
        """外側境界（球）への到達を検出するイベント関数を返す。"""
        def event_r_max(s, pos, MLT, sign):
            x, y, z = pos
            return x**2 + y**2 + z**2 - r_max**2

        event_r_max.terminal = True
        event_r_max.direction = +1   # 内→外のみ
        return event_r_max

    # --------------------------------------------------
    # 1方向トレース（内部実装）
    # --------------------------------------------------
    def _trace_one_direction(
            self,
            x0, y0, z0,
            MLT,
            sign,
            s_max,
            r_max,
            planet_shape,
            h_stop_RJ,
            r_planet_RJ,
            rtol,
            atol,
    ):
        """
        1方向（sign=±1）にトレースして結果を返す。

        Parameters
        ----------
        planet_shape : str   'ellipsoid' または 'sphere'
        h_stop_RJ   : float  楕円体モード時の停止高度 [RJ]
        r_planet_RJ : float  球モード時の停止半径    [RJ]

        Returns
        -------
        dict with keys: x, y, z, s, stop_reason
        """
        # 内側終端イベントをモードで切り替え
        if planet_shape == 'ellipsoid':
            event_inner = self._make_event_ellipsoid(h_stop_RJ)
        elif planet_shape == 'sphere':
            event_inner = self._make_event_sphere(r_planet_RJ)
        else:
            raise ValueError(
                f"planet_shape は 'ellipsoid' または 'sphere' を指定してください: "
                f"'{planet_shape}'"
            )

        event_r_max = self._make_event_r_max(r_max)

        sol = solve_ivp(
            fun=self._rhs,
            t_span=(0.0, s_max),
            y0=[x0, y0, z0],
            method='RK45',
            args=(MLT, sign),
            events=[event_inner, event_r_max],
            dense_output=False,
            rtol=rtol,
            atol=atol,
        )

        # 停止理由の判定
        if sol.t_events[0].size > 0:
            stop_reason = 'surface'   # 楕円体/球どちらのモードでも 'surface'
        elif sol.t_events[1].size > 0:
            stop_reason = 'r_max'
        else:
            stop_reason = 'max_arc'

        return {
            'x': sol.y[0],
            'y': sol.y[1],
            'z': sol.y[2],
            's': sol.t,
            'stop_reason': stop_reason,
        }

    # --------------------------------------------------
    # 公開インターフェース
    # --------------------------------------------------
    def trace(
            self,
            x0,
            y0,
            z0,
            MLT=0.0,
            direction='both',
            planet_shape='ellipsoid',
            h_stop_km=0.0,
            r_planet=1.0,
            s_max=500.0,
            r_max=150.0,
            rtol=1e-6,
            atol=1e-6,
    ):
        """
        磁力線をトレースする。

        Parameters
        ----------
        x0, y0, z0    : float
            開始点 [RJ]、SIII右手座標系
        MLT           : float
            磁気地方時 [hr]
        direction     : str
            '+'    : B方向のみ
            '-'    : B逆方向のみ
            'both' : 両方向（デフォルト）
        planet_shape  : str
            'ellipsoid' : 木星を楕円体として扱う（デフォルト）
                          赤道半径 1.0 RJ、極半径 ≈ 0.9353 RJ
            'sphere'    : 任意半径の球として扱う
                          半径は r_planet で指定
        h_stop_km     : float
            [ellipsoid モード専用]
            楕円体表面からの停止高度 [km]（デフォルト 0 = 表面）
            例: 電離圏研究では 200〜500 km 程度
        r_planet      : float
            [sphere モード専用]
            球の停止半径 [RJ]（デフォルト 1.0 RJ = 木星表面相当）
            例: 衛星軌道や任意の磁気圏境界を指定する場合に使う
        s_max         : float
            片方向の最大弧長 [RJ]（デフォルト 500 RJ）
        r_max         : float
            外側終端半径 [RJ]（デフォルト 150 RJ）
        rtol, atol    : float
            RK45 の相対・絶対許容誤差（デフォルト 1e-6）

        Returns
        -------
        dict
            x, y, z    : np.ndarray  SIII右手系 [RJ]
            s          : np.ndarray  弧長 [RJ]（開始点が s=0）
            B_mag      : np.ndarray  各点の |B| [nT]
            stop_fwd   : str または None  +方向の停止理由
            stop_bwd   : str または None  -方向の停止理由
                'surface'  : 惑星表面（楕円体または球）に到達
                'r_max'    : 外側境界に到達
                'max_arc'  : 最大弧長に到達
        """
        h_stop_RJ = h_stop_km / RJ_KM
        r_planet_RJ = r_planet          # すでに [RJ] 単位

        results = {}

        # +方向
        if direction in ('+', 'both'):
            fwd = self._trace_one_direction(
                x0, y0, z0, MLT,
                sign=+1,
                s_max=s_max,
                r_max=r_max,
                planet_shape=planet_shape,
                h_stop_RJ=h_stop_RJ,
                r_planet_RJ=r_planet_RJ,
                rtol=rtol,
                atol=atol,
            )
            results['fwd'] = fwd

        # -方向
        if direction in ('-', 'both'):
            bwd = self._trace_one_direction(
                x0, y0, z0, MLT,
                sign=-1,
                s_max=s_max,
                r_max=r_max,
                planet_shape=planet_shape,
                h_stop_RJ=h_stop_RJ,
                r_planet_RJ=r_planet_RJ,
                rtol=rtol,
                atol=atol,
            )
            results['bwd'] = bwd

        # 結合（both の場合は bwd を逆順にして fwd の手前に繋げる）
        if direction == 'both':
            # bwd は開始点から逆方向: 逆順にして先頭に
            x = np.concatenate([results['bwd']['x'][::-1],
                                results['fwd']['x'][1:]])
            y = np.concatenate([results['bwd']['y'][::-1],
                                results['fwd']['y'][1:]])
            z = np.concatenate([results['bwd']['z'][::-1],
                                results['fwd']['z'][1:]])
            # 弧長は bwd を負にして連結
            s_bwd = -results['bwd']['s'][::-1]
            s_fwd = results['fwd']['s'][1:]
            s = np.concatenate([s_bwd, s_fwd])
            stop_fwd = results['fwd']['stop_reason']
            stop_bwd = results['bwd']['stop_reason']

        elif direction == '+':
            x = results['fwd']['x']
            y = results['fwd']['y']
            z = results['fwd']['z']
            s = results['fwd']['s']
            stop_fwd = results['fwd']['stop_reason']
            stop_bwd = None

        else:  # '-'
            x = results['bwd']['x']
            y = results['bwd']['y']
            z = results['bwd']['z']
            s = results['bwd']['s']
            stop_fwd = None
            stop_bwd = results['bwd']['stop_reason']

        # 各点の |B| を計算
        B_mag = np.array([
            np.sqrt(sum(b**2 for b in self._total_B_nT(xi, yi, zi, MLT)))
            for xi, yi, zi in zip(x, y, z)
        ])

        return {
            'x':        x,
            'y':        y,
            'z':        z,
            's':        s,
            'B_mag':    B_mag,
            'stop_fwd': stop_fwd,
            'stop_bwd': stop_bwd,
        }
