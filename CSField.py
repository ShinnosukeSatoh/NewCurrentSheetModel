""" CSField.py

Created on Sep 25, 2026
@author: Shin Satoh

Description:


Version
0.0.1 (Sep 25, 2026)

"""
import numpy as np
import math

# ======================================================
# CONSTANTS
# ======================================================
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # [kg]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
OMGJ = 1.75868E-4        # JUPITER SPIN ANGULAR VELOCITY [rad/s]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]
phiRH0 = math.radians(-65.8)      # [rad]     Connerney+2020
TILT0 = math.radians(6.7)         # [rad]
Ai_H = 1.0               # 水素 [原子量]
Ai_O = 16.0              # 酸素 [原子量]
Ai_S = 32.0              # 硫黄 [原子量]


class CSField():
    def __init__(self) -> None:
        pass

    def calc_Bvec(self, x, y, z):
        I_phi = 1.0    # [nT]
        c = 1.0
        d = 1.0
        D = 1.0*RJ     # [m]
        rho = math.sqrt(x**2+y**2+z**2)     # [m]

        sgn = z/abs(z)
        if abs(z) <= D:
            f1 = (c-z+D)/math.sqrt(rho**2+(c-z+D)**2) + (c+z+D) / \
                math.sqrt(rho**2+(c+z+D)**2) - 2*c/math.sqrt(rho**2+c**2)
            f2 = (d-z+D)/math.sqrt(rho**2+(d-z+D)**2) + (d+z+D) / \
                math.sqrt(rho**2+(d+z+D)**2) - 2*d/math.sqrt(rho**2+d**2)
            B_rho = sgn*(MU0*I_phi/(4*rho*D))*(f1-f2)

            f3 = 2/math.sqrt(rho**2+c**2) - 1/math.sqrt(rho **
                                                        2+(c-z+D)**2) - 1/math.sqrt(rho**2+(c+z+D)**2)
            f3 = 2/math.sqrt(rho**2+c**2) - 1/math.sqrt(rho **
                                                        2+(c-z+D)**2) - 1/math.sqrt(rho**2+(c+z+D)**2)

        return 0
