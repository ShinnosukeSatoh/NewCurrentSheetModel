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
MU0 = 4*np.pi*1E-7       # 真空中の透磁率 [H m-1]
AMU2KG = 1.66E-27        # [kg]
RJ = 71492.0E+3          # JUPITER RADIUS [m]


# ======================================================
# CLASS FOR CALCULATION
# ======================================================
class CSField():
    def __init__(self) -> None:
        pass

    def config(
            self,
            I_rho=16.7,
            I_phi=3.8E-4,
            D=3.6,
            theta_d_deg=9.3,
            phi_d_deg=155.8,
            c1=12.6,
            c2=22.0,
            c3=25.5,
            c4=0.0,
            w1=-0.13,
            w2=2.19,
            w3=-2.09,
            w4=0.0,
    ):
        """
        Args:
            I_rho (float): Radial current intensity [MA]
            I_phi (float): Azimuthal current surface density [MA m-1]
            D (float): Half thickness of the current sheet [RJ]
            theta_d_deg (float): Tilt angle of current sheet normal [deg]
            phi_d_deg (float): Azimuthal angle of the tilt of
                                the current sheet normal sheet tilt
                                (right-hand) [deg]
            c1 (float): c1 < c2 < c3 [RJ]
            c2 (float): c1 < c2 < c3 [RJ]
            c3 (float): c1 < c2 < c3 [RJ]
            c4 (float): [RJ]
            w1 (float): Weight of the Hankel kernel
            w2 (float): Weight of the Hankel kernel
            w3 (float): Weight of the Hankel kernel
            w4 (float): Weight of the Hankel kernel
        """
        self.I_rho = I_rho*1E+6           # Radial current inteisity [A]
        self.I_phi = I_phi*(1E+6)         # Azimuthal current density [A m-1]
        self.D = D                        # Current sheet half thickness [RJ]
        self.theta_d_deg = theta_d_deg    # [deg]
        self.phi_d_deg = phi_d_deg        # [deg]
        self.c1 = c1                      # c1 < c2 < c3 [RJ]
        self.c2 = c2                      # c1 < c2 < c3 [RJ]
        self.c3 = c3                      # c1 < c2 < c3 [RJ]
        self.c4 = c4                      # [RJ]
        self.w1 = w1                      # Weight of the Hankel kernel
        self.w2 = w2                      # Weight of the Hankel kernel
        self.w3 = w3                      # Weight of the Hankel kernel
        self.w4 = w4                      # Weight of the Hankel kernel

    def _sys3_2_cs(
            self,
            x,
            y,
            z,
    ):
        theta_d = np.radians(self.theta_d_deg)
        phi_d = np.radians(self.phi_d_deg)

        R = np.array([
            [math.cos(theta_d)*math.cos(phi_d),
             math.cos(theta_d)*math.sin(phi_d),
             -math.sin(theta_d)],
            [-math.sin(phi_d),
             math.cos(phi_d),
             0.0],
            [math.sin(theta_d)*math.cos(phi_d),
             math.sin(theta_d)*math.sin(phi_d),
             math.cos(theta_d)]
        ])

        pos_sys3 = np.array([x, y, z])
        pos_cs = R @ pos_sys3

        x_cs, y_cs, z_cs = pos_cs[0], pos_cs[1], pos_cs[2]
        rho_cs = math.sqrt(x_cs**2 + y_cs**2)
        phi_cs = math.atan2(y_cs, x_cs)

        return x_cs, y_cs, z_cs, rho_cs, phi_cs

    def _cs_2_sys3(
        self,
        B_rho_cs,
        B_phi_cs,
        B_z_cs,
        rho_cs,
        phi_cs,
    ):
        theta_d = math.radians(self.theta_d_deg)
        phi_d = math.radians(self.phi_d_deg)

        Bx_cs = B_rho_cs * np.cos(phi_cs) - B_phi_cs * np.sin(phi_cs)
        By_cs = B_rho_cs * np.sin(phi_cs) + B_phi_cs * np.cos(phi_cs)
        Bz_cs = B_z_cs

        RT = np.array([
            [math.cos(theta_d)*math.cos(phi_d),
             -math.sin(phi_d),
             math.sin(theta_d)*math.cos(phi_d)],
            [math.cos(theta_d)*math.sin(phi_d),
             math.cos(phi_d),
             math.sin(theta_d)*math.sin(phi_d)],
            [-math.sin(theta_d),
             0.0,
             math.cos(theta_d)]
        ])

        B_cs = np.array([Bx_cs, By_cs, Bz_cs])
        B_sys3 = RT @ B_cs

        return B_sys3[0], B_sys3[1], B_sys3[2]

    def _B_vector_cylindrical(
            self,
            x,
            y,
            z
    ):
        """
        Args:
            x (float): SIII right hand [RJ]
            y (float): SIII right hand [RJ]
            z (float): SIII right hand [RJ]

        Returns:
            tuple: B1 vector (B_rho, B_phi, B_Z) [T]
        """
        I_rho = self.I_rho  # Radial current density [A]
        I_phi = self.I_phi  # Azimuthal current density [A]
        c1 = self.c1        # [RJ]
        c2 = self.c2        # [RJ]
        c3 = self.c3        # [RJ]
        w1 = self.w1        # [RJ]
        w2 = self.w2        # [RJ]
        w3 = self.w3        # [RJ]
        D = self.D          # Half thickness of the current sheet [RJ]

        x_cs, y_cs, z_cs, rho_cs, phi_cs = self._sys3_2_cs(
            x,
            y,
            z,
        )

        # print('rho_cs, c, d:', rho_cs, c, d)

        sgn = np.sign(z_cs)
        # print('sgn:', sgn)
        if abs(z_cs) <= D:
            # print('abs(z_cs) <= D')
            u_c11 = c1+z_cs+D
            u_c12 = c1-z_cs+D
            u_c21 = c2+z_cs+D
            u_c22 = c2-z_cs+D
            u_c31 = c3+z_cs+D
            u_c32 = c3-z_cs+D

            f11 = u_c11/math.sqrt(rho_cs**2+u_c11**2)
            f12 = -u_c12/math.sqrt(rho_cs**2+u_c12**2)
            f21 = u_c21/math.sqrt(rho_cs**2+u_c21**2)
            f22 = -u_c22/math.sqrt(rho_cs**2+u_c22**2)
            f31 = u_c31/math.sqrt(rho_cs**2+u_c31**2)
            f32 = -u_c32/math.sqrt(rho_cs**2+u_c32**2)

            B_rho = w1*(f11+f12)+w2*(f21+f22)+w3*(f31+f32)
            B_rho *= MU0*I_phi/(4*rho_cs*D)       # [T]

            g11 = 2/math.sqrt(rho_cs**2+c1**2)
            g12 = -1/math.sqrt(rho_cs**2+u_c11**2)
            g13 = -1/math.sqrt(rho_cs**2+u_c12**2)
            g21 = 2/math.sqrt(rho_cs**2+c2**2)
            g22 = -1/math.sqrt(rho_cs**2+u_c21**2)
            g23 = -1/math.sqrt(rho_cs**2+u_c22**2)
            g31 = 2/math.sqrt(rho_cs**2+c3**2)
            g32 = -1/math.sqrt(rho_cs**2+u_c31**2)
            g33 = -1/math.sqrt(rho_cs**2+u_c32**2)

            B_Z = w1*(g11+g12+g13)+w2*(g21+g22+g23)+w3*(g31+g32+g33)
            B_Z *= MU0*I_phi/(4*D)                # [T]

            B_phi = -((MU0*I_rho)/(2*np.pi*rho_cs*RJ))*(z_cs/D)  # [T]

        elif abs(z_cs) > D:
            # print('abs(z_cs) > D')
            v_c11 = c1+abs(z_cs)+D
            v_c12 = c1+abs(z_cs)-D
            v_c21 = c2+abs(z_cs)+D
            v_c22 = c2+abs(z_cs)-D
            v_c31 = c3+abs(z_cs)+D
            v_c32 = c3+abs(z_cs)-D

            f11 = v_c11/math.sqrt(rho_cs**2+v_c11**2)
            f12 = -v_c12/math.sqrt(rho_cs**2+v_c12**2)
            f21 = v_c21/math.sqrt(rho_cs**2+v_c21**2)
            f22 = -v_c22/math.sqrt(rho_cs**2+v_c22**2)
            f31 = v_c31/math.sqrt(rho_cs**2+v_c31**2)
            f32 = -v_c32/math.sqrt(rho_cs**2+v_c32**2)

            B_rho = w1*(f11+f12)+w2*(f21+f22)+w3*(f31+f32)
            B_rho *= sgn*(MU0*I_phi/(4*rho_cs*D))  # [T]

            g11 = 1/math.sqrt(rho_cs**2+v_c12**2)
            g12 = -1/math.sqrt(rho_cs**2+v_c11**2)
            g21 = 1/math.sqrt(rho_cs**2+v_c22**2)
            g22 = -1/math.sqrt(rho_cs**2+v_c21**2)
            g31 = 1/math.sqrt(rho_cs**2+v_c32**2)
            g32 = -1/math.sqrt(rho_cs**2+v_c31**2)

            B_Z = w1*(g11+g12)+w2*(g21+g22)+w3*(g31+g32)
            B_Z *= MU0*I_phi/(4*D)  # [T]

            B_phi = -sgn*((MU0*I_rho)/(2*np.pi*rho_cs*RJ))  # [T]

        # B_norm = math.sqrt(B_rho**2+B_phi**2+B_Z**2)
        # print('B_norm [nT]:', B_norm*1E+9)
        # print('B_rho [nT]:', B_rho*1E+9)
        # print('B_phi [nT]:', B_phi*1E+9)
        # print('B_Z [nT]:', B_Z*1E+9)

        B_x, B_y, B_z = self._cs_2_sys3(
            B_rho,
            B_phi,
            B_Z,
            rho_cs,
            phi_cs,
        )

        return B_x, B_y, B_z

    def _convert_coordinates(
            self,
            B_x,
            B_y,
            B_z,
            x,
            y,
            z,
            output_coords,
    ):
        if output_coords == 'cartesian':
            return B_x, B_y, B_z

        elif output_coords == 'cylindrical':
            return self._cartesian_to_cylindrical(
                B_x, B_y, B_z, x, y
            )

        elif output_coords == 'spherical':
            return self._cartesian_to_spherical(
                B_x, B_y, B_z, x, y, z
            )

        else:
            raise ValueError(
                f'Unknown coordinate system: {output_coords}'
            )

    def _cartesian_to_cylindrical(
        self, B_x, B_y, B_z, x, y
    ):
        phi = math.atan2(y, x)

        B_rho = B_x*math.cos(phi)+B_y*math.sin(phi)
        B_phi = -B_x*math.sin(phi)+B_y*math.cos(phi)

        return B_rho, B_phi, B_z

    def _cartesian_to_spherical(
        self, B_x, B_y, B_z, x, y, z
    ):
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z/r)
        phi = math.atan2(y, x)

        B_r = (
            B_x*math.sin(theta)*math.cos(phi)
            + B_y*math.sin(theta)*math.sin(phi)
            + B_z*math.cos(theta)
        )

        B_theta = (
            B_x*math.cos(theta)*math.cos(phi)
            + B_y*math.cos(theta)*math.sin(phi)
            - B_z*math.sin(theta)
        )

        B_phi = (
            - B_x*math.sin(phi)
            + B_y*math.cos(phi)
        )

        return B_r, B_theta, B_phi

    def magnetic_field(
            self,
            x,
            y,
            z,
            output_coords='cartesian'
    ):
        """
        Args:
            x (float): SIII right hand [RJ]
            y (float): SIII right hand [RJ]
            z (float): SIII right hand [RJ]
            output_coords (str): 'cartesian', 'cylindrical', or 'spherical'

        return: [A m-1]
        """
        B_x, B_y, B_z = self._B_vector_cylindrical(x, y, z)

        return self._convert_coordinates(
            B_x, B_y, B_z,
            x, y, z,
            output_coords
        )

    def I_phi_profile(
        self,
        MLT,
        rho_cs,
        each_term=False,
    ):
        """
        Args:
            MLT (float): [hr]
            rho_cs (float): [RJ]

        return: [A m-1]
        """
        f1 = self.I_phi*(rho_cs)/(((rho_cs)**2+(self.c1)**2)**(1.5))
        f2 = self.I_phi*(rho_cs)/(((rho_cs)**2+(self.c2)**2)**(1.5))
        f3 = self.I_phi*(rho_cs)/(((rho_cs)**2+(self.c3)**2)**(1.5))
        total = self.w1*f1+self.w2*f2+self.w3*f3
        if self.w4 != 0.0:
            f4 = self.I_phi*(rho_cs)/(((rho_cs)**2+(self.c4)**2)**(1.5))
            total = self.w1*f1+self.w2*f2+self.w3*f3+self.w4*f4
        if each_term:
            self.f1 = f1
            self.f2 = f2
            self.f3 = f3
            if self.w4 != 0.0:
                self.f4 = f4
        return total

    def J_phi_Wang22(self, rho_cs, z, MLT):
        """
        Args:
            rho_cs (float): [RJ]
            z (float): [RJ]
            MLT (float): [hr]

        return: [MA RJ-2]
        """
        R = rho_cs

        def A0(x):
            K0 = 1.17E+0
            K1 = 7.00E-2
            K2 = 3.73E-2
            K3 = 1.38E-2
            K4 = 1.15E-2
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def A1(x):
            K0 = 1.14E+0
            K1 = -1.13E-2
            K2 = -4.29E-3
            K3 = -2.95E-4
            K4 = 4.27E-3
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def A2(x):
            K0 = 2.54E-1
            K1 = 4.78E-2
            K2 = 3.88E-2
            K3 = 1.13E-3
            K4 = 1.05E-2
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def F1(x):
            C0 = 6.50E-1
            C1 = 3.25E-1
            return C0+C1*x

        def F2(x):
            C2 = 1.25E+1
            C3 = -1.36E+1
            C4 = 4.28E+0
            return C2+C3*np.log10(x)+C4*np.log10(x)*np.log10(x)

        def B1(x):
            x1 = 7.0    # [RJ]
            x2 = 11.0   # [RJ]

            if x < x1:
                return F1(x)
            elif x > x2:
                return F2(x)
            elif x1 <= x:
                return (F1(x)*(x2-x) + F2(x)*(x-x1))/(x2-x1)

        return A0(MLT)*np.exp(-0.5*((np.log10(R)-A1(MLT))/(A2(MLT)))**2)*np.exp(-0.5*(z/B1(R))**2)

    def I_phi_Wang22(self, rho_cs, MLT):
        """
        Args:
            rho_cs (float): cylindrical [RJ]
            z (float): cylindrical [RJ]
            MLT (float): [hr]

        return: [MA RJ-2]
        """
        R = rho_cs

        def A0(x):
            K0 = 1.17E+0
            K1 = 7.00E-2
            K2 = 3.73E-2
            K3 = 1.38E-2
            K4 = 1.15E-2
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def A1(x):
            K0 = 1.14E+0
            K1 = -1.13E-2
            K2 = -4.29E-3
            K3 = -2.95E-4
            K4 = 4.27E-3
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def A2(x):
            K0 = 2.54E-1
            K1 = 4.78E-2
            K2 = 3.88E-2
            K3 = 1.13E-3
            K4 = 1.05E-2
            t = x*2*np.pi/24
            return K0+K1*math.cos(t)+K2*math.sin(t)+K3*math.cos(2*t)+K4*math.sin(2*t)

        def F1(x):
            C0 = 6.50E-1
            C1 = 3.25E-1
            return C0+C1*x

        def F2(x):
            C2 = 1.25E+1
            C3 = -1.36E+1
            C4 = 4.28E+0
            return C2+C3*np.log10(x)+C4*np.log10(x)*np.log10(x)

        def B1(x):
            x1 = 7.0    # [RJ]
            x2 = 11.0   # [RJ]

            if x < x1:
                return F1(x)
            elif x > x2:
                return F2(x)
            elif x1 <= x:
                return (F1(x)*(x2-x) + F2(x)*(x-x1))/(x2-x1)

        return A0(MLT)*np.exp(-0.5*((np.log10(R)-A1(MLT))/(A2(MLT)))**2)*math.sqrt(2*np.pi)*B1(R)

    def I_phi_Con20(self, rho_cs):
        """
        Args:
            rho_cs (float): cylindrical [RJ]

        return: [A m-1]
        """
        D_con2020 = 3.6     # [RJ]
        I_0 = (139.6*1E-9)*(4*D_con2020/MU0)     # [A m-1]
        return I_0/rho_cs
