import numpy as np

from constants import Cv
import fdm

def vel_sanity(u, w, dx, dz, dt, name=""):
    assert np.isfinite(u).all() and np.isfinite(w).all(), f"{name}: NaN/Inf present"
    umax = np.max(np.abs(u))
    wmax = np.max(np.abs(w))

    cfl_x = umax * dt / dx if dx > 0 else np.inf
    cfl_z = wmax * dt / dz if dz > 0 else np.inf

    print(
        f"{name}: |u|max={umax:.3e}, |w|max={wmax:.3e}, CFLx={cfl_x:.3e}, CFLz={cfl_z:.3e}"
    )
    div = (
        np.diff(u, axis=1, prepend=u[:, :1]) / dx
        + np.diff(w, axis=0, prepend=w[:1, :]) / dz
    )
    div_rel = np.linalg.norm(div) / (np.linalg.norm(np.abs(u) + np.abs(w)) + 1e-30)
    print(f"{name}: div L2={np.linalg.norm(div):.3e}, rel={div_rel:.3e}")
    return cfl_x, cfl_z


def apply_A_check(P, rho0, dx, dz):
    Nz, Nx = P.shape
    rho = rho0[:, 0].astype(np.float64)
    rho_iphalf = 0.5 * (rho[:-1] + rho[1:])
    out = np.zeros_like(P)

    # second derivative in x
    d2x = fdm.fdm_2_c(P, axis=1, periodic=True, dx=dx)

    # interior
    for i in range(1, Nz - 1):
        out[i, :] = (
            -rho_iphalf[i - 1] * P[i - 1, :]
            + (rho_iphalf[i - 1] + rho_iphalf[i]) * P[i, :]
            - rho_iphalf[i] * P[i + 1, :]
        ) / (dz * dz) + rho[i] * d2x[i, :]

    # bottom
    out[0, :] = rho_iphalf[0] * (P[0, :] - P[1, :]) / (dz * dz) + rho[0] * d2x[0, :]

    # top
    out[-1, :] = (
        rho_iphalf[-1] * (P[-1, :] - P[-2, :]) / (dz * dz) + rho[-1] * d2x[-1, :]
    )

    return out


def stats(name, arr):
    arrf = np.asarray(arr)
    return (
        name,
        np.nanmin(arrf),
        np.nanmax(arrf),
        np.mean(arrf),
        np.isfinite(arrf).all(),
    )


def check_fields(step, t_now, u, w, U, qv, ql, Pp, rho0, label="pre"):
    print(f"Step {step}, t={t_now:.6f}, {label} stats:")
    for nm, arr in [
        ("u", u),
        ("w", w),
        ("U", U),
        ("qv", qv),
        ("ql", ql),
        ("Pp", Pp),
        ("rho0", rho0),
    ]:
        _, nmin, nmax, nmean, isfin = stats(nm, arr)
        print(
            f"  {nm:3s} min/max/mean/finite: {nmin:.6e} {nmax:.6e} {nmean:.6e} {isfin}"
        )
    # check T positivity
    T_local = U / Cv
    if np.any(T_local <= 0) or not np.isfinite(T_local).all():
        print("  !!! Temperature invalid: min_T =", np.min(T_local))
        raise RuntimeError("Invalid temperature detected")


def get_grid(start, stop, h=None, n=None, dtype=np.float64):
    if h and n:
        raise TypeError("Provide either only n or h")
    elif h:
        grid = np.arange(start, stop, h, dtype=dtype)
        return grid, h, np.size(grid)
    elif n:
        grid = np.linspace(start, stop, n, dtype=dtype)
        return grid, np.diff(grid[:2])[0], n
    else:
        raise TypeError("Provide either n or h")


def l2_norm(num, ana):
    """
    Parameters:

    num : numerical
    ana : analytical
    """
    return np.linalg.norm(num - ana)


def get_colors_from_cmap(cmap_name, n):
    """
    Get n evenly spaced discrete colors from a named matplotlib colormap.

    Parameters:
    cmap_name (str): Name of the matplotlib colormap (e.g. 'RdBu', 'viridis', etc.)
    n (int): Number of colors to get

    Returns:
    colors (list of str): List of n color specifications compatible with plt.plot's color argument
    """
    from matplotlib.pyplot import get_cmap

    cmap = get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n))
    return colors


def get_standard_atmosphere(p, coordinates="pressure", t_s=288.15):
    """International Standard Atmosphere (ISA).

    The temperature profile is defined between 0-85 km (1089 h-0.004 hPa).
    Values exceeding this range are linearly interpolated.

    Parameters:
        z (float or ndarray): Geopotential height above MSL [m]
            or pressure [Pa] (see ``coordinates``).
        t_s (float, array): surface temperature in Kelvin
        coordinates (str): Either 'height' or 'pressure'.


    Returns:
        ndarray: Atmospheric temperature [K].

    """
    from typhon.physics import standard_atmosphere

    std_temp = standard_atmosphere(p, coordinates)

    p1 = 400e2
    p2 = 100e2
    d_temp_s = t_s - std_temp[0]
    d_temp = np.zeros_like(p)

    ind1 = np.where(p > p1, 1, 0)
    ind2 = np.where((p > p2) & (p < p1), 1, 0)

    d_temp -= d_temp_s * ind1
    d_temp[ind2 == 1] -= np.linspace(d_temp_s, 0, sum(ind2))

    std_temp += d_temp

    return std_temp


def get_temp_profile(z, coordinates="height"):
    """International Standard Atmosphere (ISA).

    The temperature profile is defined between 0-85 km (1089 h-0.004 hPa).
    Values exceeding this range are linearly interpolated.

    Parameters:
        z (float or ndarray): Geopotential height above MSL [m]
            or pressure [Pa] (see ``coordinates``).
        coordinates (str): Either 'height' or 'pressure'.

    Returns:
        ndarray: Atmospheric temperature [K].
    """
    from scipy.interpolate import interp1d

    # international standard atmosphere
    height = np.array([-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    pressure = np.array(
        [108_900, 22_632, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734]
    )
    temp = np.array([+19.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28])

    if coordinates == "height":
        z_ref = height
    elif coordinates == "pressure":
        z_ref = np.log(pressure)
        z = np.log(z)
    else:
        raise ValueError(
            f'"{coordinates}" coordinate is unsupported. ' 'Use "height" or "pressure".'
        )

    return interp1d(z_ref, temp + 273.15, fill_value="extrapolate")(z)


def get_humidity_profile(z, q0=18.65e-3, qt=1e-14, zt=15000, zq1=4000, zq2=7500):
    r"""Compute the water vapor volumetric mixing ratio as function of height.

    .. math::
        \mathrm{H_2O} = q_0
          \exp\left(-\frac{z}{z_{q1}}\right)
          \exp\left[\left(-\frac{z}{z_{q2}}\right)^2\right]

    Parameters:
        z (ndarray): Height [m].
        q0 (float): Specific humidity at the surface [kg/kg].
        qt (float): Specific humidity in the stratosphere [kg/kg].
        zt (float): Troposphere height [m].
        zq1, zq2 (float): Shape parameters.

    Returns:
        ndarray: specific humidity [kg/kg].

    Reference:
        Wing et al., 2017, Radiative-Convective Equilibrium Model
        Intercomparison Project

    """
    q_v = q0 * np.exp(-z / zq1) * np.exp(-((z / zq2) ** 2))

    q_v[z > zt] = qt

    return q_v


def get_liquid_water_profile(z, cloud_bot=3000.0, cloud_top=5000, cloud_q_l=0.003):
    """
    Return initial liquid water content profile q_l (kg/kg) for given altitude z [m],
    based on specified profile type.
    """

    q_l = np.where((z > cloud_bot) & (z < cloud_top), cloud_q_l, 0.001)

    return q_l
