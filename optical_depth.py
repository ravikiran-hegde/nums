import numpy as np


def calc_tau_s(lat, TAU_0P, TAU_0E):
    """
    calculate the optical depth at the surface as a function of the latitude
    Reference: Frierson et al. 2006

    Parameters:

    lat  (float or array)   : latitude in degrees
    TAU_0P (float) : surface optical depth at the pole
    TAU_0E (float) : surface optical depth at the equator


    Returns:

    tau_s (same as lat) : optical depth at the surface

    """
    tau_s = TAU_0E + (TAU_0P - TAU_0E) * (np.sin(np.deg2rad(lat))) ** 2
    return tau_s


def calc_tau_atm(p, tau_s, f_l, p_surf=108900.0):
    """
    calculate the optical depth for the atmospheric levels
    Reference: Frierson et al. 2006

    Parameters:

    p       : pressure array for the atmosphere
    tau_s   : optical depth at the surface
    f_l     : linear optical depth parameter (for stratosphere) 0.1

    Returns:
    tau_atm : optical depth at the atmospheric levels
    """

    # p = np.atleast_1d(p)[:, None, None]
    # tau_s = np.atleast_1d(tau_s)[None,:, None]
    # f_l = np.atleast_1d(f_l)[None, None,:]

    p_scaled = p / p_surf
    tau_atm = tau_s * (f_l * p_scaled + (1 - f_l) * p_scaled**4)

    return np.squeeze(tau_atm)
