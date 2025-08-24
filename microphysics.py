import numpy as np

from constants import Lv


def es_tetens(T):
    """
    Calculate saturation vapor pressure over water using Tetens equation (in Pa).
    Parameters
    ----------
    T: Temperature [K]

    Returns
    -------
    Saturation vapur pressure in Pa
    """
    return 610.94 * np.exp((17.625 * (T - 273.15)) / (T - 273.15 + 237.3))


def e_s(T):
    """
    Calculate saturation vapor pressure

    Paramaters
    ----------
    T: Temperature [K]

    Returns
    -------
    Saturation vapur pressure in Pa
    """
    return es_tetens(T)


def calc_saturation_mixing_ratio(T, P, epsilon=0.622):
    """
    Compute the saturation mixing ratio q_vs

    Paramaters
    ----------
    T: Temperature [K]
    P: Pressure [Pa]
    epsilon

    Returns
    -------
    Saturation mixing ratio
    """
    e_s_value = e_s(T)
    return epsilon * e_s_value / (P - e_s_value)


def saturation_adjustment(q_v, q_l, T, P):
    """
    Perform the saturation adjustment

    Paramaters
    ----------
    q_v: water vapour mixing ratio
    q_l: liquid water mixing ratio
    T: Temperature [K]
    P: Pressure [Pa]

    Returns
    -------
    new_q_v: updated water vapour mixing ratio
    new_q_l: updated liquid water mixing ratio
    U_change_sa: U change due to saturation adjustment
    """
    q_vs = calc_saturation_mixing_ratio(T, P)

    mask = q_v > q_vs
    delta_q = np.where(mask, q_v - q_vs, 0.0)
    new_q_v = np.where(mask, q_vs, q_v)
    new_q_l = q_l + delta_q
    U_change_sa = -Lv * delta_q

    return new_q_v, new_q_l, U_change_sa
