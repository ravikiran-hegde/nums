import numpy as np

DELTA_S = 1.4  # Latitudinal variation of shortwave radiation
GLOBAL_MEAN_NET_SOLAR_FLUX = 938 / 4  # Watts per square meter.
# Includes effect of albedo


def second_legendre_pol(theta):
    """
    Second Legendre polynomial

    Parameters:

    theta     : in degrees

    Returns : value of the second order legendre polynomial
    """
    return 0.25 * (1 - 3 * (np.sin(np.deg2rad(theta))) ** 2)


def calc_solar_flux(lat):
    """
    calculate the net solar flux at the surface as a function of the latitude
    reference: Frierson et al. 2006

    Parameters:

    lat     : latitude in degrees

    Returns : Net solar flux at the surface
    """

    solar_flux = GLOBAL_MEAN_NET_SOLAR_FLUX * (1 + DELTA_S * second_legendre_pol(lat))

    return solar_flux
