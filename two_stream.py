import numpy as np

SIGMA = 5.6734e-8  # Stefan–Boltzmann constant im W m^-2 K^-4


def calc_emission_stefan_boltzamann(temp, EMISSIVITY=1):
    """
    Equation for black body emission
    """
    return SIGMA * EMISSIVITY * (temp) ** 4


def solve_two_stream(p_atm, temp_atm, tau_atm):
    """
    Solve for the radiation in the atmospheric levels using two-stream approximation.

    Parameters:
    p_atm       : atmospheric pressure (1D array of shape (N,)), just passed here for consistency and not used.
    temp_atm    : atmospheric temperature (2D array of shape (N, n1) or (N,))
    tau_atm     : optical depth of the atmosphere (2D array of shape (N, n2) or (N,))

    Returns:
    rnet    : net radiation flux (longwave) (shape (N, n1, n2))
    ru      : upward flux (shape (N, n1, n2))
    rd      : downward flux (shape (N, n1, n2))
    """
    p_atm = np.asarray(p_atm)
    tau_atm = np.asarray(tau_atm)
    temp_atm = np.asarray(temp_atm)

    # Ensure temp_atm and tau_atm are at least 2D
    if temp_atm.ndim == 1:
        temp_atm = temp_atm[:, None]  # Convert (N,) to (N, 1)
    if tau_atm.ndim == 1:
        tau_atm = tau_atm[:, None]  # Convert (N,) to (N, 1)

    # # Ensure temp_atm and tau_atm have the same number of levels (N)
    # if temp_atm.shape[0] != len(p_atm) or tau_atm.shape[0] != len(p_atm):
    #     raise ValueError("temp_atm and tau_atm must match the number of levels in p_atm.")

    # # Expand dimensions to allow broadcasting
    # temp_atm = temp_atm[..., None]  # (N, n1, 1)
    # tau_atm = tau_atm[:, None, :]  # (N, 1, n2)

    # Calculate blackbody emission [ Shape ]
    B = calc_emission_stefan_boltzamann(temp_atm)  # (N, n1, 1)

    # Calculate optical depth and transmission of each layer [Shape (N, 1, n2)]
    dtau_atm = -np.diff(tau_atm, append=0, axis=0)  # (N, 1, n2)
    trans_atm = np.exp(-dtau_atm)
    one_minus_trans = 1 - trans_atm  # self absorption of the layer ## /2 ?
    # DO : INTRODUCE HALF LAYERS TO BE MORE EXACT

    # radiation field initialization
    N, n1 = temp_atm.shape[0], temp_atm.shape[1]
    ru = np.zeros((N, n1))
    rd = np.zeros((N, n1))

    # Boundary conditions
    ru[0] = B[0, :]  # Upward flux at the surface
    rd[-1] = 0  # Downward flux at the top of the atmosphere

    # Compute ru (upward flux)
    for i in range(1, N):
        ru[i] = ru[i - 1] * trans_atm[i] + B[i] * one_minus_trans[i]

    # Compute rd (downward flux)
    for i in range(N - 2, -1, -1):
        rd[i] = rd[i + 1] * trans_atm[i] + B[i] * one_minus_trans[i]

    ru = np.squeeze(ru)
    rd = np.squeeze(rd)

    # Net radiation flux
    rnet = ru - rd

    return -rnet, -ru, -rd


def solve_two_stream_test(p_atm, temp_atm, tau_atm):
    """
    Solve for the radiation in the atmospheric levels using two-stream approximation.

    Parameters:
    p_atm       : atmospheric pressure (1D array of shape (N,)), just passed here for consistency and not used.
    temp_atm    : atmospheric temperature (2D array of shape (N, n1) or (N,))
    tau_atm     : optical depth of the atmosphere (2D array of shape (N, n2) or (N,))

    Returns:
    rnet    : net radiation flux (longwave) (shape (N, n1, n2))
    ru      : upward flux (shape (N, n1, n2))
    rd      : downward flux (shape (N, n1, n2))
    """
    p_atm = np.asarray(p_atm)
    tau_atm = np.asarray(tau_atm)
    temp_atm = np.asarray(temp_atm)

    # Ensure temp_atm and tau_atm are at least 2D
    if temp_atm.ndim == 1:
        temp_atm = temp_atm[:, None]  # Convert (N,) to (N, 1)
    if tau_atm.ndim == 1:
        tau_atm = tau_atm[:, None]  # Convert (N,) to (N, 1)

    # Ensure temp_atm and tau_atm have the same number of levels (N)
    if temp_atm.shape[0] != len(p_atm) or tau_atm.shape[0] != len(p_atm):
        raise ValueError(
            "temp_atm and tau_atm must match the number of levels in p_atm."
        )

    # Expand dimensions to allow broadcasting
    temp_atm = temp_atm[..., None]  # (N, n1, 1)
    tau_atm = tau_atm[:, None, :]  # (N, 1, n2)

    # Calculate blackbody emission [ Shape ]
    B = calc_emission_stefan_boltzamann(temp_atm)  # (N, n1, 1)

    # Calculate optical depth and transmission of each layer [Shape (N, 1, n2)]
    dtau_atm = -np.diff(tau_atm, append=0, axis=0)  # (N, 1, n2)
    trans_atm = np.exp(-dtau_atm)
    one_minus_trans = 1 - trans_atm

    # radiation field initialization
    N, n1, n2 = temp_atm.shape[0], temp_atm.shape[1]
    ru = np.zeros((N, n1, n2))
    rd = np.zeros((N, n1, n2))
    ru1 = np.zeros((N, n1, n2))
    rd1 = np.zeros((N, n1, n2))

    # Boundary conditions
    ru[0] = B[0, :, :]  # Upward flux at the surface
    rd[-1] = 0  # Downward flux at the top of the atmosphere

    # Compute ru (upward flux)
    for i in range(1, N):
        ru[i] = ru[i - 1] * trans_atm[i] + B[i] * one_minus_trans[i]
        ru1[i] = ru[i - 1] * trans_atm[i]

    # Compute rd (downward flux)
    for i in range(N - 2, -1, -1):
        rd[i] = rd[i + 1] * trans_atm[i] + B[i] * one_minus_trans[i]
        rd1[i] = rd[i + 1] * trans_atm[i]
    ru = np.squeeze(ru)
    rd = np.squeeze(rd)
    ru1 = np.squeeze(ru1)
    rd1 = np.squeeze(rd1)

    # Net radiation flux
    rnet = ru - rd

    return rnet, ru, rd, ru1, rd1
