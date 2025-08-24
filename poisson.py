import numpy as np


def solve_poisson_fft(
    rhs, rho0, dx, dz, Lx, bc_neumann_bottom, bc_neumann_top, anchor_k0=True
):
    """
    Solve   div( a(z) grad P ) = rhs(x,z)
    on domain (x in [0,Lx] periodic, z in [0,Lz] with Neumann fluxes).
    Uses rFFT in x -> independent tridiagonals in z for each kx.

    Parameters
    ----------
    rhs : ndarray (Nz, Nx)   # ordering: first axis z, second axis x
    a   : ndarray (Nz,) or scalar
          coefficient a(z) = 1/rho0(z) (must NOT vary in x)
    dx, dz : floats
    Lx : float
    bc_neumann_bottom : ndarray (Nx,)  # (a * dP/dz)|bottom as function of x
    bc_neumann_top    : ndarray (Nx,)  # (a * dP/dz)|top    as function of x
    anchor_k0 : bool
        If True, anchor the kx=0 mode to remove Neumann-Neumann nullspace
        (sets P_mean_z_at_k=0 = 0). Otherwise the solver may be singular.

    Returns
    -------
    P : ndarray (Nz, Nx)  real-valued solution, with overall mean removed.
    """

    a = 1 / rho0[:, 0]
    Nz, Nx = rhs.shape
    # rFFT along x (axis=1)
    rhs_hat = np.fft.rfft(rhs, axis=1, norm="ortho")
    b0_hat = np.fft.rfft(bc_neumann_bottom[np.newaxis, :], axis=1, norm="ortho")[0]
    bt_hat = np.fft.rfft(bc_neumann_top[np.newaxis, :], axis=1, norm="ortho")[0]

    # Fourier wavenumbers (kx^2)
    m_vals = np.arange(rhs_hat.shape[1])  # 0..Nx/2
    kx = 2.0 * np.pi * m_vals / Lx
    kx2 = kx**2

    # Ensure a is Nz long
    if np.isscalar(a):
        a_col = np.full(Nz, float(a))
    else:
        a_col = np.asarray(a, dtype=float)
        assert a_col.shape[0] == Nz

    # face-centered a at z half-levels: a_{i+1/2} = 0.5*(a_i + a_{i+1})
    a_face = 0.5 * (a_col[:-1] + a_col[1:])  # length Nz-1

    P_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    inv_dz = 1.0 / dz
    inv_dz2 = inv_dz * inv_dz

    # Loop over Fourier modes (kx)
    for mi, k2 in enumerate(kx2):
        # Build tridiagonal system A P = d (complex)
        N = Nz
        a_sub = np.zeros(N, dtype=np.complex128)  # sub-diagonal (a_i)
        b_diag = np.zeros(N, dtype=np.complex128)  # diag
        c_super = np.zeros(N, dtype=np.complex128)  # super-diag
        d = rhs_hat[:, mi].astype(np.complex128).copy()

        # interior rows 1..Nz-2
        for i in range(1, Nz - 1):
            a_sub[i] = a_face[i - 1] * inv_dz2
            c_super[i] = a_face[i] * inv_dz2
            b_diag[i] = -(a_face[i - 1] + a_face[i]) * inv_dz2 - a_col[i] * k2

        # bottom row i=0 : implement Neumann flux
        # discrete flux at bottom: a_{1/2} * (P1 - P0)/dz = f_bot_hat[mi]
        rf_b = a_face[0]  # a_{1/2}
        # Row: (-rf_b/dz) * P0 + (rf_b/dz) * P1  - a0*k2*P0 = d0
        b_diag[0] = -(rf_b * inv_dz) - a_col[0] * k2
        c_super[0] = rf_b * inv_dz
        d[0] = b0_hat[mi]

        # top row i=Nz-1 : flux at top a_{N-1/2} * (P_{N-1} - P_{N-2})/dz = f_top_hat[mi]
        rf_t = a_face[-1]  # a_{N-1/2}
        a_sub[-1] = rf_t * inv_dz
        b_diag[-1] = -(rf_t * inv_dz) - a_col[-1] * k2
        d[-1] = bt_hat[mi]

        # handle kx=0 nullspace: anchor one dof (set P0=0)
        if (k2 == 0.0) and anchor_k0:
            b_diag[0] = 1.0 + 0j
            c_super[0] = 0.0 + 0j
            a_sub[0] = 0.0 + 0j
            d[0] = 0.0 + 0j

        # Thomas algorithm (complex)
        cp = np.zeros(N, dtype=np.complex128)
        dp = np.zeros(N, dtype=np.complex128)
        beta = b_diag[0]
        if np.abs(beta) == 0:
            raise FloatingPointError("Singular first diagonal entry")
        cp[0] = c_super[0] / beta
        dp[0] = d[0] / beta
        for i in range(1, N):
            beta = b_diag[i] - a_sub[i] * cp[i - 1]
            if np.abs(beta) == 0:
                raise FloatingPointError(f"Singular at row {i}")
            cp[i] = c_super[i] / beta if i < N - 1 else 0.0 + 0j
            dp[i] = (d[i] - a_sub[i] * dp[i - 1]) / beta

        # back substitution
        xcol = np.zeros(N, dtype=np.complex128)
        xcol[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            xcol[i] = dp[i] - cp[i] * xcol[i + 1]

        P_hat[:, mi] = xcol

    # inverse FFT to physical space
    P = np.fft.irfft(P_hat, n=Nx, axis=1, norm="ortho").real

    # remove domain mean (optional but useful)
    P -= np.mean(P)

    return P
