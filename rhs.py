import numpy as np

import fdm
from constants import Cond_rt, Cv, Evap_rt, Lv, Rd, Rv, g, kappa


def rhs_u_momentum(
    u, w, dx, dz, rho0, bc_u_bottom=0.0, bc_u_top=0.0, debug_stats=False
):
    """RHS for horizontal momentum equation."""
    # Advection
    du_dx = fdm.fdm_1_c(u, axis=1, periodic=True, dx=dx)
    du_dz = fdm.fdm_1_c(
        u,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=bc_u_bottom,
        bc_right_type="neumann",
        bc_right_val=bc_u_top,
        periodic=False,
        dx=dz,
    )
    adv_u = u * du_dx + w * du_dz

    # Diffusion
    diff_u = kappa * (
        fdm.fdm_2_c(u, axis=1, periodic=True, dx=dx)
        + fdm.fdm_2_c(
            u,
            axis=0,
            bc_left_type="dirichlet",
            bc_left_val=bc_u_bottom,
            bc_right_type="neumann",
            bc_right_val=bc_u_top,
            periodic=False,
            dx=dz,
        )
    )
    # Damping
    damp_u = 0.0

    if debug_stats:
        print("Momentum RHS (u):")
        print(f"  adv      min/max: {adv_u.min()} {adv_u.max()}")
        print(f"  diff     min/max: {(diff_u/rho0).min()} {(diff_u/rho0).max()}")
        print(f"  damp     min/max: {damp_u.min()} {damp_u.max()}")
    return -adv_u + diff_u + damp_u


def rhs_w_momentum(
    w,
    u,
    U,
    qv,
    ql,
    T,
    Tv,
    Tv0,
    P0,
    rho0,
    Pp,
    dx,
    dz,
    bc_w_bottom=0.0,
    bc_w_top=0.0,
    debug_stats=False,
):
    """RHS for vertical momentum equation."""
    # Advection
    dw_dx = fdm.fdm_1_c(w, axis=1, periodic=True, dx=dx)
    dw_dz = fdm.fdm_1_c(
        w,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=bc_w_bottom,
        bc_right_type="dirichlet",
        bc_right_val=bc_w_top,
        periodic=False,
        dx=dz,
    )
    adv_w = u * dw_dx + w * dw_dz

    # Diffusion
    diff_w = kappa * (
        fdm.fdm_2_c(w, axis=1, periodic=True, dx=dx)
        + fdm.fdm_2_c(
            w,
            axis=0,
            bc_left_type="dirichlet",
            bc_left_val=bc_w_bottom,
            bc_right_type="dirichlet",
            bc_right_val=bc_w_top,
            periodic=False,
            dx=dz,
        )
    )

    # Buoyancy
    Tv = U / Cv * (1 + (Rv / Rd - 1) * qv - ql)
    b = g * (Tv - Tv0) / Tv0

    # Damping
    damp_w = 0.0

    if debug_stats:
        print("Momentum RHS (w):")
        print(f"  adv      min/max: {adv_w.min()} {adv_w.max()}")
        print(f"  diff     min/max: {(diff_w/rho0).min()} {(diff_w/rho0).max()}")
        print(f"  buoy     min/max: {(b/rho0).min()} {(b/rho0).max()}")
        print(f"  damp     min/max: {damp_w.min()} {damp_w.max()}")

    return -adv_w + diff_w + b + damp_w


def rhs_internal_energy(
    U,
    u,
    w,
    Pp,
    rho0,
    dx,
    dz,
    kT=None,
    div_jr=None,
    Q_ext=None,
    Lv=Lv,
    Cond_rt=Cond_rt,
    Evap_rt=Evap_rt,
    rad_dTdt_K_per_day=2.0,
    F_rad=None,
    bc_U_bottom=0.0,
    debug_stats=False,
):
    """
    Returns dU/dt (J/kg/s).
    """
    # Set top Neumann BC for U (prescribed flux)
    bc_U_top_val = -Cv / kT * F_rad if (F_rad is not None and kT is not None) else 0.0

    # Gradients
    dU_dx = fdm.fdm_1_c(U, axis=1, periodic=True, dx=dx)
    dU_dz = fdm.fdm_1_c(
        U,
        axis=0,
        bc_left_type="neumann",
        bc_left_val=bc_U_bottom,
        bc_right_type="neumann",
        bc_right_val=bc_U_top_val,
        periodic=False,
        dx=dz,
    )

    # Advection:
    adv_U = u * dU_dx + w * dU_dz

    # Divergence of velocity
    div_v = (
        -(
            fdm.fdm_1_c(rho0 * u, axis=1, periodic=True, dx=dx)
            + fdm.fdm_1_c(
                rho0 * w,
                axis=0,
                bc_left_type="neumann",
                bc_left_val=0.0,
                bc_right_type="neumann",
                bc_right_val=0.0,
                periodic=False,
                dx=dz,
            )
        )
        / rho0
    )

    # Conductive heat-flux divergence:
    if kT is None:
        div_jh = 0.0
    else:
        d2T_dx = fdm.fdm_2_c(U, axis=1, periodic=True, dx=dx)
        d2T_dz = fdm.fdm_2_c(
            U,
            axis=0,
            bc_left_type="neumann",
            bc_left_val=bc_U_bottom,
            bc_right_type="neumann",
            bc_right_val=bc_U_top_val,
            periodic=False,
            dx=dz,
        )
        div_jh = -kT / Cv * (d2T_dx + d2T_dz)

    # Radiative + external heating
    if div_jr is None:
        div_jr = rho0 * Cv * (rad_dTdt_K_per_day / 86400.0)
    Q_ext = 0.0 if Q_ext is None else Q_ext

    # Latent heating
    Cond_rt = 0.0 if Cond_rt is None else Cond_rt
    Evap_rt = 0.0 if Evap_rt is None else Evap_rt
    Q_lat = rho0 * Lv * (Cond_rt - Evap_rt)

    # Assemble RHS
    rhs_U = -adv_U + (-Pp * div_v + div_jh + div_jr + Q_ext + Q_lat) / rho0

    term_adv = -adv_U
    term_pw = -Pp * div_v / rho0
    term_cond = div_jh / rho0
    term_lat = Q_lat / rho0
    term_ext = Q_ext / rho0
    term_rad = div_jr / rho0

    if debug_stats:
        print("Energy RHS components:")
        print("  adv      min/max:", np.min(term_adv), np.max(term_adv))
        print("  presswrk min/max:", np.min(term_pw), np.max(term_pw))
        print("  conduct  min/max:", np.min(term_cond), np.max(term_cond))
        print("  latent   min/max:", np.min(term_lat), np.max(term_lat))
        print("  ext      min/max:", np.min(term_ext), np.max(term_ext))
        print("  rad value:", np.min(term_rad), np.max(term_rad))

    return rhs_U


def rhs_moisture(
    qv,
    u,
    w,
    rho0,
    Kq,
    dx,
    dz,
    in_rt=None,
    out_rt=None,
    bc_qv_top=0.0,
    debug_stats=False,
):
    """
    d q_x / dt  (per unit mass)
    (∂t qv + v·∇qv) = ∇·(Kq ∇qv) - (in_rt - out_rt)
    """

    # Conversion (kg/kg/s)
    in_rt = 0.0 if in_rt is None else in_rt
    out_rt = 0.0 if out_rt is None else out_rt
    conv = in_rt - out_rt
    # Set bottom Neumann BC
    bc_qv_bottom_val = conv

    # Advection
    dqv_dx = fdm.fdm_1_c(qv, axis=1, periodic=True, dx=dx)
    dqv_dz = fdm.fdm_1_c(
        qv,
        axis=0,
        bc_left_type="neumann",
        bc_left_val=bc_qv_bottom_val,
        bc_right_type="neumann",
        bc_right_val=bc_qv_top,
        periodic=False,
        dx=dz,
    )
    adv = u * dqv_dx + w * dqv_dz

    # Diffusion in flux form: div( ρ0 Kq ∇qv )
    Fx = rho0 * Kq * dqv_dx
    Fz = rho0 * Kq * dqv_dz
    dFx_dx = fdm.fdm_1_c(Fx, axis=1, periodic=True, dx=dx)
    dFz_dz = fdm.fdm_1_c(
        Fz,
        axis=0,
        bc_left_type="neumann",
        bc_left_val=bc_qv_bottom_val,
        bc_right_type="neumann",
        bc_right_val=bc_qv_top,
        periodic=False,
        dx=dz,
    )
    diff = (dFx_dx + dFz_dz) / rho0

    if debug_stats:
        print("Moisture RHS components:")
        print("  adv    min/max:", np.min(adv), np.max(adv))
        print("  diff   min/max:", np.min(diff), np.max(diff))
        print("  conv   min/max:", np.min(conv), np.max(conv))

    return -adv + diff + conv


def rhs_pres(u_star, w_star, rho0, dx, dz, dt):

    d_dx = fdm.fdm_1_c(rho0 * u_star, axis=1, periodic=True, dx=dx)
    d_dz = fdm.fdm_1_c(
        rho0 * w_star,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=rho0[0, :] * w_star[0, :],
        bc_right_type="dirichlet",
        bc_right_val=rho0[-1, :] * w_star[-1, :],
        dx=dz,
    )
    bc_bottom = rho0[0, :] * w_star[0, :] / dt
    bc_top = rho0[-1, :] * w_star[-1, :] / dt
    rhs = 1 / dt * (d_dx + d_dz)
    return rhs, bc_bottom, bc_top


def rhs_pressure_poisson(
    u,
    w,
    U,
    qv,
    ql,
    T,
    Tv,
    P0,
    rho0,
    Pp,
    dx,
    dz,
    bc_w_bottom=0.0,
    bc_w_top=0.0,
    debug_stats=False,
):
    """
    Compute the RHS of the pressure Poisson equation:
        ∇²P' = ∇.(rhs_momentum)
    with periodic BCs in x and Neumann BCs (zero normal derivative) in z.
    """

    rhs_u = rhs_u_momentum(
        u,
        w=w,
        dx=dx,
        dz=dz,
        rho0=rho0,
    )
    rhs_w = rhs_w_momentum(
        w,
        u=u,
        U=U,
        qv=qv,
        ql=ql,
        T=T,
        Tv=Tv,
        P0=P0,
        rho0=rho0,
        Pp=Pp,
        dx=dx,
        dz=dz,
    )

    diff_w = kappa * (
        fdm.fdm_2_c(w, axis=1, periodic=True, dx=dx)
        + fdm.fdm_2_c(
            w,
            axis=0,
            bc_left_type="dirichlet",
            bc_left_val=bc_w_bottom,
            bc_right_type="dirichlet",
            bc_right_val=bc_w_top,
            periodic=False,
            dx=dz,
        )
    )

    # Buoyancy
    b = g * (Tv - Tv0) / Tv0

    bc_bottom = diff_w[0, :] + b[0, :]
    bc_top = diff_w[-1, :] + b[-1, :]

    # rhs = fdm.fdm_1_c(rho0 * rhs_u, axis=1, periodic=True, dx=dx) + fdm.fdm_1_c(
    #     rho0 * rhs_w,
    #     axis=0,
    #     bc_left_type="dirichlet",
    #     bc_left_val=(rho0 * rhs_w)[0, :],
    #     bc_right_type="dirichlet",
    #     bc_right_val=(rho0 * rhs_w)[-1, :],
    #     dx=dz,
    # )
    rhs = fdm.fdm_1_c(rho0 * rhs_u, axis=1, periodic=True, dx=dx) + fdm.fdm_1_c(
        rho0 * rhs_w,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=0,  # rho0[0, :] * bc_bottom,
        bc_right_type="dirichlet",
        bc_right_val=0,  # rho0[-1, :] * bc_top,
        dx=dz,
    )
    return rhs, bc_bottom, bc_top
