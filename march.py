import numpy as np

import fdm
import utils
from poisson import solve_poisson_fft as solve_poisson
from rhs import rhs_pres, rhs_u_momentum, rhs_w_momentum


def Euler(rhs, field, t, dt, **kwargs):
    """
    Parameters
    ----------
    rhs: function that calculates the rhs of the evolution equation
    field: vector with state variables
    t: time
    dt: timestep length
    **kwargs : other arguments passed to rhs function

    """
    f = field + dt * rhs(field, t, **kwargs)

    return f, t + dt


def RungeKutta3(rhs, field, t, dt, **kwargs):
    """
    Parameters
    ----------
    rhs: function that calculates the rhs of the evolution equation
    field: vector with state variables
    t: time
    dt: timestep length
    **kwargs : other arguments passed to rhs function

    """
    k1 = rhs(field, **kwargs)
    k2 = rhs(field + dt * 0.5 * k1, **kwargs)
    k3 = rhs(field + dt * (2.0 * k2 - k1), **kwargs)
    f = field + dt * (k1 + 4.0 * k2 + k3) / 6.0
    return f, t + dt


def project_velocity(
    u_star,
    w_star,
    U,
    qv,
    ql,
    T,
    Tv,
    P0,
    rho0,
    Pp,
    dt,
    dx,
    dz,
    Z,
    Lx,
    bc_w_bottom=0.0,
    bc_w_top=0.0,
    debug_stats=False,
):
    """
    Solve for Pp from provisional velocities and correct velocities.
    Uses updated rhs_pressure_poisson for the Poisson RHS.
    Enforces Neumann BCs for pressure from the normal component of the momentum equation at the boundaries.
    """
    rhs, bc_neumann_bottom, bc_neumann_top = rhs_pres(u_star, w_star, rho0, dx, dz, dt)

    rhs_mean = np.nanmean(rhs)
    # rhs = rhs - rhs_mean  # anchor to remove nullspace

    if debug_stats:
        print("Poisson RHS mean before:", rhs_mean)
        volume = dx * dz
        integral_rhs = np.sum(rhs) * volume
        print("Integral( pres rhs) =", integral_rhs)
        print(
            "max, L2:",
            np.max(np.abs(rhs)),
            np.linalg.norm(rhs.ravel()),
        )

    # Solve Poisson equation with these Neumann BCs
    Pp = solve_poisson(rhs, rho0, dx, dz, Lx, bc_neumann_bottom, bc_neumann_top)

    _Apcheck = utils.apply_A_check(Pp, rho0, dx, dz)
    # diagnostic residual
    res = _Apcheck - rhs
    if debug_stats:
        print("Pp stats: min/max/mean/nan?", *utils.stats("Pp", Pp)[1:])
        # print(
        #     "A(Pp) stats:",
        #     _Apcheck.min(),
        #     _Apcheck.max(),
        #     "res stats:",
        #     np.nanmin(res),
        #     np.nanmax(res),
        #     "L2",
        #     np.linalg.norm(res.ravel()),
        # )
        print(
            "A(Pp) - rhs: min/max/L2:",
            res.min(),
            res.max(),
            np.linalg.norm(res.ravel()),
        )
        print("rhs: min/max/L2:", rhs.min(), rhs.max(), np.linalg.norm(rhs.ravel()))

    # compute gradients
    gradp_x = fdm.fdm_1_c(Pp, axis=1, periodic=True, dx=dx)
    gradp_z = fdm.fdm_1_c(
        Pp,
        axis=0,
        bc_left_type="neumann",
        bc_left_val=bc_neumann_bottom,
        bc_right_type="neumann",
        bc_right_val=bc_neumann_top,
        dx=dz,
    )

    if debug_stats:
        print(
            "gradp stats:", gradp_x.min(), gradp_x.max(), gradp_z.min(), gradp_z.max()
        )

    u_new = u_star - dt * gradp_x / rho0
    w_new = w_star - dt * gradp_z / rho0

    alpha = 10e-2  # s^-1
    mask = Z > 7000  # assuming z in meters
    w_new[mask] *= 1 - alpha * dt
    u_new[mask] *= 1 - alpha * dt
    # Enforce impermeable boundaries for w
    w_new[0, :] = 0.0
    w_new[-1, :] = 0.0

    # compute divergence after projection
    Fx = rho0 * u_new
    Fz = rho0 * w_new
    dFx_dx = fdm.fdm_1_c(Fx, axis=1, periodic=True, dx=dx)
    dFz_dz = fdm.fdm_1_c(
        Fz,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=bc_w_bottom,
        bc_right_type="dirichlet",
        bc_right_val=bc_w_top,
        dx=dz,
    )
    div_after = dFx_dx + dFz_dz

    if debug_stats:
        print(
            "div_after stats: max, L2:",
            np.max(np.abs(div_after)),
            np.linalg.norm(div_after.ravel()),
        )

    return u_new, w_new, Pp, div_after


def rk3_projection_step(
    u,
    w,
    U,
    qv,
    ql,
    T,
    Tv,
    Tv0,
    P0,
    Pp,
    rho0,
    dt,
    dx,
    dz,
    Z,
    Lx,
    bc_w_bottom=0.0,
    bc_w_top=0.0,
    debug_stats=False,
):
    """
    Advance u,w one time step with RK3 and projection after each Euler substep.
    Returns updated (u,w,Pp,div_after). Other fields (U,qv,ql) unchanged here.
    """

    if debug_stats:
        du_dx = fdm.fdm_1_c(u, axis=1, periodic=True, dx=dx)
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

        div_v = du_dx + dw_dz
        print(
            "div_v min/max/finite:",
            np.nanmin(div_v),
            np.nanmax(div_v),
            np.isfinite(div_v).all(),
        )

    # stage 1: u1 = u + dt * L(u)
    rhs_u1 = rhs_u_momentum(
        u,
        w=w,
        dx=dx,
        dz=dz,
        rho0=rho0,
    )
    rhs_w1 = rhs_w_momentum(
        w,
        u=u,
        U=U,
        qv=qv,
        ql=ql,
        T=T,
        Tv=Tv,
        Tv0=Tv0,
        P0=P0,
        rho0=rho0,
        Pp=Pp,
        dx=dx,
        dz=dz,
    )
    u1_star = u + dt * rhs_u1
    w1_star = w + dt * rhs_w1
    u1, w1, Pp1, div1 = project_velocity(
        u1_star,
        w1_star,
        U,
        qv,
        ql,
        T,
        Tv,
        P0,
        rho0,
        Pp,
        dt,
        dx,
        dz,
        Z,
        Lx,
    )

    # stage 2: u2 = 3/4 u + 1/4 (u1 + dt*L(u1))
    rhs_u2 = rhs_u_momentum(
        u1,
        w=w1,
        dx=dx,
        dz=dz,
        rho0=rho0,
    )
    rhs_w2 = rhs_w_momentum(
        w1,
        u=u1,
        U=U,
        qv=qv,
        ql=ql,
        T=T,
        Tv=Tv,
        Tv0=Tv0,
        P0=P0,
        rho0=rho0,
        Pp=Pp,
        dx=dx,
        dz=dz,
    )
    u2_star = u1 + dt * rhs_u2
    w2_star = w1 + dt * rhs_w2
    u2_proj, w2_proj, Pp2, div2 = project_velocity(
        u2_star,
        w2_star,
        U,
        qv,
        ql,
        T,
        Tv,
        P0,
        rho0,
        Pp,
        dt,
        dx,
        dz,
        Z,
        Lx,
    )
    u2 = 0.75 * u + 0.25 * u2_proj
    w2 = 0.75 * w + 0.25 * w2_proj

    # stage 3: final
    rhs_u3 = rhs_u_momentum(
        u2,
        w=w2,
        dx=dx,
        dz=dz,
        rho0=rho0,
    )
    rhs_w3 = rhs_w_momentum(
        w2,
        u=u2,
        U=U,
        qv=qv,
        ql=ql,
        T=T,
        Tv=Tv,
        Tv0=Tv0,
        P0=P0,
        rho0=rho0,
        Pp=Pp,
        dx=dx,
        dz=dz,
    )

    u3_star = u2 + dt * rhs_u3
    w3_star = w2 + dt * rhs_w3
    u3, w3, Pp_new, div_new = project_velocity(
        u3_star,
        w3_star,
        U,
        qv,
        ql,
        T,
        Tv,
        P0,
        rho0,
        Pp,
        dt,
        dx,
        dz,
        Z,
        Lx,
    )
    u_new = 1 / 3 * u + 2 / 3 * u3
    w_new = 1 / 3 * w + 2 / 3 * w3

    if debug_stats:
        print(
            "div_v (after rk3) min/max/finite:",
            np.nanmin(div_new),
            np.nanmax(div_new),
            np.isfinite(div_new).all(),
        )
    return u_new, w_new, Pp_new, div_new
