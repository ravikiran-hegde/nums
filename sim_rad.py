# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.integrate import cumulative_trapezoid

import microphysics
import radiation
import utils
from constants import Cond_rt, Cv, Evap_rt, Lv, Prec_rt, Rd, Rv, g, kappa
from march import RungeKutta3, rk3_projection_step
from rhs import rhs_internal_energy, rhs_moisture
from utils import check_fields, stats, vel_sanity

debug_stats = True
filename = "sim_rad_03"
# %%
# SIMULATION DOMAIN SETUP
# ---------------------------

# --- Domain ----
Lx, Lz = 60.0e3, 14.0e3  # Domain length
Nx, Nz = None, None  # Number of gridpoints
dx, dz = 1e2, 0.25e2  # Spatial step
T_final = 1000.0  # Final time
dt = 1e-1  # Time step
N_t = None  # Number of timesteps

x, dx, Nx = utils.get_grid(0, Lx, dx, Nx)
z, dz, Nz = utils.get_grid(0, Lz, dz, Nz)
X, Z = np.meshgrid(x, z)

t, dt, N_t = utils.get_grid(0, T_final, dt, N_t)

print("Δx =", dx, "Δz=", dz, " Δt =", dt)

# --- Initialisation - Hydrostatic atmosphere ---
P_surf = 108900.0  # [Pa] (1081 hPa)
T0_ = utils.get_temp_profile(z, coordinates="height")  # [K]

integrand = g / (Rd * T0_)
P0_ = P_surf * np.exp(-cumulative_trapezoid(integrand, z, initial=0))  # [Pa]

rho0_ = P0_ / (Rd * T0_)  # [kg/m^3]

# --- Initialisation - water ---
qv_ = utils.get_humidity_profile(z)
ql_ = utils.get_liquid_water_profile(z)
# ql_ = np.ones_like(qv_) * 0.001

Tv0_ = T0_ * (1 + (Rv / Rd - 1) * qv_ - ql_)
# %%
# --- Plotting initial 1D state ---
plot_fig = False
if plot_fig:
    fig, axs = plt.subplots(1, 5, figsize=(4 * 5, 6), sharey=True)

    axs[0].plot(T0_, z / 1000)
    axs[0].plot(Tv0_, z / 1000, label="Virtual temperature")
    axs[0].set_xlabel("Temperature [K]")
    axs[0].set_ylabel("Geopotential Height [km]")
    axs[0].set_title("$T_0(z)$")
    axs[0].legend()

    axs[1].plot(P0_ / 100, z / 1000)
    axs[1].set_xlabel("Pressure [hPa]")
    axs[1].set_title("$P_0(z)$")
    axs[1].set_xscale("log")

    axs[2].plot(rho0_, z / 1000)
    axs[2].set_xlabel("Density [kg/m³]")
    axs[2].set_title("$\\rho_0(z)$")

    axs[3].plot(qv_ * 1e3, z / 1000)  # convert to g/kg for readability
    axs[3].set_xlabel("Specific humidity [g/kg]")
    axs[3].set_title("$qv(z)$")

    axs[4].plot(ql_ * 1e3, z / 1000)  # convert to g/kg for readability
    axs[4].set_xlabel("Liquid Water [g/kg]")
    axs[4].set_title("$ql(z)$")

    plt.tight_layout()
    # plt.savefig("init_atm_prof.png", bbox_inches = "tight", dpi = 512)
    plt.show()
# %%
# INITIALISE 2D PROGNOSTIC FIELDS
# ---------------------------
# 2D arrays, shape (Nz, Nx)

T0 = T0_[:, np.newaxis] * np.ones_like(X)
P0 = P0_[:, np.newaxis] * np.ones_like(X)
rho0 = rho0_[:, np.newaxis] * np.ones_like(X)
qv = qv_[:, np.newaxis] * np.ones_like(X)
ql = ql_[:, np.newaxis] * np.ones_like(X)
Tv0 = Tv0_[:, np.newaxis] * np.ones_like(X)

# Velocity fields
u = np.zeros((Nz, Nx))
w = np.zeros((Nz, Nx))

# Pressure perturbation
Pp = np.zeros((Nz, Nx))

# --- Earth like ---

Phi = np.linspace(0, 360, Nx) * np.pi / 180
Eq_Pl_T_diff = 15.0
Tp = -(Eq_Pl_T_diff * np.sin(Phi) ** 2)[np.newaxis, :]

T = T0 + Tp
# Recompute initial fields after perturbation
Pp = P_surf * (
    np.exp(-cumulative_trapezoid(g / (Rd * T), Z, initial=0))
    - np.exp(-cumulative_trapezoid(g / (Rd * T0), Z, initial=0))
)
# Pp = np.zeros_like(P0)
U = Cv * T

rho0 = P0 / (Rd * T)
Tv = T * (1 + (Rv / Rd - 1) * qv - ql)

if plot_fig:
    fig, ax = plt.subplots(figsize=(10, Lz / Lx / 1.1 * 10))

    pcm = ax.pcolormesh(X / 1000, Z / 1000, T, cmap="magma", shading="auto")
    cbar = plt.colorbar(pcm, ax=ax)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Z (km)")
    ax.set_title("Temperature at t = 0 s")
    plt.tight_layout()
    plt.show()
# ---------------------------
# BOUNDARY CONDITIONS SETUP
# ---------------------------
# Boundary values
bc_u_bottom = 0.0  # Dirichlet at bottom (no-slip)
bc_u_top = 0.0  # Neumann at top (free-slip, value is derivative)
bc_w_bottom = 0.0  # Dirichlet at bottom (impermeable)
bc_w_top = 0.0  # Dirichlet at top (impermeable)
bc_U_bottom = 0.0  # Neumann at bottom (adiabatic)
bc_U_top = None  # Neumann at top (prescribed flux, set below)
bc_qv_bottom = None  # Neumann at bottom (evap flux, set below)
bc_qv_top = 0.0  # Neumann at top (no flux)
bc_ql_bottom = 0.0  # Neumann at bottom
bc_ql_top = 0.0  # Neumann at top


# %%
# %%
# TIME-STEPPING LOOP
# ---------------------------
def stepper(rhs, field, t, dt, **kwargs):
    # Easily switch between RungeKutta3, Euler, etc.
    return RungeKutta3(rhs, field, t, dt, **kwargs)


output_list = []

out_freq = 15
try:
    for n in range(N_t):
        t_now = t[n]

        check_fields(n, t_now, u, w, U, qv, ql, Pp, rho0, label="before projection")

        u, w, Pp, div = rk3_projection_step(
            u=u,
            w=w,
            U=U,
            qv=qv,
            ql=ql,
            T=T,
            Tv=Tv,
            Tv0=Tv0,
            P0=P0,
            Pp=Pp,
            rho0=rho0,
            dt=dt,
            dx=dx,
            dz=dz,
            bc_w_bottom=bc_w_bottom,
            bc_w_top=bc_w_bottom,
            Z=Z,
            Lx=Lx,
        )
        check_fields(n, t_now, u, w, U, qv, ql, Pp, rho0, label="after projection")

        cflx, cflz = vel_sanity(u, w, dx, dz, dt, name="momentum update")
        rad_net = radiation.get_radiation(U / Cv, P0, lats = Phi)
        div_jr = radiation.get_div_jr(rad_net=rad_net, dx=dx, dz=dz)
        print(
            f"""
            Radiation:
            rad_net min/max/mean/nan? : {stats("",rad_net)}
            rad_net min/max/mean/nan? : {stats("",div_jr)}
            """
        )
        # 5. Update internal energy (U) using RK3
        U, _ = stepper(
            rhs_internal_energy,
            U,
            t_now,
            dt,
            u=u,
            w=w,
            Pp=Pp,
            rho0=rho0,
            dx=dx,
            dz=dz,
            Lv=Lv,
            bc_U_bottom=bc_U_bottom,
            div_jr=div_jr,
            F_rad=-rad_net[-1, :],
        )
        check_fields(n, t_now, u, w, U, qv, ql, Pp, rho0, label="after energy update")

        # 6. Update moisture fields (qv, ql) using RK3
        qv, _ = stepper(
            rhs_moisture,
            qv,
            t_now,
            dt,
            u=u,
            w=w,
            rho0=rho0,
            Kq=kappa,
            dx=dx,
            dz=dz,
            in_rt=Evap_rt,
            out_rt=Cond_rt,
            bc_qv_top=bc_qv_top,
        )
        ql, _ = stepper(
            rhs_moisture,
            ql,
            t_now,
            dt,
            u=u,
            w=w,
            rho0=rho0,
            Kq=kappa,
            dx=dx,
            dz=dz,
            in_rt=Cond_rt,
            out_rt=Prec_rt,
            bc_qv_top=bc_ql_top,
        )

        # # 7. Apply microphysics (saturation adjustment) cellwise
        qv, ql, U_change = microphysics.saturation_adjustment(qv, ql, U / Cv, P0 + Pp)
        U -= U_change
        check_fields(n, t_now, u, w, U, qv, ql, Pp, rho0, label="after microphysics")

        # 8. Output/progress
        if n % min(out_freq, N_t) == 0:
            print(f"Step {n}/{N_t}, t={t_now:.3f}")
            # add state to a xarray and save output at the end
            T = U / Cv

            # Save state to output_list
            # Save checkpoint every out_freq steps, but only keep in memory for final output
            ds = xr.Dataset(
                {
                    "T": (("z", "x"), T),
                    "qv": (("z", "x"), qv),
                    "ql": (("z", "x"), ql),
                    "u": (("z", "x"), u),
                    "w": (("z", "x"), w),
                    "Pp": (("z", "x"), Pp),
                    "rad": (("z", "x"), rad_net),
                },
                coords={
                    "z": z,
                    "x": x,
                    "time": t_now,
                },
            )
            output_list.append(ds)
            # Checkpoint: save to disk every out_freq steps (optional, for safety)
            if n % (out_freq * 5) == 0 and n > 0:
                try:
                    xr.concat(output_list, dim="time").to_netcdf(
                        f"{filename}_checkpoint.nc"
                    )
                except Exception:
                    filename = filename + f"{np.random.randint(100,999)}"
                    xr.concat(output_list, dim="time").to_netcdf(
                        filename + "_checkpoint.nc"
                    )
                    print(f"Output saved to {filename}_checkpoint.nc")
except Exception as e:
    print("Exception occurred:", e)
    if output_list:
        ds_all = xr.concat(output_list, dim="time")
        try:
            ds_all.to_netcdf(f"{filename}_partial.nc")
            print(f"Partial output saved to {filename}_partial.nc")
        except Exception as e:
            filename = filename + f"{np.random.randint(100,999)}"
            ds_all.to_netcdf(filename + "_partial.nc")
            print(f"Partial output saved to {filename}_partial.nc")


# %%
# --- Save all outputs at the end ---
if output_list:
    ds_all = xr.concat(output_list, dim="time")
    try:
        ds_all.to_netcdf(f"{filename}.nc")
        print("Run complete")
        print(f"Output saved to {filename}.nc")

    except Exception:
        filename = filename + f"{np.random.randint(100,999)}"
        ds_all.to_netcdf(filename + ".nc")
        print(f"Output saved to {filename}.nc")

# %%
# --- Plot initial and final temperature perturbation ---
plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.title("Initial T perturbation")
plt.contourf(
    X / 1000, Z / 1000, (output_list[0]["T"].values - T0), levels=100, cmap="viridis"
)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("Final T perturbation")
plt.contourf(
    X / 1000, Z / 1000, (output_list[-1]["T"].values - T0), levels=100, cmap="viridis"
)
plt.colorbar()
plt.tight_layout()
plt.show()
