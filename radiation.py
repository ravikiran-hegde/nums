import fdm
import optical_depth
import solar_radiation
import two_stream

TAU_0P = 1.5  # surface optical depth at the pole
TAU_0E = 6  # surface optical depth at the equator
f_l = 0.3


def get_radiation(temp_atm, p, lats=0.0):

    tau_s = optical_depth.calc_tau_s(lats, TAU_0P, TAU_0E)
    tau_atm = optical_depth.calc_tau_atm(p, tau_s, f_l)
    lw_net, lw_u, lw_d = two_stream.solve_two_stream(p, temp_atm, tau_atm)

    sw_net = solar_radiation.calc_solar_flux(lat=lats)

    rad_net = lw_net + sw_net

    return rad_net


def get_div_jr(rad_net, dx, dz):

    dr_dx = fdm.fdm_1_c(rad_net, axis=1, periodic=True, dx=dx)
    dr_dz = fdm.fdm_1_c(
        rad_net,
        axis=0,
        bc_left_type="dirichlet",
        bc_left_val=rad_net[0, :],
        bc_right_type="dirichlet",
        bc_right_val=rad_net[-1, :],
        periodic=False,
        dx=dz,
    )

    # Advection:
    div_jr = dr_dx + dr_dz
    return div_jr
