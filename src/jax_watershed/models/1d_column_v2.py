"""
One-dimensional column-based biogeochemical modeling, including:
- surface/subsurface energy balance;
- surface/subsurface water mass balance;
- surface/subsurface carbon/nutrient cycle.

Author: Peishi Jiang
Date: 2023.03.23.
"""

# TODO: Let's first write it in a way that directly takes in the example forcing data.
# Further modifications are needed.

import jax
import jax.numpy as jnp

# from diffrax import NewtonNonlinearSolver

# from jax_watershed.shared_utilities.constants import R_DA as Rda
# from jax_watershed.shared_utilities.constants import C_TO_K as c2k

# from jax_watershed.physics.energy_fluxes.surface_energy import solve_surface_energy
from jax_watershed.physics.energy_fluxes.surface_energy import (
    calculate_surface_energy_fluxes,
    solve_canopy_energy,
    # solve_surface_energy_canopy_ground,
)
from jax_watershed.physics.energy_fluxes.subsurface_energy import (
    # solve_subsurface_energy,
    solve_subsurface_energy_varyingG,
)

# from jax_watershed.shared_utilities.forcings import ushn2_forcings_daily
from jax_watershed.shared_utilities.forcings import ushn2_forcings_30min
from jax_watershed.shared_utilities.domain import Time, Column
from jax_watershed.subjects import Surface, Soil

# from ..shared_utilities.forcings import ushn2_forcings
# from ..shared_utilities.domain import Time, Column
# from ..subjects import Soil, Surface
# from diffrax import diffeqsolve, ODETerm, Dopri5


# TODO: Check the data types!
# Set float64
# from jax.config import config
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------- #
#                           Model parameter settings                           #
# ---------------------------------------------------------------------------- #
# Spatio-temporal information/discretizations
t0, tn, dt = 0.0, 200.0, 1.0 / 24.0  # [day]
# t0, tn, dt = 0.0, 200.0, 1.  # [day]
z0, zn, nz = 0.0, 10.0, 21  # [m]
latitude, longitude, zone = 31.31, -120.77, 8

# Surface characteristics
pft_ind = 10
f_snow, f_cansno = 0.0, 0.0
z_a, z0m, z0c, d = 2.5, 0.05, 0.05, 0.05
gsoil, gstomatal = 1e10, 1.0 / 180.0

# Subsurface characteristics
# κ = 0.05
# dz_soil1 = z0


# ---------------------------------------------------------------------------- #
#                               Read forcing data                              #
# ---------------------------------------------------------------------------- #
forcings = ushn2_forcings_30min
# forcings = ushn2_forcings_daily
# Get the indices of different focings
forcing_list = forcings.varn_list
solar_rad_ind, L_down_ind = forcing_list.index("SW_IN"), forcing_list.index("LW_IN")
L_ind, u_a_ind, q_a_ind = (
    forcing_list.index("LAI"),
    forcing_list.index("WS"),
    forcing_list.index("SH"),
)
T_a_ind, pres_a_ind, ρ_atm_ind = (
    forcing_list.index("TA"),
    forcing_list.index("PA"),
    forcing_list.index("ROA"),
)


# ---------------------------------------------------------------------------- #
#                            Initialize the subjects                           #
# ---------------------------------------------------------------------------- #
time = Time(t0=t0, tn=tn, dt=dt, start_time="2016-01-05 12:00:00")
soil_column = Column(xs=jnp.linspace(z0, zn, nz))
Δz = soil_column.Δx

surface = Surface(ts=time, space=Column(xs=soil_column.xs[:2]))
soil = Soil(ts=time, space=soil_column)


# ---------------------------------------------------------------------------- #
#                     Numerically solve the model over time                    #
# ---------------------------------------------------------------------------- #
t_prev, t_now = t0, t0 + dt  # t_now == t_prev for the initial step
tind_prev, tind_now = 0, 0

# JIT the functions
# solve_surface_energy_jit = jax.jit(solve_surface_energy)
# solve_surface_energy_jit = jax.jit(solve_surface_energy_canopy_ground)
# solve_surface_energy_jit = jax.jit(solve_surface_energy_canopy_ground_clm)
# solve_subsurface_energy_jit = jax.jit(solve_subsurface_energy)
solve_surface_energy_jit = jax.jit(solve_canopy_energy)
solve_subsurface_energy_jit = jax.jit(solve_subsurface_energy_varyingG)
calculate_surface_energy_fluxes_jit = jax.jit(calculate_surface_energy_fluxes)

while t_now < tn:
    # ------------------------- Get the current time step ------------------------ #
    t_now_fmt = time.return_formatted_time(t_now)
    year, day, hour = t_now_fmt.year, t_now_fmt.timetuple().tm_yday, t_now_fmt.hour
    # year, day, hour = t_now_fmt.year, t_now_fmt.day, t_now_fmt.hour
    # hour = 12

    # Get the forcing data
    forcing_now = forcings.interpolate_time(t_now)
    solar_rad_t2, L_down_t2 = forcing_now[solar_rad_ind], forcing_now[L_down_ind]
    L_t2, u_a_t2, q_a_t2 = (
        forcing_now[L_ind],
        forcing_now[u_a_ind],
        forcing_now[q_a_ind],
    )
    T_a_t2, pres_a_t2, ρ_atm_t2 = (
        forcing_now[T_a_ind],
        forcing_now[pres_a_ind],
        forcing_now[ρ_atm_ind],
    )
    S_t2 = 0.0  # TODO: Assume the stem area index zero

    # - Get the necessary model parameters/states at the current/next time steps - #
    l_t1 = surface.states["l"][tind_prev, 0]  # pyright: ignore
    T_v_t1, T_g_t1 = (
        surface.states["T_v"][tind_prev, 0],  # pyright: ignore
        surface.states["T_g"][tind_prev, 0],  # pyright: ignore
    )
    # T_soil1_t1 = T_g_t1  # TODO: replace it with the real first layer soil temperature
    Tsoil_t1 = soil.states["Tsoil"][tind_prev]
    T_soil1_t1 = Tsoil_t1[0]
    κ_soil1 = soil.parameters["κ"][0]
    dz_soil1 = soil_column.Δx[0]

    l_guess, T_v_t2_guess, T_g_t2_guess = l_t1, T_v_t1, T_g_t1
    # print()

    # ----------------------------- Evolve the model ----------------------------- #
    # -------------------------- 1. Solve surface energy ------------------------- #
    (
        l_t2,
        T_v_t2,
        ε_g,
        ε_v,
        S_v_t2,
        S_g_t2,
        # ) = solve_canopy_energy(
    ) = solve_surface_energy_jit(
        l_guess=-10.0,
        longitude=longitude,
        latitude=latitude,
        year=year,
        day=day,
        hour=hour,
        zone=zone,
        f_snow=f_snow,
        f_cansno=f_cansno,
        pft_ind=pft_ind,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
        solar_rad_t2=solar_rad_t2,
        L_down_t2=L_down_t2,
        L_t2=L_t2,
        S_t2=S_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        T_a_t2=T_a_t2,
        pres_a_t2=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_v_t2_guess=T_v_t2_guess,
        T_g_t1=T_g_t1,
        T_g_t2_guess=T_g_t2_guess,
    )

    # -------------------- 2. Solve subsurface energy (e.g., ) ------------------- #
    Tsoil_t2 = solve_subsurface_energy_jit(
        # Tsoil_t2 = solve_subsurface_energy_varyingG(
        Tsoil=Tsoil_t1,
        κ=soil.parameters["κ"],
        cv=soil.parameters["cv"],
        Δz=Δz,
        Δt=dt,
        l=l_t2,
        S_g=S_g_t2,
        L_down=L_down_t2,
        pres=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        L=L_t2,
        S=S_t2,
        ε_g=ε_g,
        ε_v=ε_v,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )
    T_g_t2 = Tsoil_t2[0]

    # ----------------------- 3. Adjust the surface energy fluxes ------------------- #
    # L_v_t2, L_g_t2, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2 = calculate_surface_energy_fluxes( # noqa: E501
    (
        L_v_t2,
        L_g_t2,
        H_v_t2,
        H_g_t2,
        E_v_t2,
        E_g_t2,
        G_t2,
    ) = calculate_surface_energy_fluxes_jit(  # noqa: E501
        l=l_t2,
        S_v_t2=S_v_t2,
        S_g_t2=S_g_t2,
        L_down_t2=L_down_t2,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        pres_a_t2=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        L_t2=L_t2,
        S_t2=S_t2,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    # --------------------------- Do some printing here -------------------------- #
    print("Time: {}".format(t_now))
    args = dict(
        # l_guess=l_guess,
        l_guess=-1,
        longitude=longitude,
        latitude=latitude,
        year=year,
        day=day,
        hour=hour,
        zone=zone,
        f_snow=f_snow,
        f_cansno=f_cansno,
        pft_ind=pft_ind,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
        solar_rad_t2=solar_rad_t2,
        L_down_t2=L_down_t2,
        L_t2=L_t2,
        S_t2=S_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        T_a_t2=T_a_t2,
        pres_a_t2=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_v_t2_guess=T_v_t2_guess,
        T_g_t1=T_g_t1,
        T_g_t2_guess=T_g_t2_guess,
    )
    # print(args)

    # break
    # if jnp.isnan(l_t2):
    # if t_now > 106:
    if T_g_t2 > 400:
        break
    else:
        # print(l_guess, T_v_t2_guess, T_g_t2_guess)
        print("Air temperature: {}".format(T_a_t2))
        print("Updated states: {}".format([l_t2, T_v_t2, T_g_t2, Tsoil_t2]))
        print("")

    # Update the model state
    # TODO: make the following cleaner
    surface.set_state_value(state_name="l", time_ind=tind_now, space_ind=0, value=l_t2)
    surface.set_state_value(
        state_name="T_v", time_ind=tind_now, space_ind=0, value=T_v_t2
    )
    surface.set_state_value(
        state_name="T_g", time_ind=tind_now, space_ind=0, value=T_g_t2
    )
    soil.set_state_value(state_name="Tsoil", time_ind=tind_now, value=Tsoil_t2)

    # Update the time step
    t_prev = t_now
    t_now = min(t_now + dt, tn)

    # Update the time indices
    tind_prev = tind_now
    tind_now = min(tind_now + 1, time.nt)
