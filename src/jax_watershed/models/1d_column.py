"""
One-dimensional column-based biogeochemical modeling, including:
- surface/subsurface energy balance;
- surface/subsurface water mass balance;
- surface/subsurface carbon/nutrient cycle.

Author: Peishi Jiang
Date: 2023.03.23.
"""

# TODO: Let's first write it in a way that directly takes in the example forcing data. Further modifications are needed.

import jax
import jax.numpy as jnp

# from diffrax import NewtonNonlinearSolver

# from jax_watershed.physics.energy_fluxes.surface_energy import ground_energy_balance, leaf_energy_balance, calculate_Ts_from_TvTgTa, calculate_qs_from_qvqgqa
# from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_fluxes, calculate_longwave_fluxes
# from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_elevation
# from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_ground_albedos, calculate_ground_vegetation_emissivity
# from jax_watershed.physics.energy_fluxes.radiative_transfer import main_calculate_solar_fluxes
# from jax_watershed.physics.energy_fluxes.turbulent_fluxes import *
# from jax_watershed.physics.water_fluxes import q_from_e_pres, esat_from_temp, qsat_from_temp_pres

# from jax_watershed.shared_utilities.constants import R_DA as Rda
# from jax_watershed.shared_utilities.constants import C_TO_K as c2k

from jax_watershed.physics.energy_fluxes.surface_energy import solve_surface_energy
from jax_watershed.shared_utilities.forcings import ushn2_forcings
from jax_watershed.shared_utilities.domain import Time, Column
from jax_watershed.subjects import Soil, Surface
# from ..shared_utilities.forcings import ushn2_forcings
# from ..shared_utilities.domain import Time, Column
# from ..subjects import Soil, Surface


# TODO: Check the data types!
# Set float64
# from jax.config import config; config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------- #
#                           Model parameter settings                           #
# ---------------------------------------------------------------------------- #
# Spatio-temporal information/discretizations
t0, tn, dt = 0., 200., 1. # [day]
z0, zn, nz = 0.05, 10.-0.05, 19 # [m]
latitude, longitude, zone  = 31.31, -120.77, 8

# Surface characteristics
pft_ind = 10
f_snow, f_cansno = 0., 0.
z_a, z0m, z0c, d = 2.5, 0.05, 0.05, 0.05 
gsoil, gstomatal = 1e10, 1./180.

# Subsurface characteristics
κ = 0.05
dz_soil1 = z0


# ---------------------------------------------------------------------------- #
#                               Read forcing data                              #
# ---------------------------------------------------------------------------- #
forcings                       = ushn2_forcings
# Get the indices of different focings
forcing_list                   = forcings.varn_list
solar_rad_ind, L_down_ind      = forcing_list.index('SW_IN'), forcing_list.index('LW_IN')
L_ind, u_a_ind, q_a_ind        = forcing_list.index('LAI'), forcing_list.index('WS'), forcing_list.index('SH')
T_a_ind, pres_a_ind, ρ_atm_ind = forcing_list.index('TA'), forcing_list.index('PA'), forcing_list.index('ROA')


# ---------------------------------------------------------------------------- #
#                            Initialize the subjects                           #
# ---------------------------------------------------------------------------- #
time = Time(t0=t0, tn=tn, dt=dt, start_time='2016-01-05 12:00:00')
column = Column(xs=jnp.linspace(z0, zn, nz))
surface = Surface(ts=time, space=Column(xs=jnp.array([0.])))
soil    = Surface(ts=time, space=column)


# ---------------------------------------------------------------------------- #
#                     Numerically solve the model over time                    #
# ---------------------------------------------------------------------------- #
t_prev, t_now  = t0, t0  # t_now == t_prev for the initial step
tind_prev, tind_now = 0, 0

# JIT the function
solve_surface_energy_jit = jax.jit(solve_surface_energy)

while t_now < tn:
    # ------------------------- Get the current time step ------------------------ #
    t_now_fmt       = time.return_formatted_time(t_now)
    year, day, hour = t_now_fmt.year, t_now_fmt.timetuple().tm_yday , t_now_fmt.hour
    # year, day, hour = t_now_fmt.year, t_now_fmt.day, t_now_fmt.hour
    # hour = 12
     
    # Get the forcing data
    forcing_now                 = forcings.interpolate_time(t_now)
    solar_rad_t2, L_down_t2     = forcing_now[solar_rad_ind], forcing_now[L_down_ind]
    L_t2, u_a_t2, q_a_t2        = forcing_now[L_ind], forcing_now[u_a_ind], forcing_now[q_a_ind]
    T_a_t2, pres_a_t2, ρ_atm_t2 = forcing_now[T_a_ind], forcing_now[pres_a_ind], forcing_now[ρ_atm_ind]
    S_t2                        = 0. # TODO: Assume the stem area index zero

    # - Get the necessary model parameters/states at the current/next time steps - #
    l_t1           = surface.states['l'][tind_prev,0]
    T_v_t1, T_g_t1 = surface.states['T_v'][tind_prev,0], surface.states['T_g'][tind_prev,0]
    # T_soil1_t1     = T_g_t1  # TODO: replace it with the real first layer soil temperature

    l_guess, T_v_t2_guess, T_g_t2_guess = l_t1, T_v_t1, T_g_t1

    # ----------------------------- Evolve the model ----------------------------- #
    # -------------------------- 1. Solve surface energy ------------------------- #
    # l, T_v_t2, T_g_t2, S_v_t2, S_g_t2, L_v_t2, L_g_t2, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2 = solve_surface_energy_jit(
    # l, T_v_t2, T_g_t2, S_v_t2, S_g_t2, L_v_t2, L_g_t2, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2 = solve_surface_energy(
    l, T_v_t2, S_v_t2, S_g_t2, L_v_t2, L_g_t2, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2 = solve_surface_energy_jit(
        l_guess=-1.,
        # l_guess=l_guess,
        longitude=longitude, latitude=latitude, year=year, day=day, hour=hour, zone=zone,
        f_snow=f_snow, f_cansno=f_cansno, pft_ind=pft_ind,
        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil, 
        solar_rad_t2=solar_rad_t2, L_down_t2=L_down_t2, L_t2=L_t2, S_t2=S_t2, 
        u_a_t2=u_a_t2, q_a_t2=q_a_t2, T_a_t2=T_a_t2, pres_a_t2=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
        T_v_t1=T_v_t1, T_v_t2_guess=T_v_t2_guess, T_g_t1=T_g_t1, T_g_t2=T_g_t2_guess,
        # T_soil1_t1=T_soil1_t1, κ=κ, dz=dz_soil1,
    )

    # -------------------- 2. Solve subsurface energy (e.g., ) ------------------- #


    # ----------------------- 3. Adjust the surface energy ----------------------- #


    # --------------------------- Do some printing here -------------------------- #
    print("Time: {}".format(t_now))
    args=dict(
        # l_guess=l_guess,
        l_guess=-1,
        longitude=longitude, latitude=latitude, year=year, day=day, hour=hour, zone=zone,
        f_snow=f_snow, f_cansno=f_cansno, pft_ind=pft_ind,
        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil, 
        solar_rad_t2=solar_rad_t2, L_down_t2=L_down_t2, L_t2=L_t2, S_t2=S_t2, 
        u_a_t2=u_a_t2, q_a_t2=q_a_t2, T_a_t2=T_a_t2, pres_a_t2=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
        T_v_t1=T_v_t1, T_v_t2_guess=T_v_t2_guess, T_g_t1=T_g_t1, T_g_t2=T_g_t2_guess,
        # T_soil1_t1=T_soil1_t1, κ=κ, dz=dz_soil1,
    )
    # print(args)

    # break
    if jnp.isnan(l):
        break
    else:
        # print(l_guess, T_v_t2_guess, T_g_t2_guess)
        print("Updated states: {}".format([l, T_v_t2,]))
        print("")

    # Update the model state
    # TODO: make the following cleaner
    surface.set_state_value(state_name='l', time_ind=tind_now, space_ind=0, value=l)
    surface.set_state_value(state_name='T_v', time_ind=tind_now, space_ind=0, value=T_v_t2)
    # surface.set_state_value(state_name='T_g', time_ind=tind_now, space_ind=0, value=T_g_t2)

    # Update the time step
    t_prev = t_now
    t_now = min(t_now + dt, tn)

    # Update the time indices
    tind_prev = tind_now
    tind_now = min(tind_now + 1, time.nt)


# def solve_surface_energy(
#     l_guess,
#     longitude, latitude, year, day, hour, zone,
#     f_snow, f_cansno, pft_ind,
#     z_a, z0m, z0c, d, gstomatal, gsoil, 
#     solar_rad_t2, L_down_t2, L_t2, S_t2, 
#     u_a_t2, q_a_t2, T_a_t2, pres_a_t2, ρ_atm_t2, 
#     T_v_t1, T_v_t2_guess, T_g_t1, T_g_t2_guess,
#     T_soil1_t1, κ, dz,
#     atol=1e-5, rtol=1e-7
# ) -> tuple:
#     # Calculate solar elevation angle
#     solar_elev_angle = calculate_solar_elevation(
#         latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
#         zone=zone, is_day_saving=False
#     )

#     # # Calculate the air density
#     # e_a = (pres_a_t2*1e3 * q_a_t2) / (0.622 + 0.378 * q_a_t2) # [Pa]
#     # ρ_atm = (pres_a_t2*1e3 - 0.378*e_a) / (Rda * T_a_t2) # [kg m-3]
#     # print(e_a, ρ_atm)

#     # Calculate the albedos and emissivities
#     α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
#     α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
#     ε_g, ε_v                = calculate_ground_vegetation_emissivity(
#         solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind
#         )

#     # Calculate the solar radiation fluxes reaching the canopy
#     S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
#         solar_rad=solar_rad_t2, pres=pres_a_t2, solar_elev_angle=solar_elev_angle, 
#         α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
#         f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
#     )

#     # Jointly estimate T_g_t2, T_v_t2 and L
#     func = lambda x, args: calculate_ground_canopy_temp_L(x, **args)
#     args = dict(S_v=S_v, S_g=S_g, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
#             T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1, T_g_t2=T_g_t2_guess, 
#             T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, q_g_t2=q_g_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v, 
#             z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

#     solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
#     solution = solver(func, [l_guess, T_v_t2_guess, T_g_t2_guess], args=args)
#     l, T_v_t2, T_g_t2 = solution.root

#     # Update the longwave radiatios
#     L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
#         L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g, 
#         T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
#         L=L_t2, S=S_t2
#     )

#     # Monin-Ob similarity theory (MOST)
#     ψm_a  = calculate_ψm_most(ζ=z_a-d / l)
#     ψm_s  = calculate_ψm_most(ζ=z0m / l)
#     ψc_a  = calculate_ψc_most(ζ=z_a-d / l)
#     ψc_s  = calculate_ψc_most(ζ=z0c / l)
#     ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

#     # Calculate the conductances of heat and water vapor
#     # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
#     gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
#     gaw = gam
#     gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L_t2, S=S_t2)    
#     gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
#     ggm = calculate_conductance_ground_canopy(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m)
#     ggw = calculate_conductance_ground_canopy_water_vapo(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m, gsoil=gsoil)
#     # print(gvw, gvm)

#     # Calculate the saturated specific humidity from temperature and pressure
#     e_v_sat_t2 = esat_from_temp(T=T_v_t2)
#     q_v_sat_t2 = q_from_e_pres(pres=pres_a_t2, e=e_v_sat_t2)

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
#     q_g_t2     = q_g_t2_sat

#     # Calculate the temperature and specific humidity of the canopy air/surface
#     T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)
#     q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)
#     # print(T_s_t2, q_s_t2)

#     # Calculate the updated Obukhov length
#     tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]
#     qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]
#     # Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
#     # Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5

#     # Update the canopy fluxes 
#     H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
#     E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]


#     # Update the ground fluxes
#     H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
#     E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
#     G   = calculate_G(T_g=T_g_t2, T_s1=T_soil1_t1, κ=κ, dz=dz) 

#     return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


# def calculate_ground_canopy_temp_L(
#     x, S_v, S_g, L_down, pres, ρ_atm_t2,
#     T_soil1_t1, κ, dz,
#     T_v_t1, T_g_t1, T_g_t2, T_a_t2, 
#     u_a_t2, q_a_t2, q_g_t2, L, S, ε_g, ε_v,
#     z_a, z0m, z0c, d, gstomatal, gsoil):

#     # T_v_t2 = x 
#     L_guess, T_v_t2, T_g_t2 = x 

#     # Calculate the longwave radiation absorbed by the leaf/canopy
#     L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
#         L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
#         T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
#         L=L, S=S
#     )

#     # Monin-Ob similarity theory (MOST)
#     ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
#     ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
#     ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
#     ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
#     ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

#     # Calculate the conductances of heat and water vapor
#     # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
#     gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
#     gaw = gam
#     gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)    
#     gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
#     ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
#     ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)
#     # print(gvw, gvm)

#     # Calculate the saturated specific humidity from temperature and pressure
#     e_v_sat_t2 = esat_from_temp(T=T_v_t2)
#     q_v_sat_t2 = q_from_e_pres(pres=pres, e=e_v_sat_t2)

#     # Calculate the temperature and specific humidity of the canopy air/surface
#     T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)
#     q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)
#     # print(T_s_t2, q_s_t2)

#     # Calculate the updated Obukhov length
#     tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]
#     qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]
#     Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
#     Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5
#     # jax.debug.print("{}", jnp.array([ustar, tstar, qstar, Tzv, Tvstar]))
#     L_est  = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)
#     dL     = L_est - L_guess

#     # Solve the energy balance to estimate the vegetation temperature
#     denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)

#     # Solve the energy balance to estimate the ground temperature
#     denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)

#     return jnp.array([dL, denergy_v, denergy_g])