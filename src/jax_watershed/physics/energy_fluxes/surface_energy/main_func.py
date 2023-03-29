"""
These are some main functions for solving surface energy balances.

Author: Peishi Jiang
Date: 2023.03.24

"""

import jax
import jax.numpy as jnp

from diffrax import NewtonNonlinearSolver

from .monin_obukhov import perform_most_dual_source, func_most_dual_source
from .canopy_energy import leaf_energy_balance
from .ground_energy import ground_energy_balance
from .surface_state import calculate_Ts_from_TvTgTa, calculate_qs_from_qvqgqa
from ..radiative_transfer import calculate_solar_fluxes, calculate_longwave_fluxes
from ..radiative_transfer import calculate_solar_elevation
from ..radiative_transfer import calculate_ground_albedos, calculate_ground_vegetation_emissivity
from ..radiative_transfer import main_calculate_solar_fluxes
from ..turbulent_fluxes import *
from ...water_fluxes import q_from_e_pres, esat_from_temp, qsat_from_temp_pres

from ....shared_utilities.constants import R_DA as Rda
from ....shared_utilities.constants import C_TO_K as c2k

# TODO: Add documentations
# TODO: Further modulize the functions

def solve_surface_energy(
    l_guess,
    longitude, latitude, year, day, hour, zone,
    f_snow, f_cansno, pft_ind,
    z_a, z0m, z0c, d, gstomatal, gsoil, 
    solar_rad_t2, L_down_t2, L_t2, S_t2, 
    u_a_t2, q_a_t2, T_a_t2, pres_a_t2, ρ_atm_t2, 
    T_v_t1, T_v_t2_guess, T_g_t1, T_g_t2_guess,
    atol=1e-1, rtol=1e-3
    # atol=1e-5, rtol=1e-7
) -> tuple:
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
        zone=zone, is_day_saving=False
    )
    # jax.debug.print("Solar elevation angle: {}", solar_elev_angle)

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
    ε_g, ε_v                = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind
        )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2, pres=pres_a_t2*1e-3, solar_elev_angle=solar_elev_angle, 
        α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
    )

    # Estimate the ground temperature T_g_t2
    func = lambda x, args: residual_canopy_temp(x, **args)
    args = dict(T_g_t2=T_g_t2_guess, l_guess=l_guess, S_v=S_v, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
            T_v_t1=T_v_t1, T_g_t1=T_g_t1, T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v, 
            z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))
    solution = solver(func, T_v_t2_guess, args=args)
    T_v_t2 = solution.root
    # T_g_t2 = T_g_t2_guess

    # Update the longwave radiatios
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2_guess,
        L=L_t2, S=S_t2
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2_guess, pres=pres_a_t2)
    q_g_t2     = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    l, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(
       L_guess=l_guess, pres=pres_a_t2, T_v=T_v_t2, T_g=T_g_t2_guess, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L_t2, S=S_t2,
       z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil 
    )

    # Update the canopy fluxes 
    H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
    E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]

    # Update the ground fluxes
    H_g = calculate_H(T_1=T_g_t2_guess, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    # G   = calculate_G(T_g=T_g_t2_guess, T_s1=T_soil1_t1, κ=κ, dz=dz) 
    G   = S_g - L_g - H_g - E_g

    return l, T_v_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


def residual_canopy_temp(
    x, T_g_t2, l_guess, S_v, L_down, pres, ρ_atm_t2,
    T_v_t1, T_g_t1, T_a_t2, 
    u_a_t2, q_a_t2, L, S, ε_g, ε_v,
    z_a, z0m, z0c, d, gstomatal, gsoil):
 
    T_v_t2 = x 

    # Calculate the longwave radiation absorbed by the leaf/canopy
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L, S=S
    )
    # jax.debug.print("L_v and L_g: {}", jnp.array([L_v, L_g]))

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    q_g_t2     = q_g_t2_sat

    # Solve Monin-Obukhov length
    kwarg = dict(
        pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,
        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
    )
    func = lambda L, args: func_most_dual_source(L, **args)
    solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7)
    solution = solver(func, l_guess, args=kwarg)
    L_update = solution.root

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    _, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(
       L_guess=L_update, pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,
       z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil 
    )

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)

    # # Solve the energy balance to estimate the ground temperature
    # denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)

    # return jnp.array([denergy_v, denergy_g])
    return denergy_v

def solve_surface_energy_v2(
    l_guess,
    longitude, latitude, year, day, hour, zone,
    f_snow, f_cansno, pft_ind,
    z_a, z0m, z0c, d, gstomatal, gsoil, 
    solar_rad_t2, L_down_t2, L_t2, S_t2, 
    u_a_t2, q_a_t2, T_a_t2, pres_a_t2, ρ_atm_t2, 
    T_v_t1, T_v_t2_guess, T_g_t1, T_g_t2_guess,
    T_soil1_t1, κ, dz,
    atol=1e-5, rtol=1e-7
) -> tuple:
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
        zone=zone, is_day_saving=False
    )

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
    ε_g, ε_v                = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind
        )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2, pres=pres_a_t2*1e-3, solar_elev_angle=solar_elev_angle, 
        α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
    )

    # Jointly estimate T_g_t2, T_v_t2
    func = lambda x, args: residual_ground_canopy_temp(x, **args)
    args = dict(l_guess=l_guess, S_v=S_v, S_g=S_g, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
            T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1, 
            T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v, 
            z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))
    solution = solver(func, [T_v_t2_guess, T_g_t2_guess], args=args)
    T_v_t2, T_g_t2 = solution.root
    # solution = solver(func, T_v_t2_guess, args=args)
    # T_v_t2 = solution.root
    # T_g_t2 = T_g_t2_guess

    # Update the longwave radiatios
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L_t2, S=S_t2
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
    q_g_t2     = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    l, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(
       L_guess=l_guess, pres=pres_a_t2, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L_t2, S=S_t2,
       z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil 
    )

    # Update the canopy fluxes 
    H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
    E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]

    # Update the ground fluxes
    H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    G   = calculate_G(T_g=T_g_t2, T_s1=T_soil1_t1, κ=κ, dz=dz) 

    return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


def residual_ground_canopy_temp(
    x, l_guess, S_v, S_g, L_down, pres, ρ_atm_t2,
    T_soil1_t1, κ, dz,
    T_v_t1, T_g_t1, T_a_t2, 
    u_a_t2, q_a_t2, L, S, ε_g, ε_v,
    z_a, z0m, z0c, d, gstomatal, gsoil):
 
    T_v_t2, T_g_t2 = x 
    # T_v_t2 = x 

    # Calculate the longwave radiation absorbed by the leaf/canopy
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L, S=S
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    q_g_t2     = q_g_t2_sat

    # Solve Monin-Obukhov length
    kwarg = dict(
        pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,
        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
    )
    func = lambda L, args: func_most_dual_source(L, **args)
    solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7)
    solution = solver(func, l_guess, args=kwarg)
    L_update = solution.root
    jax.debug.print("Updated L: {}", L_update)
    jax.debug.print("Updated T_g and T_v: {}", x)

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    _, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(
       L_guess=L_update, pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,
       z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil 
    )

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)

    # Solve the energy balance to estimate the ground temperature
    denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)

    jax.debug.print("Updated denergy_v: {}", denergy_v)
    jax.debug.print("Updated denergy_g: {}", denergy_g)

    return jnp.array([denergy_v, denergy_g])
    # return denergy_v


def solve_surface_energy_v3(
    l_guess,
    longitude, latitude, year, day, hour, zone,
    f_snow, f_cansno, pft_ind,
    z_a, z0m, z0c, d, gstomatal, gsoil, 
    solar_rad_t2, L_down_t2, L_t2, S_t2, 
    u_a_t2, q_a_t2, T_a_t2, pres_a_t2, ρ_atm_t2, 
    T_v_t1, T_v_t2_guess, T_g_t1, T_g_t2_guess,
    T_soil1_t1, κ, dz,
    atol=1e-5, rtol=1e-7
) -> tuple:
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
        zone=zone, is_day_saving=False
    )

    # # Calculate the air density
    # e_a = (pres_a_t2*1e3 * q_a_t2) / (0.622 + 0.378 * q_a_t2) # [Pa]
    # ρ_atm = (pres_a_t2*1e3 - 0.378*e_a) / (Rda * T_a_t2) # [kg m-3]
    # print(e_a, ρ_atm)

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
    ε_g, ε_v                = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind
        )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2, pres=pres_a_t2*1e-3, solar_elev_angle=solar_elev_angle, 
        α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
    )

    # Jointly estimate T_g_t2, T_v_t2 and L
    func = lambda x, args: residual_ground_canopy_temp_L(x, **args)
    args = dict(S_v=S_v, S_g=S_g, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2, 
            T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1, 
            T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v, 
            z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol, max_steps=1000)
    solution = solver(func, [l_guess, T_v_t2_guess, T_g_t2_guess], args=args)
    l, T_v_t2, T_g_t2 = solution.root

    # Update the longwave radiatios
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L_t2, S=S_t2
    )

    # Monin-Ob similarity theory (MOST)
    ψm_a  = calculate_ψm_most(ζ=z_a-d / l)
    ψm_s  = calculate_ψm_most(ζ=z0m / l)
    ψc_a  = calculate_ψc_most(ζ=z_a-d / l)
    ψc_s  = calculate_ψc_most(ζ=z0c / l)
    ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

    # Calculate the conductances of heat and water vapor
    # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
    gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
    gaw = gam
    gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L_t2, S=S_t2)    
    gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
    ggm = calculate_conductance_ground_canopy(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m)
    ggw = calculate_conductance_ground_canopy_water_vapo(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m, gsoil=gsoil)
    # print(gvw, gvm)

    # Calculate the saturated specific humidity from temperature and pressure
    e_v_sat_t2 = esat_from_temp(T=T_v_t2)
    q_v_sat_t2 = q_from_e_pres(pres=pres_a_t2, e=e_v_sat_t2)

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
    q_g_t2     = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    # Calculate the temperature and specific humidity of the canopy air/surface
    T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)
    q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)
    # print(T_s_t2, q_s_t2)

    # Calculate the updated Obukhov length
    tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]
    qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]
    # Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
    # Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5

    # Update the canopy fluxes 
    H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
    E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]

    # Update the ground fluxes
    H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    G   = calculate_G(T_g=T_g_t2, T_s1=T_soil1_t1, κ=κ, dz=dz) 

    return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


def residual_ground_canopy_temp_L(
    x, S_v, S_g, L_down, pres, ρ_atm_t2,
    T_soil1_t1, κ, dz,
    T_v_t1, T_g_t1, T_a_t2, 
    u_a_t2, q_a_t2, L, S, ε_g, ε_v,
    z_a, z0m, z0c, d, gstomatal, gsoil):

    # T_v_t2 = x 
    L_guess, T_v_t2, T_g_t2 = x 

    # Calculate the longwave radiation absorbed by the leaf/canopy
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L, S=S
    )

    # Monin-Ob similarity theory (MOST)
    # jax.debug.print("The guessing L: {}", L_guess)
    ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
    ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
    ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
    ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
    ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)
    # TODO: check why ustar is negative. For now, we force it to be positive
    ustar = jnp.absolute(ustar)

    # Calculate the conductances of heat and water vapor
    # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
    gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
    gaw = gam
    gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)    
    gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
    ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
    ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)
    # jax.debug.print("ustar, gvw, ggw: {}", jnp.array([ustar, gvw, ggw]))
    # print(gvw, gvm)

    # Calculate the saturated specific humidity from temperature and pressure
    e_v_sat_t2 = esat_from_temp(T=T_v_t2)
    q_v_sat_t2 = q_from_e_pres(pres=pres, e=e_v_sat_t2)
    # jax.debug.print("q_v_sat_t2: {}", jnp.array([q_v_sat_t2, T_v_t2, pres, e_v_sat_t2]))

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    q_g_t2     = q_g_t2_sat

    # Calculate the temperature and specific humidity of the canopy air/surface
    T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)
    q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)
    # print(T_s_t2, q_s_t2)

    # Calculate the updated Obukhov length
    tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]
    qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]
    Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
    Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5
    L_est  = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)
    dL     = L_est - L_guess

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)

    # Solve the energy balance to estimate the ground temperature
    denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)

    jax.debug.print("Updated denergy_v: {}", denergy_v)
    jax.debug.print("Updated denergy_g: {}", denergy_g)

    return jnp.array([dL, denergy_v, denergy_g])