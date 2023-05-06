"""
These are some main functions for solving surface energy balances.

Author: Peishi Jiang
Date: 2023.03.24.

"""

import jax
import jax.numpy as jnp

from typing import Tuple
from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import λ_VAP as λ

from diffrax import NewtonNonlinearSolver

from .monin_obukhov import perform_most_dual_source, func_most_dual_source

from .monin_obukhov import calculate_initial_guess_L
from .canopy_energy import leaf_energy_balance
from .ground_energy import ground_energy_balance
from ..radiative_transfer import calculate_longwave_fluxes
from ..radiative_transfer import calculate_solar_elevation
from ..radiative_transfer import (
    calculate_ground_albedos,
    calculate_ground_vegetation_emissivity,
)
from ..radiative_transfer import main_calculate_solar_fluxes

# from ..turbulent_fluxes import *  # noqa: F403
from ..turbulent_fluxes import calculate_E, calculate_H
from ...water_fluxes import qsat_from_temp_pres


# TODO: Further modulize the functions


l_thres = 1


def solve_surface_energy(
    l_guess: Float_0D,
    longitude: Float_0D,
    latitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int,
    f_snow: Float_0D,
    f_cansno: Float_0D,
    pft_ind: int,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
    solar_rad_t2: Float_0D,
    L_down_t2: Float_0D,
    L_t2: Float_0D,
    S_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    T_a_t2: Float_0D,
    pres_a_t2: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2_guess: Float_0D,
    T_g_t1: Float_0D,
    T_g_t2: Float_0D,
    atol: Float_0D = 1e-1,
    rtol: Float_0D = 1e-3
    # atol=1e-5, rtol=1e-7
) -> Tuple:
    """Solve the surface energy by estimating the vegetation temperature, where the
       Obukhov length is solved at each iteration.

    Args:
        l_guess (Float_0D): The initial guess of the Obukhov length.
        latitude (Float_0D): The latitude [degree].
        longitude (Float_0D): The longitude [degree].
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        f_snow (Float_0D): The fraction of ground covered by snow [-]
        f_cansno (Float_0D): The canopy snow-covered fraction [-]
        pft_ind (int): The index of plant functional type based on the pft_clm5
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]
        solar_rad_t2 (Float_0D): The incoming solar radiation at the current time step t2 [W m-2]
        L_down_t2 (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        L_t2 (Float_0D): The leaf area index at the current time step t2 [m2 m-2]
        S_t2 (Float_0D): The steam area index at the current time step t2 [m2 m-2]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        pres_a_t2 (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2_guess (Float_0D): The guess of the vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_g_t2 (Float_0D): The ground temperature at the current time step t2 [degK]
        atol (Float_0D, optional): The absolute tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-1.
        rtol (Float_0D, optional): The relative tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-3.

    Returns:
        tuple: _description_
    """  # noqa: E501
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude,
        longitude=longitude,
        year=year,
        day=day,
        hour=hour,
        zone=zone,
        is_day_saving=False,
    )
    # jax.debug.print("Solar elevation angle: {}", solar_elev_angle)

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, "PAR")
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, "NIR")
    ε_g, ε_v = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle,
        f_snow=f_snow,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2,
        pres=pres_a_t2 * 1e-3,
        solar_elev_angle=solar_elev_angle,
        α_g_db_par=α_g_db_par,
        α_g_dif_par=α_g_dif_par,
        α_g_db_nir=α_g_db_nir,
        α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow,
        f_cansno=f_cansno,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )
    # jax.debug.print('solar radiation: {}', jnp.array([S_v, S_g]))

    # Estimate the canopy temperature T_v_t2
    def func(x, args):
        return residual_canopy_temp(x, **args)

    args = dict(
        T_g_t2=T_g_t2,
        l_guess=l_guess,
        S_v=S_v,
        L_down=L_down_t2,
        pres=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
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

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))  # noqa: E501
    solution = solver(func, T_v_t2_guess, args=args)
    T_v_t2 = solution.root
    # T_g_t2 = T_g_t2_guess

    # Update the longwave radiatios
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L_t2,
        S=S_t2,
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
    q_g_t2 = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    (
        l,  # noqa: E741
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l_guess,
        pres=pres_a_t2,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L_t2,
        S=S_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    # Update the canopy fluxes
    H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)
    E_v = calculate_E(
        q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]

    # Update the ground fluxes
    H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    # G   = calculate_G(T_g=T_g_t2_guess, T_s1=T_soil1_t1, κ=κ, dz=dz)
    G = S_g - L_g - H_g - E_g

    return l, T_v_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


def solve_canopy_energy(
    l_guess: Float_0D,
    longitude: Float_0D,
    latitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int,
    f_snow: Float_0D,
    f_cansno: Float_0D,
    pft_ind: int,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
    solar_rad_t2: Float_0D,
    L_down_t2: Float_0D,
    L_t2: Float_0D,
    S_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    T_a_t2: Float_0D,
    pres_a_t2: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2_guess: Float_0D,
    T_g_t1: Float_0D,
    T_g_t2_guess: Float_0D,
    atol: Float_0D = 1e-1,
    rtol: Float_0D = 1e-3
    # atol=1e-5, rtol=1e-7
) -> Tuple:
    """Solve the surface energy by estimating the vegetation temperature, where the
       Obukhov length is solved at each iteration.

    Args:
        l_guess (Float_0D): The initial guess of the Obukhov length.
        latitude (Float_0D): The latitude [degree].
        longitude (Float_0D): The longitude [degree].
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        f_snow (Float_0D): The fraction of ground covered by snow [-]
        f_cansno (Float_0D): The canopy snow-covered fraction [-]
        pft_ind (int): The index of plant functional type based on the pft_clm5
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]
        solar_rad_t2 (Float_0D): The incoming solar radiation at the current time step t2 [W m-2]
        L_down_t2 (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        L_t2 (Float_0D): The leaf area index at the current time step t2 [m2 m-2]
        S_t2 (Float_0D): The steam area index at the current time step t2 [m2 m-2]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        pres_a_t2 (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        ρ_atm_t2 (Float_0D): The air density at the reference height at the current time step t2 [kg m-3]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2_guess (Float_0D): The guess of the vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_g_t2_guess (Float_0D): The guess of the ground temperature at the current time step t2 [degK]
        atol (Float_0D, optional): The absolute tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-1.
        rtol (Float_0D, optional): The relative tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-3.

    Returns:
        tuple: _description_
    """  # noqa: E501
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude,
        longitude=longitude,
        year=year,
        day=day,
        hour=hour,
        zone=zone,
        is_day_saving=False,
    )
    # jax.debug.print("Solar elevation angle: {}", solar_elev_angle)

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, "PAR")
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, "NIR")
    ε_g, ε_v = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle,
        f_snow=f_snow,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2,
        pres=pres_a_t2 * 1e-3,
        solar_elev_angle=solar_elev_angle,
        α_g_db_par=α_g_db_par,
        α_g_dif_par=α_g_dif_par,
        α_g_db_nir=α_g_db_nir,
        α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow,
        f_cansno=f_cansno,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )
    # jax.debug.print('solar radiation: {}', jnp.array([S_v, S_g]))

    # ---------------------- Estimate the initial guess of L --------------------- #
    l_guess = calculate_initial_guess_L(
        T_a=T_a_t2,
        T_s=(T_a_t2 + T_g_t2_guess) / 2.0,
        q_a=q_a_t2,
        q_s=q_a_t2,
        u_a=u_a_t2,
        z_a=z_a,
        d=d,
        z0m=z0m,
    )
    # jax.debug.print("The initial guess of L: {}", l_guess)

    # ------------------ Estimate the canopy temperature T_v_t2 ------------------ #
    # jax.debug.print("Calculating T_v_t2 ...")
    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat_guess = qsat_from_temp_pres(T=T_g_t2_guess, pres=pres_a_t2)
    q_g_t2_guess = q_g_t2_sat_guess

    def func_canopy_temp(x, args):
        return residual_canopy_temp(x, **args)

    args = dict(
        T_g_t2=T_g_t2_guess,
        l_guess=l_guess,
        l_reset=l_guess,
        S_v=S_v,
        L_down=L_down_t2,
        pres=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_g_t1=T_g_t1,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        q_g_t2=q_g_t2_guess,
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

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))  # noqa: E501
    # solution = solver(func_canopy_temp, T_v_t2_guess, args=args)
    # T_v_t2 = solution.root

    def update_canopy_temperature(carry, x=None):
        T_v_t2_guess, args = carry
        # Solve the canopy temperature
        solution = solver(func_canopy_temp, T_v_t2_guess, args=args)
        T_v_t2_update = solution.root
        # Update the Obukhov length
        (
            l_update,
            gam,
            gaw,
            gvm,
            gvw,
            ggm,
            ggw,
            q_v_sat_t2,
            T_s_t2,
            q_s_t2,
        ) = perform_most_dual_source(
            L_guess=args["l_guess"],
            pres=args["pres"],
            T_v=T_v_t2_update,
            T_g=args["T_g_t2"],
            T_a=args["T_a_t2"],
            u_a=args["u_a_t2"],
            q_a=args["q_a_t2"],
            q_g=args["q_g_t2"],
            L=args["L"],
            S=args["S"],
            z_a=args["z_a"],
            z0m=args["z0m"],
            z0c=args["z0c"],
            d=args["d"],
            gstomatal=args["gstomatal"],
            gsoil=args["gsoil"],
        )
        new_args = args
        new_args["l_guess"] = l_update
        # return T_v_t2_update, new_args
        return [T_v_t2_update, new_args], [T_v_t2_update, l_update]

    carry, y_list = jax.lax.scan(
        update_canopy_temperature, init=[T_v_t2_guess, args], xs=None, length=10
    )
    T_v_t2, new_args = carry
    l = new_args["l_guess"]  # noqa: E741
    # jax.debug.print("The final new_args: {}", new_args)
    # jax.debug.print("The list of updated T_v_t2 and l : {}", y_list)

    # ---------------------- Update the longwave radiations ---------------------- #
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2_guess,
        L=L_t2,
        S=S_t2,
    )

    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    (
        _,  # noqa: E741
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l,
        pres=pres_a_t2,
        T_v=T_v_t2,
        T_g=T_g_t2_guess,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2_guess,
        L=L_t2,
        S=S_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )
    # jax.debug.print("Conductances: {}", jnp.array([gvm, gvw]))

    # ------------------------- Calculate the canopy fluxes ------------------------ #
    # print(gvm)
    H_v_t2 = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)
    E_v_t2 = calculate_E(
        q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]

    # ------------------------- Update the ground fluxes ------------------------- #
    H_g_t2 = calculate_H(T_1=T_g_t2_guess, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g_t2 = calculate_E(
        q_1=q_g_t2_guess, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw
    )  # [kg m-2 s-1]
    # G   = calculate_G(T_g=T_g_t2_guess, T_s1=T_soil1_t1, κ=κ, dz=dz)
    G_t2 = S_g - L_g - H_g_t2 - λ * E_g_t2

    # jax.debug.print(
    #     # "Canopy fluxes: {}", jnp.array([S_v, L_v, H_v_t2, λ * E_v_t2, gvm, gvw, T_v_t2, T_s_t2, q_v_sat_t2, q_s_t2])  # noqa: E501
    #     "Canopy fluxes: {}", jnp.array([S_v, L_v, H_v_t2, λ*E_v_t2, T_v_t2, T_s_t2])  # noqa: E501
    # )  # noqa: E501

    return l, T_v_t2, ε_g, ε_v, S_v, S_g, L_v, L_g, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2
    # return l, T_v_t2, ε_g, ε_v


def calculate_surface_energy_fluxes(
    l,  # noqa: E741
    S_v_t2,
    S_g_t2,
    L_down_t2,
    T_v_t1,
    T_v_t2,
    T_g_t1,
    T_g_t2,  # noqa: E741
    ε_v,
    ε_g,
    pres_a_t2,
    ρ_atm_t2,
    T_a_t2,
    u_a_t2,
    q_a_t2,
    z_a,
    z0m,
    z0c,
    d,
    L_t2,
    S_t2,
    gstomatal,
    gsoil,
):
    # ---------------------- Update the longwave radiations ---------------------- #
    L_v_t2, L_g_t2, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L_t2,
        S=S_t2,
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
    q_g_t2 = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    (
        _,
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l,
        pres=pres_a_t2,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L_t2,
        S=S_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )
    # jax.debug.print("Conductances: {}", jnp.array([gvm, gvw]))

    # ------------------------- Update the canopy fluxes ------------------------- #
    # print(gvm)
    H_v_t2 = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)
    E_v_t2 = calculate_E(
        q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]

    # ------------------------- Update the ground fluxes ------------------------- #
    H_g_t2 = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g_t2 = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    # G   = calculate_G(T_g=T_g_t2_guess, T_s1=T_soil1_t1, κ=κ, dz=dz)
    G_t2 = S_g_t2 - L_g_t2 - H_g_t2 - λ * E_g_t2

    # jax.debug.print("l, T_v_t2, T_g_t2: {}", jnp.array([l, T_v_t2, T_g_t2]))
    # jax.debug.print(
    #     "Canopy fluxes: {}", jnp.array([S_v_t2, L_v_t2, H_v_t2, λ * E_v_t2])
    # )  # noqa: E501
    # jax.debug.print(
    #     "Ground fluxes: {}", jnp.array([S_g_t2, L_g_t2, H_g_t2, λ * E_g_t2, G_t2])
    # )  # noqa: E501

    return L_v_t2, L_g_t2, H_v_t2, H_g_t2, E_v_t2, E_g_t2, G_t2


def solve_surface_energy_canopy_ground(
    l_guess: Float_0D,
    longitude: Float_0D,
    latitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int,
    f_snow: Float_0D,
    f_cansno: Float_0D,
    pft_ind: int,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
    solar_rad_t2: Float_0D,
    L_down_t2: Float_0D,
    L_t2: Float_0D,
    S_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    T_a_t2: Float_0D,
    pres_a_t2: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2_guess: Float_0D,
    T_g_t1: Float_0D,
    T_g_t2_guess: Float_0D,
    T_soil1_t1: Float_0D,
    κ: Float_0D,
    dz: Float_0D,
    atol: Float_0D = 1e-1,
    rtol: Float_0D = 1e-3
    # atol=1e-5, rtol=1e-7
) -> Tuple:
    """Solve the surface energy by estimating the vegetation temperature, where the
       Obukhov length is solved at each iteration.

    Args:
        l_guess (Float_0D): The initial guess of the Obukhov length.
        latitude (Float_0D): The latitude [degree].
        longitude (Float_0D): The longitude [degree].
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        f_snow (Float_0D): The fraction of ground covered by snow [-]
        f_cansno (Float_0D): The canopy snow-covered fraction [-]
        pft_ind (int): The index of plant functional type based on the pft_clm5
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]
        solar_rad_t2 (Float_0D): The incoming solar radiation at the current time step t2 [W m-2]
        L_down_t2 (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        L_t2 (Float_0D): The leaf area index at the current time step t2 [m2 m-2]
        S_t2 (Float_0D): The steam area index at the current time step t2 [m2 m-2]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        pres_a_t2 (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2_guess (Float_0D): The guess of the vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_g_t2_guess (Float_0D): The guess of the ground temperature at the current time step t2 [degK]
        T_soil1_t1 (Float_0D): The temperature of the first soil layer [degK]
        κ (Float_0D): the thermal conductivity [W m-1 K-1]
        dz (Float_0D): The soil depth of the first soil layer [m]
        atol (Float_0D, optional): The absolute tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-1.
        rtol (Float_0D, optional): The relative tolerance error used by diffrax.NewtonNonlinearSolver. Defaults to 1e-3.

    Returns:
        tuple: _description_
    """  # noqa: E501
    # Calculate solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude,
        longitude=longitude,
        year=year,
        day=day,
        hour=hour,
        zone=zone,
        is_day_saving=False,
    )
    # jax.debug.print("Solar elevation angle: {}", solar_elev_angle)

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, "PAR")
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, "NIR")
    ε_g, ε_v = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle,
        f_snow=f_snow,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )

    # Calculate the solar radiation fluxes reaching the canopy
    S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
        solar_rad=solar_rad_t2,
        pres=pres_a_t2 * 1e-3,
        solar_elev_angle=solar_elev_angle,
        α_g_db_par=α_g_db_par,
        α_g_dif_par=α_g_dif_par,
        α_g_db_nir=α_g_db_nir,
        α_g_dif_nir=α_g_dif_nir,
        f_snow=f_snow,
        f_cansno=f_cansno,
        L=L_t2,
        S=S_t2,
        pft_ind=pft_ind,
    )
    # jax.debug.print('solar radiation: {}', jnp.array([S_v, S_g]))

    # ---------------------- Estimate the initial guess of L --------------------- #
    l_guess = calculate_initial_guess_L(
        T_a=T_a_t2,
        T_s=(T_a_t2 + T_g_t2_guess) / 2.0,
        q_a=q_a_t2,
        q_s=q_a_t2,
        u_a=u_a_t2,
        z_a=z_a,
        d=d,
        z0m=z0m,
    )
    # jax.debug.print("The initial guess of L: {}", l_guess)

    # ------------------ Estimate the canopy temperature T_v_t2 ------------------ #
    # jax.debug.print("Calculating T_v_t2 ...")

    def func_canopy_temp(x, args):
        return residual_canopy_temp(x, **args)

    args = dict(
        T_g_t2=T_g_t2_guess,
        l_guess=l_guess,
        S_v=S_v,
        L_down=L_down_t2,
        pres=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
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

    solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))  # noqa: E501
    solution = solver(func_canopy_temp, T_v_t2_guess, args=args)
    T_v_t2 = solution.root

    # - Use the updated Obukhov to perform Monin Obukhov similarity theory again - #
    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2_guess, pres=pres_a_t2)
    q_g_t2_guess = q_g_t2_sat

    kwarg = dict(
        pres=pres_a_t2,
        T_v=T_v_t2,
        T_g=T_g_t2_guess,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2_guess,
        L=L_t2,
        S=S_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    def func(L, args):
        return func_most_dual_source(L, **args)

    solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7)
    solution = solver(func, l_guess, args=kwarg)

    l = solution.root  # noqa: E741
    l = jax.lax.cond(  # noqa: E741
        jnp.abs(l) > l_thres, lambda x: x, lambda x: l_thres * jnp.sign(x), l
    )

    # ------------------ Estimate the ground temperature T_g_t2 ------------------ #
    # jax.debug.print("Calculating T_g_t2 ...")

    # def func_ground_temp(x, args):
    #     return residual_ground_temp(x, **args)

    # args = dict(
    #     T_v_t2=T_v_t2,
    #     l_guess=l,
    #     S_g=S_g,
    #     L_down=L_down_t2,
    #     pres=pres_a_t2,
    #     ρ_atm_t2=ρ_atm_t2,
    #     T_v_t1=T_v_t1,
    #     T_g_t1=T_g_t1,
    #     T_a_t2=T_a_t2,
    #     u_a_t2=u_a_t2,
    #     q_a_t2=q_a_t2,
    #     T_soil1_t1=T_soil1_t1,
    #     κ=κ,
    #     dz=dz,
    #     L=L_t2,
    #     S=S_t2,
    #     ε_g=ε_g,
    #     ε_v=ε_v,
    #     z_a=z_a,
    #     z0m=z0m,
    #     z0c=z0c,
    #     d=d,
    #     gstomatal=gstomatal,
    #     gsoil=gsoil,
    # )

    # solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # solution = solver(func_ground_temp, T_g_t2_guess, args=args)

    # T_g_t2 = solution.root
    T_g_t2 = T_g_t2_guess
    # T_g_t2 = T_v_t2

    # ---------------------- Update the longwave radiations ---------------------- #
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L_t2,
        S=S_t2,
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
    q_g_t2 = q_g_t2_sat
    # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

    # # - Use the updated Obukhov to perform Monin Obukhov similarity theory again - #
    # kwarg = dict(
    #     pres=pres_a_t2, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2,
    #     u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2,
    #     L=L_t2, S=S_t2, z_a=z_a, z0m=z0m, z0c=z0c,
    #     d=d, gstomatal=gstomatal, gsoil=gsoil,
    # )

    # def func(L, args):
    #     return func_most_dual_source(L, **args)

    # solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7)
    # solution = solver(func, l_guess, args=kwarg)

    # l = solution.root  # noqa: E741

    (
        _,  # noqa: E741
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l,
        pres=pres_a_t2,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L_t2,
        S=S_t2,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )
    # jax.debug.print("Conductances: {}", jnp.array([gvm, gvw]))

    # ------------------------- Update the canopy fluxes ------------------------- #
    # print(gvm)
    H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)
    E_v = calculate_E(
        q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]

    # ------------------------- Update the ground fluxes ------------------------- #
    H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    # G   = calculate_G(T_g=T_g_t2_guess, T_s1=T_soil1_t1, κ=κ, dz=dz)
    G = S_g - L_g - H_g - λ * E_g

    # jax.debug.print("l, T_v_t2, T_g_t2: {}", jnp.array([l, T_v_t2, T_g_t2]))
    # jax.debug.print("Canopy fluxes: {}", jnp.array([S_v, L_v, H_v, λ * E_v]))
    # jax.debug.print("Ground fluxes: {}", jnp.array([S_g, L_g, H_g, λ * E_g, G]))

    return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


def residual_canopy_temp(
    x,
    T_g_t2,
    l_guess,
    S_v,
    L_down,
    pres,
    ρ_atm_t2,
    T_v_t1,
    T_g_t1,
    T_a_t2,
    u_a_t2,
    q_a_t2,
    q_g_t2,
    L,
    S,
    ε_g,
    ε_v,
    z_a,
    z0m,
    z0c,
    d,
    gstomatal,
    gsoil,
    l_reset=l_thres,
):

    T_v_t2 = x

    # Calculate the longwave radiation absorbed by the leaf/canopy
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L,
        S=S,
    )
    # jax.debug.print("L_v and L_g: {}", jnp.array([L_v, L_g]))

    # # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    # q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    # q_g_t2 = q_g_t2_sat

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    (
        l_update,
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l_guess,
        pres=pres,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L,
        S=S,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )
    # jax.debug.print("Conductances: {}", jnp.array([gvm, gvw]))

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(
        T_v=T_v_t2,
        T_s=T_s_t2,
        q_v_sat=q_v_sat_t2,
        q_s=q_s_t2,
        gh=gvm,
        ge=gvw,
        S_v=S_v,
        L_v=L_v,
        ρ_atm=ρ_atm_t2,
    )

    # # Solve the energy balance to estimate the ground temperature
    # denergy_g = ground_energy_balance(
    # T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2,
    # q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2
    # )

    # return jnp.array([denergy_v, denergy_g])
    return denergy_v


def residual_ground_temp(
    x,
    T_v_t2,
    l_guess,
    S_g,
    L_down,
    pres,
    ρ_atm_t2,
    T_v_t1,
    T_g_t1,
    T_a_t2,
    u_a_t2,
    q_a_t2,
    T_soil1_t1,
    κ,
    dz,
    L,
    S,
    ε_g,
    ε_v,
    z_a,
    z0m,
    z0c,
    d,
    gstomatal,
    gsoil,
):

    T_g_t2 = x

    # Calculate the longwave radiation absorbed by the leaf/canopy
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L,
        S=S,
    )

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    q_g_t2 = q_g_t2_sat

    l_update = l_guess

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    (
        _,
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat_t2,
        T_s_t2,
        q_s_t2,
    ) = perform_most_dual_source(
        L_guess=l_update,
        pres=pres,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L,
        S=S,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    # Solve the energy balance to estimate the ground temperature
    denergy_g = ground_energy_balance(
        T_g=T_g_t2,
        T_s=T_s_t2,
        T_s1=T_soil1_t1,
        κ=κ,
        dz=dz,
        q_g=q_g_t2,
        q_s=q_s_t2,
        gh=ggm,
        ge=ggw,
        S_g=S_g,
        L_g=L_g,
        ρ_atm=ρ_atm_t2,
    )

    # return jnp.array([denergy_v, denergy_g])
    return denergy_g


# def residual_ground_temp(
#     x,
#     T_v_t2,
#     l_guess,
#     S_g,
#     L_down,
#     pres,
#     ρ_atm_t2,
#     T_v_t1,
#     T_g_t1,
#     T_a_t2,
#     u_a_t2,
#     q_a_t2,
#     T_soil1_t1,
#     κ,
#     dz,
#     L,
#     S,
#     ε_g,
#     ε_v,
#     z_a,
#     z0m,
#     z0c,
#     d,
#     gstomatal,
#     gsoil,
# ):

#     T_g_t2 = x

#     # Calculate the longwave radiation absorbed by the leaf/canopy
#     L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
#         L_down=L_down,
#         ε_v=ε_v,
#         ε_g=ε_g,
#         T_v_t1=T_v_t1,
#         T_v_t2=T_v_t2,
#         T_g_t1=T_g_t1,
#         T_g_t2=T_g_t2,
#         L=L,
#         S=S,
#     )

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
#     q_g_t2 = q_g_t2_sat

#     # Solve Monin-Obukhov length
#     kwarg = dict(
#         pres=pres,
#         T_v=T_v_t2,
#         T_g=T_g_t2,
#         T_a=T_a_t2,
#         u_a=u_a_t2,
#         q_a=q_a_t2,
#         q_g=q_g_t2,
#         L=L,
#         S=S,
#         z_a=z_a,
#         z0m=z0m,
#         z0c=z0c,
#         d=d,
#         gstomatal=gstomatal,
#         gsoil=gsoil,
#     )

#     def func(L, args):
#         return func_most_dual_source(L, **args)

#     solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7, max_steps=5)
#     solution = solver(func, l_guess, args=kwarg)
#     l_update = solution.root
#     # jax.debug.print("L_v and L_g: {}", jnp.array([L_v, L_g]))
#     # l_update = jnp.min(jn)
#     # l_update = l_guess
#     # jax.debug.print("l and T_g_t2: {}", jnp.array([l_update, T_g_t2]))

#     # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
#     (
#         _,
#         gam,
#         gaw,
#         gvm,
#         gvw,
#         ggm,
#         ggw,
#         q_v_sat_t2,
#         T_s_t2,
#         q_s_t2,
#     ) = perform_most_dual_source(
#         L_guess=l_update,
#         pres=pres,
#         T_v=T_v_t2,
#         T_g=T_g_t2,
#         T_a=T_a_t2,
#         u_a=u_a_t2,
#         q_a=q_a_t2,
#         q_g=q_g_t2,
#         L=L,
#         S=S,
#         z_a=z_a,
#         z0m=z0m,
#         z0c=z0c,
#         d=d,
#         gstomatal=gstomatal,
#         gsoil=gsoil,
#     )

#     # Solve the energy balance to estimate the ground temperature
#     denergy_g = ground_energy_balance(
#         T_g=T_g_t2,
#         T_s=T_s_t2,
#         T_s1=T_soil1_t1,
#         κ=κ,
#         dz=dz,
#         q_g=q_g_t2,
#         q_s=q_s_t2,
#         gh=ggm,
#         ge=ggw,
#         S_g=S_g,
#         L_g=L_g,
#         ρ_atm=ρ_atm_t2,
#     )

#     # return jnp.array([denergy_v, denergy_g])
#     return denergy_g


# def solve_surface_energy_v2(
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

#     # Calculate the albedos and emissivities
#     α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
#     α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
#     ε_g, ε_v                = calculate_ground_vegetation_emissivity(
#         solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind  # noqa: E501
#         )

#     # Calculate the solar radiation fluxes reaching the canopy
#     S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
#         solar_rad=solar_rad_t2, pres=pres_a_t2*1e-3, solar_elev_angle=solar_elev_angle,  # noqa: E501
#         α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,  # noqa: E501
#         f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
#     )

#     # Jointly estimate T_g_t2, T_v_t2
#     func = lambda x, args: residual_ground_canopy_temp(x, **args)
#     args = dict(l_guess=l_guess, S_v=S_v, S_g=S_g, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2,  # noqa: E501
#             T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1,
#             T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v,  # noqa: E501
#             z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

#     solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
#     # jax.debug.print("jac : {}", solver.jac(func, [T_v_t2_guess, T_g_t2_guess], args=args))  # noqa: E501
#     solution = solver(func, [T_v_t2_guess, T_g_t2_guess], args=args)
#     T_v_t2, T_g_t2 = solution.root
#     # solution = solver(func, T_v_t2_guess, args=args)
#     # T_v_t2 = solution.root
#     # T_g_t2 = T_g_t2_guess

#     # Update the longwave radiatios
#     L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
#         L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g,
#         T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
#         L=L_t2, S=S_t2
#     )

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
#     q_g_t2     = q_g_t2_sat
#     # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

#     # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
#     l, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(  # noqa: E501
#        L_guess=l_guess, pres=pres_a_t2, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L_t2, S=S_t2,  # noqa: E501
#        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
#     )

#     # Update the canopy fluxes
#     H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
#     E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]  # noqa: E501

#     # Update the ground fluxes
#     H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
#     E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
#     G   = calculate_G(T_g=T_g_t2, T_s1=T_soil1_t1, κ=κ, dz=dz)

#     return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


# def residual_ground_canopy_temp(
#     x, l_guess, S_v, S_g, L_down, pres, ρ_atm_t2,
#     T_soil1_t1, κ, dz,
#     T_v_t1, T_g_t1, T_a_t2,
#     u_a_t2, q_a_t2, L, S, ε_g, ε_v,
#     z_a, z0m, z0c, d, gstomatal, gsoil):

#     T_v_t2, T_g_t2 = x
#     # T_v_t2 = x

#     # Calculate the longwave radiation absorbed by the leaf/canopy
#     L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
#         L_down=L_down, ε_v=ε_v, ε_g=ε_g,
#         T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
#         L=L, S=S
#     )

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
#     q_g_t2     = q_g_t2_sat

#     # Solve Monin-Obukhov length
#     kwarg = dict(
#         pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,  # noqa: E501
#         z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
#     )
#     func = lambda L, args: func_most_dual_source(L, **args)
#     solver = NewtonNonlinearSolver(atol=1e-5, rtol=1e-7)
#     solution = solver(func, l_guess, args=kwarg)
#     L_update = solution.root
#     jax.debug.print("Updated L: {}", L_update)
#     jax.debug.print("Updated T_g and T_v: {}", x)

#     # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
#     _, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(  # noqa: E501
#        L_guess=L_update, pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,  # noqa: E501
#        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
#     )

#     # Solve the energy balance to estimate the vegetation temperature
#     denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)  # noqa: E501

#     # Solve the energy balance to estimate the ground temperature
#     denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)  # noqa: E501

#     jax.debug.print("Updated denergy_v: {}", denergy_v)
#     jax.debug.print("Updated denergy_g: {}", denergy_g)

#     return jnp.array([denergy_v, denergy_g])
#     # return denergy_v


# def solve_surface_energy_v3(
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
#         solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L_t2, S=S_t2, pft_ind=pft_ind  # noqa: E501
#         )

#     # Calculate the solar radiation fluxes reaching the canopy
#     S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
#         solar_rad=solar_rad_t2, pres=pres_a_t2*1e-3, solar_elev_angle=solar_elev_angle,  # noqa: E501
#         α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,  # noqa: E501
#         f_snow=f_snow, f_cansno=f_cansno, L=L_t2, S=S_t2, pft_ind=pft_ind
#     )

#     # Jointly estimate T_g_t2, T_v_t2 and L
#     func = lambda x, args: residual_ground_canopy_temp_L(x, **args)
#     args = dict(S_v=S_v, S_g=S_g, L_down=L_down_t2, pres=pres_a_t2, ρ_atm_t2=ρ_atm_t2,
#             T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1,
#             T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L_t2, S=S_t2, ε_g=ε_g, ε_v=ε_v,  # noqa: E501
#             z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

#     solver = NewtonNonlinearSolver(atol=atol, rtol=rtol, max_steps=1000)
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
#     ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)  # noqa: E501

#     # Calculate the conductances of heat and water vapor
#     # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)  # noqa: E501
#     gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)  # noqa: E501
#     gaw = gam
#     gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L_t2, S=S_t2)
#     gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
#     ggm = calculate_conductance_ground_canopy(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m)
#     ggw = calculate_conductance_ground_canopy_water_vapo(L=L_t2, S=S_t2, ustar=ustar, z0m=z0m, gsoil=gsoil)  # noqa: E501
#     # print(gvw, gvm)

#     # Calculate the saturated specific humidity from temperature and pressure
#     e_v_sat_t2 = esat_from_temp(T=T_v_t2)
#     q_v_sat_t2 = q_from_e_pres(pres=pres_a_t2, e=e_v_sat_t2)

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres_a_t2)
#     q_g_t2     = q_g_t2_sat
#     # jax.debug.print("Ground specific humidity: {}", q_g_t2_sat)

#     # Calculate the temperature and specific humidity of the canopy air/surface
#     T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)  # noqa: E501
#     q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)  # noqa: E501
#     # print(T_s_t2, q_s_t2)

#     # Calculate the updated Obukhov length
#     tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]  # noqa: E501
#     qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]  # noqa: E501
#     # Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
#     # Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5  # noqa: E501

#     # Update the canopy fluxes
#     H_v = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2*gvm)
#     E_v = calculate_E(q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw)  # [kg m-2 s-1]  # noqa: E501

#     # Update the ground fluxes
#     H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
#     E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
#     G   = calculate_G(T_g=T_g_t2, T_s1=T_soil1_t1, κ=κ, dz=dz)

#     return l, T_v_t2, T_g_t2, S_v, S_g, L_v, L_g, H_v, H_g, E_v, E_g, G


# def residual_ground_canopy_temp_L(
#     x, S_v, S_g, L_down, pres, ρ_atm_t2,
#     T_soil1_t1, κ, dz,
#     T_v_t1, T_g_t1, T_a_t2,
#     u_a_t2, q_a_t2, L, S, ε_g, ε_v,
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
#     # jax.debug.print("The guessing L: {}", L_guess)
#     ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
#     ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
#     ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
#     ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
#     ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)  # noqa: E501
#     # TODO: check why ustar is negative. For now, we force it to be positive
#     ustar = jnp.absolute(ustar)

#     # Calculate the conductances of heat and water vapor
#     # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)  # noqa: E501
#     gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)  # noqa: E501
#     gaw = gam
#     gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)
#     gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
#     ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
#     ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)  # noqa: E501
#     # jax.debug.print("ustar, gvw, ggw: {}", jnp.array([ustar, gvw, ggw]))
#     # print(gvw, gvm)

#     # Calculate the saturated specific humidity from temperature and pressure
#     e_v_sat_t2 = esat_from_temp(T=T_v_t2)
#     q_v_sat_t2 = q_from_e_pres(pres=pres, e=e_v_sat_t2)
#     # jax.debug.print("q_v_sat_t2: {}", jnp.array([q_v_sat_t2, T_v_t2, pres, e_v_sat_t2]))  # noqa: E501

#     # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
#     q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
#     q_g_t2     = q_g_t2_sat

#     # Calculate the temperature and specific humidity of the canopy air/surface
#     T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)  # noqa: E501
#     q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)  # noqa: E501
#     # print(T_s_t2, q_s_t2)

#     # Calculate the updated Obukhov length
#     tstar  = calculate_Tstar_most(T1=T_s_t2, T2=T_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]  # noqa: E501
#     qstar  = calculate_qstar_most(q1=q_s_t2, q2=q_a_t2, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]  # noqa: E501
#     Tzv    = T_a_t2 * (1 + 0.608 * q_a_t2)
#     Tvstar = tstar * (1 + 0.608 * q_a_t2) + 0.608 * T_a_t2 * qstar # Eq(5.17) in CLM5
#     L_est  = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)
#     dL     = L_est - L_guess

#     # Solve the energy balance to estimate the vegetation temperature
#     denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm_t2)  # noqa: E501

#     # Solve the energy balance to estimate the ground temperature
#     denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm_t2)  # noqa: E501

#     jax.debug.print("Updated denergy_v: {}", denergy_v)
#     jax.debug.print("Updated denergy_g: {}", denergy_g)

#     return jnp.array([dL, denergy_v, denergy_g])
