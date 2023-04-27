"""
Main functions for solving surface energy balances based on CLM implementation.

Author: Peishi Jiang
Date: 2023.04.25.

"""

import jax
import jax.numpy as jnp

from typing import Tuple
from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import λ_VAP as λ

from diffrax import NewtonNonlinearSolver

from .monin_obukhov import perform_most_dual_source
from .monin_obukhov import calculate_initial_guess_L

# from .monin_obukhov import func_most_dual_source
# from .monin_obukhov import calculate_conductances_dual_source
# from .canopy_energy import leaf_energy_balance
from .ground_energy import ground_energy_balance
from .surface_state import calculate_qs_from_qvqgqa, calculate_Ts_from_TvTgTa

from ..radiative_transfer import calculate_longwave_fluxes
from ..radiative_transfer import calculate_canopy_longwave_fluxes
from ..radiative_transfer import calculate_solar_elevation
from ..radiative_transfer import (
    calculate_ground_albedos,
    calculate_ground_vegetation_emissivity,
)
from ..radiative_transfer import main_calculate_solar_fluxes
from ..turbulent_fluxes import calculate_E, calculate_H

from ...water_fluxes import qsat_from_temp_pres


# Define some derivative functions using jax.grad
dLv_dTv_func = jax.grad(calculate_canopy_longwave_fluxes, argnums=0)
dHv_dTv_func = jax.grad(calculate_H, argnums=0)
dEv_dTv_func = jax.grad(calculate_E, argnums=0)
dqvsat_dTv_func = jax.grad(qsat_from_temp_pres, argnums=0)


def solve_surface_energy_canopy_ground_clm(
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
    l_guess = -10.0
    # jax.debug.print("The initial guess of L: {}", l_guess)

    # ------------------ Estimate the canopy temperature T_v_t2 ------------------ #
    # jax.debug.print("Calculating T_v_t2 ...")
    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2_guess, pres=pres_a_t2)
    q_g_t2_guess = q_g_t2_sat

    l, T_v_t2 = estimate_canopy_temperature(  # noqa: E741
        l_guess=l_guess,
        S_v_t2=S_v,
        L_down_t2=L_down_t2,
        pres_a_t2=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2_guess,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2_guess,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        q_g_t2=q_g_t2_guess,
        L_t2=L_t2,
        S_t2=S_t2,
        ε_g=ε_g,
        ε_v=ε_v,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    # ------------------ Estimate the ground temperature T_g_t2 ------------------ #
    # jax.debug.print("Calculating T_g_t2 ...")

    def func_ground_temp(x, args):
        return residual_ground_temp(x, **args)

    args = dict(
        l=l,
        T_v_t2=T_v_t2,
        S_g=S_g,
        L_down=L_down_t2,
        pres=pres_a_t2,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_g_t1=T_g_t1,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        T_soil1_t1=T_soil1_t1,
        κ=κ,
        dz=dz,
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
    solution = solver(func_ground_temp, T_g_t2_guess, args=args)

    T_g_t2 = solution.root

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


# TODO
def estimate_canopy_temperature(
    l_guess,
    S_v_t2,
    L_down_t2,
    pres_a_t2,
    ρ_atm_t2,
    T_v_t1,
    T_v_t2,
    T_g_t1,
    T_g_t2,
    T_a_t2,
    u_a_t2,
    q_a_t2,
    q_g_t2,
    L_t2,
    S_t2,
    ε_g,
    ε_v,
    z_a,
    z0m,
    z0c,
    d,
    gstomatal,
    gsoil,
):
    def cond_fun(args):
        niter, l_updated, T_v_t2_updated, ΔT_v_t2 = args
        # return (niter<40) and (jnp.abs(ΔT_v_t2)>0.01)
        return niter < 40

    def body_fun(args):
        niter, l, T_v_t2, _ = args  # noqa: E741
        (
            l_updated,
            T_v_t2_updated,
            λE_v_t2_updated,
            H_v_t2_updated,
            L_v_t2_updated,
        ) = estimate_canopy_temperature_each_iteration(
            l=l,
            S_v_t2=S_v_t2,
            L_down_t2=L_down_t2,
            pres_a_t2=pres_a_t2,
            ρ_atm_t2=ρ_atm_t2,
            T_v_t1=T_v_t1,
            T_v_t2=T_v_t2,
            T_g_t1=T_g_t1,
            T_g_t2=T_g_t2,
            T_a_t2=T_a_t2,
            u_a_t2=u_a_t2,
            q_a_t2=q_a_t2,
            q_g_t2=q_g_t2,
            L_t2=L_t2,
            S_t2=S_t2,
            ε_g=ε_g,
            ε_v=ε_v,
            z_a=z_a,
            z0m=z0m,
            z0c=z0c,
            d=d,
            gstomatal=gstomatal,
            gsoil=gsoil,
        )
        # jax.debug.print(
        #     "Updated l and T_v_t2: {}", jnp.array([l_updated, T_v_t2_updated])
        # )
        ΔT_v_t2 = T_v_t2_updated - T_v_t2
        niter += 1
        return niter, l_updated, T_v_t2_updated, ΔT_v_t2

    init_val = (0, l_guess, T_v_t2, 0.0)

    niter, l_final, T_v_t2_final, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return l_final, T_v_t2_final


def estimate_canopy_temperature_each_iteration(
    l,  # noqa: E741
    S_v_t2,
    L_down_t2,
    pres_a_t2,
    ρ_atm_t2,
    T_v_t1,
    T_v_t2,
    T_g_t1,
    T_g_t2,
    T_a_t2,
    u_a_t2,
    q_a_t2,
    q_g_t2,
    L_t2,
    S_t2,
    ε_g,
    ε_v,
    z_a,
    z0m,
    z0c,
    d,
    gstomatal,
    gsoil,
):
    # TODO: sunlit and shaded stomatal resistances
    # gsoil =

    # calculate the conductances
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

    # sensible heat flux from vegetation
    H_v_t2 = calculate_H(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)

    # latent heat flux from vegetation
    E_v_t2 = calculate_E(
        q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]
    λE_v_t2 = λ * E_v_t2

    # longwave radiation from vegetation
    L_v_t2, _, _, _, _, _, _ = calculate_longwave_fluxes(
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
    L_v_t2 = calculate_canopy_longwave_fluxes(
        T_v=T_v_t2, T_g=T_g_t1, L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g
    )

    # TODO: difference between the constrained and computed latent heat flux
    Δ1 = 0.0

    # change in vegetation temperature and the corresponding energy error
    # dHv_dTv = dHv_dTv_func(T_1=T_v_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=2 * gvm)
    # dLv_dTv = dLv_dTv_func(
    #     L_down=L_down_t2, ε_v=ε_v, ε_g=ε_g, T_v_t1=T_v_t1, T_v_t2=T_v_t2,
    #     T_g_t1=T_g_t1, T_g_t2=T_g_t2, L=L_t2, S=S_t2,
    # )
    # dλEv_Tv = λ * dEv_dTv_func(
    #     q_1=q_v_sat_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=gvw
    #     ) * dqvsat_dTv_func(T=T_v_t2, pres=pres_a_t2)
    dHv_dTv = dHv_dTv_func(T_v_t2, T_s_t2, ρ_atm_t2, 2 * gvm)
    dLv_dTv = dLv_dTv_func(T_v_t2, T_g_t1, L_down_t2, ε_v, ε_g)
    dλEv_Tv = (
        λ
        * dEv_dTv_func(q_v_sat_t2, q_s_t2, ρ_atm_t2, gvw)
        * dqvsat_dTv_func(T_v_t2, pres_a_t2)
    )

    ΔT_v = (S_v_t2 - L_v_t2 - H_v_t2 - λE_v_t2) / (dLv_dTv + dHv_dTv + dλEv_Tv)
    T_v_t2_updated = T_v_t2 + ΔT_v
    Δ2 = S_v_t2 - L_v_t2 - H_v_t2 - λE_v_t2 - ΔT_v * (dLv_dTv + dHv_dTv + dλEv_Tv)

    # saturated vapor pressure, saturated specific humidity on new Tv
    q_v_sat_t2_updated = qsat_from_temp_pres(T=T_v_t2_updated, pres=pres_a_t2)

    # canopy air temperature and specific humidity
    T_s_t2_updated = calculate_Ts_from_TvTgTa(
        Tv=T_v_t2_updated, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm
    )
    q_s_t2_updated = calculate_qs_from_qvqgqa(
        qv_sat=q_v_sat_t2_updated, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw
    )

    # water vapor flux from vegetation
    E_v_t2_updated = calculate_E(
        q_1=q_v_sat_t2_updated, q_2=q_s_t2_updated, ρ_atm=ρ_atm_t2, ge=gvw
    )  # [kg m-2 s-1]
    λE_v_t2_updated = λ * E_v_t2_updated

    # TODO: transpiration
    Et_v_t2_updated = 0.0

    # energy error due to the constraint on the water vapor flux
    Δ3 = jnp.max(jnp.array([0.0, E_v_t2_updated - Et_v_t2_updated]))

    # update sensible heat flux
    H_v_t2_updated = calculate_H(
        T_1=T_v_t2_updated, T_2=T_s_t2_updated, ρ_atm=ρ_atm_t2, gh=2 * gvm
    )
    H_v_t2_updated += Δ1 + Δ2 + Δ3

    # update vegetation longwave radiation
    L_v_t2_updated, _, _, _, _, _, _ = calculate_longwave_fluxes(
        L_down=L_down_t2,
        ε_v=ε_v,
        ε_g=ε_g,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2_updated,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
        L=L_t2,
        S=S_t2,
    )

    # TODO: Monin-Obukhov length
    (
        l_updated,  # noqa: E741
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
        T_v=T_v_t2_updated,
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

    return l_updated, T_v_t2_updated, λE_v_t2_updated, H_v_t2_updated, L_v_t2_updated


def residual_ground_temp(
    x,
    T_v_t2,
    l,  # noqa: E741
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

    # Perform Monin Obukhov similarity theory again (MOST)
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
