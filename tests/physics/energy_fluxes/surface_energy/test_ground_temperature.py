import unittest

import jax
import jax.numpy as jnp

from diffrax import NewtonNonlinearSolver

from jax_watershed.physics.energy_fluxes.surface_energy import ground_energy_balance, leaf_energy_balance, calculate_Ts_from_TvTgTa, calculate_qs_from_qvqgqa
from jax_watershed.physics.energy_fluxes.surface_energy import func_most_dual_source, perform_most_dual_source

from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_fluxes, calculate_longwave_fluxes
from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_elevation
from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_ground_albedos, calculate_ground_vegetation_emissivity
from jax_watershed.physics.energy_fluxes.radiative_transfer import main_calculate_solar_fluxes

from jax_watershed.physics.energy_fluxes.turbulent_fluxes import *

from jax_watershed.physics.water_fluxes import qsat_from_temp_pres

from jax_watershed.shared_utilities.constants import R_DA as Rda
from jax_watershed.shared_utilities.constants import C_TO_K as c2k

# TODO: See the conversion among relative humidity, specific humidity, vapor pressure, saturated vapor pressure, temperature here:
# https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html

class TestGroundTemperature(unittest.TestCase):

    def test_estimate_ground_temperature(self):
        print("Performing test_estimate_ground_temperature()...")
        # solar_rad, L_down, pres        = 352., 200., 101.976
        solar_rad, L_down, pres        = 352., 200., 101.976
        L, S, pft_ind                  = 2., 1., 10
        latitude, longitude            = 31.31, 120.77
        year, day, hour, zone          = 2023, 68, 10., -8
        # T_v_t1, T_v_t2, T_g_t1, T_g_t2 = 15., 16., 10., 11.
        T_v_t1, T_v_t2 = 15.+c2k, 15.+c2k
        # T_g_t1, T_g_t2, q_g_t2 = 10.+c2k, 11.+c2k, 0.01
        T_g_t1, T_g_t2 = 10.+c2k, 11.+c2k
        T_a_t2, u_a_t2, z_a, q_a_t2 = 12.+c2k, 4., 2.5, 0.015
        T_soil1_t1, dz, κ = 9.+c2k, 0.05, 1. 
        z0m, z0c, d, L_guess = 0.05, 0.05, 0.05, -1.
        gsoil, gstomatal = 1e10, 1./180.
        f_snow, f_cansno = 0.1, 0.1

        atol, rtol = 1e-5, 1e-7

        # Calculate solar elevation angle
        solar_elev_angle = calculate_solar_elevation(
            latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
            zone=zone, is_day_saving=False
        )

        # Calculate the air density
        e_a = (pres*1e3 * q_a_t2) / (0.622 + 0.378 * q_a_t2) # [Pa]
        ρ_atm = (pres*1e3 - 0.378*e_a) / (Rda * T_a_t2) # [kg m-3]
        # print(e_a, ρ_atm)

        # Calculate the albedos and emissivities
        α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
        α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
        ε_g, ε_v                = calculate_ground_vegetation_emissivity(
            solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L, S=S, pft_ind=pft_ind
            )

        # Calculate the solar radiation fluxes reaching the canopy
        S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
            solar_rad=solar_rad, pres=pres, solar_elev_angle=solar_elev_angle, 
            α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
            f_snow=f_snow, f_cansno=f_cansno, L=L, S=S, pft_ind=pft_ind
        )

        # Calculate the longwave radiation absorbed by the leaf/canopy
        L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
            L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
            T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
            L=L, S=S
        )

        # Monin-Ob similarity theory (MOST)
        ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
        ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
        ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
        ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
        ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

        # Calculate the conductances of heat and water vapor
        # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
        gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
        gaw = gam
        gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)    
        gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
        ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
        ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)
        # print(gvw, gvm)

        # Calculate the saturated specific humidity from temperature and pressure
        # TODO: Check the units!
        a, b = 17.2693882, 35.86
        e_v_sat_t2 = 610.78 * jnp.exp(a * (T_v_t2 - c2k) / (T_v_t2 - b)) # [Pa]
        q_v_sat_t2 = (0.622 * e_v_sat_t2) / (pres*1e3 - 0.378 *e_v_sat_t2) # [kg kg-1]
        # print(e_v_sat_t2, q_v_sat_t2, q_g_t2, q_a_t2)

        # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
        q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres*1e3)
        q_g_t2     = q_g_t2_sat

        # Calculate the temperature and specific humidity of the canopy air/surface
        T_s_t2 = calculate_Ts_from_TvTgTa(Tv=T_v_t2, Tg=T_g_t2, Ta=T_a_t2, gam=gam, gvm=gvm, ggm=ggm)
        q_s_t2 = calculate_qs_from_qvqgqa(qv_sat=q_v_sat_t2, qg=q_g_t2, qa=q_a_t2, gaw=gaw, gvw=gvw, ggw=ggw)
        # print(T_s_t2, q_s_t2)

        # # Solve the energy balance to estimate the vegetation temperature
        # # print(leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2*1e-3, q_s=q_s_t2*1e-3, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm))
        # args = dict(T_s=T_s_t2, q_v_sat=q_v_sat_t2*1e-3, q_s=q_s_t2*1e-3, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm)
        # func = lambda T_v, args: leaf_energy_balance(T_v=T_v, **args)
        # # print(func(T_v_t2, args))

        # Solve the energy balance to estimate the ground temperature
        # print(ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2*1e-3, q_s=q_s_t2*1e-3, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm))
        args = dict(T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm)
        func = lambda T_g, args: ground_energy_balance(T_g=T_g, **args)
        # print(func(T_v_t2, args))

        solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
        solution = solver(func, T_g_t2, args=args)
        T_g_t2_final = solution.root
        print("The guess ground temperature is: {}".format(T_g_t2))
        print("The estimated ground temperature is: {}".format(T_g_t2_final))
        # print(func(T_g_t2_final, args))
        print("")
        self.assertTrue(is_solar_rad_balanced)


    def test_estimate_ground_canopy_temperature_and_L(self):
        print("Performing test_estimate_ground_canopy_temperature_and_L()...")
        solar_rad, L_down, pres        = 352., 200., 101.976
        L, S, pft_ind                  = 2., 1., 10
        latitude, longitude            = 31.31, 120.77
        year, day, hour, zone          = 2023, 68, 10., -8
        # T_v_t1, T_v_t2, T_g_t1, T_g_t2 = 15., 16., 10., 11.
        T_v_t1, T_v_t2 = 15.+c2k, 15.+c2k
        # T_g_t1, T_g_t2, q_g_t2 = 10.+c2k, 11.+c2k, 0.01
        T_g_t1, T_g_t2 = 10.+c2k, 11.+c2k
        T_a_t2, u_a_t2, z_a, q_a_t2 = 12.+c2k, 4., 2.5, 0.015
        T_soil1_t1, dz, κ = 9.+c2k, 0.05, 1. 
        z0m, z0c, d, L_guess = 0.05, 0.05, 0.05, -1.
        gsoil, gstomatal = 1e10, 1./180.
        f_snow, f_cansno = 0.1, 0.1

        atol, rtol = 1e-5, 1e-7

        # Calculate solar elevation angle
        solar_elev_angle = calculate_solar_elevation(
            latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
            zone=zone, is_day_saving=False
        )

        # Calculate the air density
        e_a = (pres*1e3 * q_a_t2) / (0.622 + 0.378 * q_a_t2) # [Pa]
        ρ_atm = (pres*1e3 - 0.378*e_a) / (Rda * T_a_t2) # [kg m-3]
        # print(e_a, ρ_atm)

        # Calculate the albedos and emissivities
        α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
        α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
        ε_g, ε_v                = calculate_ground_vegetation_emissivity(
            solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L, S=S, pft_ind=pft_ind
            )

        # Calculate the solar radiation fluxes reaching the canopy
        S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
            solar_rad=solar_rad, pres=pres, solar_elev_angle=solar_elev_angle, 
            α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
            f_snow=f_snow, f_cansno=f_cansno, L=L, S=S, pft_ind=pft_ind
        )

        # Jointly estimate T_v_t2 and L
        func = lambda x, args: calculate_ground_canopy_temp_L(x, **args)
        args = dict(S_v=S_v, S_g=S_g, L_down=L_down, pres=pres, ρ_atm=ρ_atm, 
             T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1, T_g_t2=T_g_t2, 
             T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L, S=S, ε_g=ε_g, ε_v=ε_v, 
             z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

        solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
        solution = solver(func, [L_guess, T_v_t2, T_g_t2], args=args)
        L_est, T_v_t2_final, T_g_t2_final = solution.root
        print("The guess and estimated vegetation temperatures: {}, {}".format(T_v_t2, T_v_t2_final))
        print("The guess and estimated ground temperatures: {}, {}".format(T_g_t2, T_g_t2_final))
        # print(func(T_v_t2_final, args))
        print("")
        self.assertTrue(is_solar_rad_balanced)


    def test_estimate_L_then_ground_canopy_temperature(self):
        print("Performing test_estimate_L_then_ground_canopy_temperature()...")
        # solar_rad, L_down, pres        = 352., 200., 101.976
        solar_rad, L_down, pres        = 352., 200., 101976.
        L, S, pft_ind                  = 2., 1., 10
        latitude, longitude            = 31.31, 120.77
        year, day, hour, zone          = 2023, 68, 10., -8
        # T_v_t1, T_v_t2, T_g_t1, T_g_t2 = 15., 16., 10., 11.
        T_v_t1, T_v_t2 = 15.+c2k, 15.+c2k
        # T_g_t1, T_g_t2, q_g_t2 = 10.+c2k, 11.+c2k, 0.01
        T_g_t1, T_g_t2 = 10.+c2k, 11.+c2k
        T_a_t2, u_a_t2, z_a, q_a_t2 = 12.+c2k, 4., 2.5, 0.015
        T_soil1_t1, dz, κ = 9.+c2k, 0.05, 1. 
        z0m, z0c, d, L_guess = 0.05, 0.05, 0.05, -1.
        gsoil, gstomatal = 1e10, 1./180.
        f_snow, f_cansno = 0.1, 0.1

        atol, rtol = 1e-5, 1e-7

        # Calculate solar elevation angle
        solar_elev_angle = calculate_solar_elevation(
            latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
            zone=zone, is_day_saving=False
        )

        # Calculate the air density
        e_a = (pres * q_a_t2) / (0.622 + 0.378 * q_a_t2) # [Pa]
        ρ_atm = (pres - 0.378*e_a) / (Rda * T_a_t2) # [kg m-3]
        # print(e_a, ρ_atm)

        # Calculate the albedos and emissivities
        α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
        α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
        ε_g, ε_v                = calculate_ground_vegetation_emissivity(
            solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L, S=S, pft_ind=pft_ind
            )

        # Calculate the solar radiation fluxes reaching the canopy
        S_v, S_g, is_solar_rad_balanced = main_calculate_solar_fluxes(
            solar_rad=solar_rad, pres=pres*1e-3, solar_elev_angle=solar_elev_angle, 
            α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir,
            f_snow=f_snow, f_cansno=f_cansno, L=L, S=S, pft_ind=pft_ind
        )

        # Jointly estimate T_v_t2 and L
        func = lambda x, args: calculate_ground_canopy_temp(x, **args)
        args = dict(L_guess=L_guess, S_v=S_v, S_g=S_g, L_down=L_down, pres=pres, ρ_atm=ρ_atm, 
             T_soil1_t1=T_soil1_t1, κ=κ, dz=dz, T_v_t1=T_v_t1, T_g_t1=T_g_t1, T_g_t2=T_g_t2, 
             T_a_t2=T_a_t2, u_a_t2=u_a_t2, q_a_t2=q_a_t2, L=L, S=S, ε_g=ε_g, ε_v=ε_v, 
             z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil)

        solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
        solution = solver(func, [T_v_t2, T_g_t2], args=args)
        T_v_t2_final, T_g_t2_final = solution.root
        # print("The guess and estimated L: {}, {}".format(L_guess, L_est))
        print("The guess and estimated vegetation temperatures: {}, {}".format(T_v_t2, T_v_t2_final))
        print("The guess and estimated ground temperatures: {}, {}".format(T_g_t2, T_g_t2_final))
        # print(func(T_v_t2_final, args))
        print("")
        self.assertTrue(is_solar_rad_balanced)


def calculate_ground_canopy_temp_L(
    x, S_v, S_g, L_down, pres, ρ_atm,
    T_soil1_t1, κ, dz,
    T_v_t1, T_g_t1, T_g_t2, T_a_t2, 
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

    # kwarg = dict(uz=u_a_t2, Tz=T_a_t2, qz=q_a_t2, Ts=ts, qs=qs, z=z, d=d, z0m=z0m, z0c=z0c)
    # args = dict(uz=uz, tz=tz, qz=qz, ts=ts, qs=qs, z=z, d=d, z0m=z0m, z0c=z0c)
    # func = lambda L, args: func_most(L, **args)
    # print(func(L_guess, kwarg))
    # solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
    # solution = solver(func, L_guess, args=kwarg)
    # L = solution.root

    # Monin-Ob similarity theory (MOST)
    ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
    ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
    ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
    ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
    ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

    # Calculate the conductances of heat and water vapor
    # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
    gam = calculate_scalar_conduct_surf_atmos(uref=u_a_t2, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
    gaw = gam
    gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)    
    gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
    ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
    ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)
    # print(gvw, gvm)

    # Calculate the saturated specific humidity from temperature and pressure
    # TODO: Check the units!
    a, b = 17.2693882, 35.86
    e_v_sat_t2 = 610.78 * jnp.exp(a * (T_v_t2 - c2k) / (T_v_t2 - b)) # [Pa]
    q_v_sat_t2 = (0.622 * e_v_sat_t2) / (pres*1e3 - 0.378 *e_v_sat_t2) # [kg kg-1]
    # print(e_v_sat_t2, q_v_sat_t2, q_g_t2, q_a_t2)

    # TODO: Calculate the specific humidity on the ground (Eq(5.73) in CLM5)
    q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres*1e3)
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
    # jax.debug.print("{}", jnp.array([ustar, tstar, qstar, Tzv, Tvstar]))
    L_est  = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)
    dL     = L_est - L_guess

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm)

    # Solve the energy balance to estimate the ground temperature
    denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm)

    return jnp.array([dL, denergy_v, denergy_g])


def calculate_ground_canopy_temp(
    x, L_guess, S_v, S_g, L_down, pres, ρ_atm,
    T_soil1_t1, κ, dz,
    T_v_t1, T_g_t1, T_g_t2, T_a_t2, 
    u_a_t2, q_a_t2, L, S, ε_g, ε_v,
    z_a, z0m, z0c, d, gstomatal, gsoil):

    T_v_t2, T_g_t2 = x 

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
    solution = solver(func, L_guess, args=kwarg)
    L_update = solution.root
    # jax.debug.print("Updated L: {}", L_update)

    # # Monin-Ob similarity theory (MOST)
    # ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
    # ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
    # ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
    # ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
    # ustar = calculate_ustar_most(u1=0, u2=u_a_t2, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

    # Use the updated Obukhov to perform Monin Obukhov similarity theory again (MOST)
    _, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2 = perform_most_dual_source(
       L_guess=L_update, pres=pres, T_v=T_v_t2, T_g=T_g_t2, T_a=T_a_t2, u_a=u_a_t2, q_a=q_a_t2, q_g=q_g_t2, L=L, S=S,
       z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil 
    )

    # Solve the energy balance to estimate the vegetation temperature
    denergy_v = leaf_energy_balance(T_v=T_v_t2, T_s=T_s_t2, q_v_sat=q_v_sat_t2, q_s=q_s_t2, gh=gvm, ge=gvw, S_v=S_v, L_v=L_v, ρ_atm=ρ_atm)

    # Solve the energy balance to estimate the ground temperature
    denergy_g = ground_energy_balance(T_g=T_g_t2, T_s=T_s_t2, T_s1=T_soil1_t1, κ=κ, dz=dz, q_g=q_g_t2, q_s=q_s_t2, gh=ggm, ge=ggw, S_g=S_g, L_g=L_g, ρ_atm=ρ_atm)

    return jnp.array([denergy_v, denergy_g])