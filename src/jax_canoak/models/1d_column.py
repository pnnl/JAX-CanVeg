"""
One-dimensional column-based hydrobiogeochemical modeling, including:
- surface/subsurface energy balance;
- surface/subsurface water mass balance;
- surface/subsurface carbon/nutrient cycle.

Author: Peishi Jiang
Date: 2023.07.06.
"""

# TODO: Let's first write it in a way that directly takes in the example forcing data.
# Further modifications are needed.

import jax
import jax.numpy as jnp
import h5py
import numpy as np

# from jax_watershed.physics.energy_fluxes.surface_energy import solve_surface_energy
from jax_canoak.physics.energy_fluxes import (
    rnet,
    par,
    nir,
    gfunc,
    diffuse_direct_radiation,
    irflux,
)
from jax_canoak.physics.combined_fluxes import energy_and_carbon_fluxes
from jax_canoak.physics.carbon_fluxes import angle, lai_time, stomata
from jax_canoak.physics.carbon_fluxes import soil_respiration
from jax_canoak.physics.energy_fluxes import soil_energy_balance, set_soil
from jax_canoak.physics.energy_fluxes import friction_velocity
from jax_canoak.shared_utilities import conc

from jax_canoak.shared_utilities.forcings import Alf_forcings_30min as forcings
from jax_canoak.shared_utilities.forcings import Alf_divergence as dispersion_matrix
from jax_canoak.shared_utilities.forcings import get_input_t
from jax_canoak.shared_utilities.domain import Time
from jax_canoak.shared_utilities.constants import grasshof, pr33, sc33, scc33
from jax_canoak.shared_utilities.constants import kball, epsoil, ustar_ref
from jax_canoak.shared_utilities.constants import mass_air, mass_CO2
from jax_canoak.shared_utilities.utils import filter_array

# from jax_canoak.subjects import Surface, Soil

# TODO: Check the data types!
# Set float64
# from jax.config import config
# jax.config.update("jax_enable_x64", True)

plotting = True

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
# Spatio-temporal information/discretizations
t0, tn, dt = 181.0, 194.0, 1.0 / 48.0  # [day]
# t0, tn, dt = 181.0, 181.2, 1.0 / 48.0  # [day]
latitude, longitude, zone = 38.1, -121.65, -8
dt_soil, total_t_soil = 20, dt * 86400  # second

# Subsurface layers
# z0, zn, nz = 0.0, 10.0, 21  # subsurface layers [m]
nsoil = 9
soilsze = nsoil + 3

# Canopy layers
ht = 1.0  # canopy height [m]
jtot = 30  # number of canopy layers
sze, jktot = jtot + 2, jtot + 1
delz = ht / jtot  # height of each layer
zh65 = 0.65 / ht
pai = 0.0  # plant area index
# height of mid point of layer scaled
ht_midpt = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
lai_freq_scaled = jnp.array([0.6, 0.6, 0.6, 0.6, 0.6])
# lai_freq_np = np.array([0.05, .3,.3, .3, .05]) * lai

# Domain layers
jtot3 = 150  # number of layers in the domain, three times canopy height
sze3 = jtot3 + 2
izref = jtot3 - 1  # array index of reference ht

# Optical properties of PAR and NIR
par_reflect, par_trans, par_soil_refl = 0.0377, 0.072, 0.0
nir_reflect, nir_trans, nir_soil_refl = 0.6, 0.26, 0.0
par_absorbed = 1 - par_reflect - par_trans
nir_absorbed = 1 - nir_reflect - nir_trans
ratradnoon = 0.0

# Surface characteristics
# pft_ind = 10
# f_snow, f_cansno = 0.0, 0.0
# z_a, z0m, z0c, d = 5., 0.05, 0.05, 0.05
# gsoil, gstomatal = 1e10, 1.0 / 180.0

# Subsurface characteristics
# κ = 0.05
# dz_soil1 = z0


# ---------------------------------------------------------------------------- #
#                               Read forcing data                              #
# ---------------------------------------------------------------------------- #
# Atmospheric forcings
forcing_list = forcings.varn_list
rg_ind, pa_ind, lai_ind = (
    forcing_list.index("Rg"),
    forcing_list.index("PA"),
    forcing_list.index("LAI"),
)

# Check the shape of Thomson dispersion matrix
if dispersion_matrix.shape != (jtot3, jtot):
    raise Exception(
        "The shape of the divergence matrix is not identical to the domain size!"
    )

# ---------------------------------------------------------------------------- #
#                            Initialize the subjects                           #
# ---------------------------------------------------------------------------- #
time = Time(t0=t0, tn=tn, dt=dt, start_time="2018-01-01 00:00:00")
# soil_column = Column(xs=jnp.linspace(z0, zn, nz))
# Δz = soil_column.Δx
# surface = Surface(ts=time, space=Column(xs=soil_column.xs[:2]))
# soil = Soil(ts=time, space=soil_column, κ=None)

# ---------------------------------------------------------------------------- #
#                     Numerically solve the model over time                    #
# ---------------------------------------------------------------------------- #
t_prev, t_now = t0, t0 + dt  # t_now == t_prev for the initial step
tind_prev, tind_now = 0, 0

# JIT the functions
rnet, par, nir, gfunc = jax.jit(rnet), jax.jit(par), jax.jit(nir), jax.jit(gfunc)
diffuse_direct_radiation = jax.jit(diffuse_direct_radiation)
gfunc, irflux = jax.jit(gfunc), jax.jit(irflux)
angle, lai_time = jax.jit(angle), jax.jit(lai_time, static_argnames=["sze"])
stomata, soil_respiration = jax.jit(stomata), jax.jit(soil_respiration)
soil_energy_balance, set_soil = jax.jit(soil_energy_balance), jax.jit(set_soil)
friction_velocity, conc = jax.jit(friction_velocity), jax.jit(conc)
energy_and_carbon_fluxes = jax.jit(energy_and_carbon_fluxes)

# Some lists to get the data
Ta_all, co2_air_all = jnp.array([]), jnp.array([])
T_soil_all, Ci_all = jnp.array([]), jnp.array([])
sun_tleaf_all, shd_tleaf_all = jnp.array([]), jnp.array([])
rnet_sun_all, rnet_sh_all = jnp.array([]), jnp.array([])
sumh_all, sumle_all = jnp.array([]), jnp.array([])
sumrn_all = jnp.array([])
time_all = jnp.array([])

while t_now < tn:
    # ------------------------- Get the current time step ------------------------ #
    t_now_fmt = time.return_formatted_time(t_now)
    year, day = t_now_fmt.year, t_now_fmt.timetuple().tm_yday
    hour = t_now_fmt.hour + t_now_fmt.minute / 60.0

    # Get the forcing data
    (
        rglobal,
        parin,
        press_kpa,
        lai,
        ta,
        ws,
        ustar,
        co2,
        ea,
        ts,
        swc,
        T_Kelvin,
        rhova_g,
        rhova_kg,
        relative_humidity,
        vpd,
        press_bars,
        press_Pa,
        pstat273,
        gcut,
        rcuticle,
        air_density,
        air_density_mole,
        soil_Tave_15cm,
        heatcoef,
        H_old,
    ) = get_input_t(forcings, t_now)
    # jax.debug.print("rglobal: {a}; parin: {b}", a=rglobal, b=parin)

    print(f"Time: {day} day; Hour: {hour}.")
    # if (day==182) and (hour>8):
    #     exit()

    # ----------------------------- Evolve the model ----------------------------- #
    # Perform some initializations
    sun_tleaf, shd_tleaf = jnp.ones(sze) * ta, jnp.ones(sze) * ta
    sun_T_filter, shd_T_filter = jnp.ones(sze) * ta, jnp.ones(sze) * ta
    source_co2 = jnp.ones(jtot)
    tair, tair_filter = jnp.ones(sze3) * ta, jnp.ones(sze3) * ta
    rhov_air, rhov_filter = jnp.ones(sze3) * rhova_kg, jnp.ones(sze3) * rhova_kg
    can_co2_air = jnp.ones(sze3) * co2
    met_zl = 0.0
    sfc_temperature = ta
    zzz_ht = delz * jnp.arange(1, jtot + 1)
    # Soil properties initlization
    (
        soil_mtime,
        T_soil,
        z_soil,
        soil_bulk_density,
        cp_soil,
        k_conductivity_soil,
    ) = set_soil(
        dt_soil,
        total_t_soil,
        jnp.zeros(nsoil + 1),
        swc,
        ts,
        ta,
        air_density,
        air_density_mole,
        press_Pa,
    )
    # jax.debug.print("z_soil - {a}", a=z_soil)
    # exit()
    T_soil = jnp.concatenate([T_soil, jnp.array([ts])])  # nsoil+2
    soil_lout, soil_heat, soil_evap = 0.0, 0.0, 0.0

    # Update LAI structure with new day
    exxpdir, dLAIdz, Gfunc_sky = lai_time(sze, lai, ht, ht_midpt, lai_freq_scaled)

    # Compute solar elevation angle
    solar_beta_rad, solar_sine_beta, solar_beta_deg = angle(
        latitude, longitude, zone, year, day, hour
    )

    # Make sure PAR is zero at night
    parin = jax.lax.cond(solar_sine_beta <= 0.01, lambda: 0.0, lambda: parin)

    # Compute the fractions of beam and diffusion radiation from incoming measurements
    ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = jax.lax.cond(
        solar_sine_beta > 0.05,
        lambda x: diffuse_direct_radiation(x[0], x[1], x[2], x[3]),
        lambda x: (ratradnoon, 0.0, 0.0, 0.0, 0.0),
        [solar_sine_beta, rglobal, parin, press_kpa],
    )
    ratradnoon = jax.lax.cond(
        (hour > 12) & (hour < 13), lambda: ratrad, lambda: ratradnoon
    )

    # Comptue leaf inclination angle distributionf function, the mean direction cosine
    # between the sun zenith angle and the angle normal to the mean leaf
    Gfunc_solar = jax.lax.cond(
        solar_sine_beta >= 0.01,
        lambda x: gfunc(x[0], x[1]),
        lambda x: jnp.zeros(sze),
        [solar_beta_rad, dLAIdz],
    )

    # Compute PAR profiles
    (
        sun_lai,
        shd_lai,
        prob_beam,
        prob_sh,
        par_up,
        par_down,
        beam_flux_par,
        quantum_sh,
        quantum_sun,
        par_sh,
        par_sun,
    ) = par(
        solar_sine_beta,
        parin,
        par_beam,
        par_reflect,
        par_trans,
        par_soil_refl,
        par_absorbed,
        dLAIdz,
        exxpdir,
        Gfunc_solar,
    )

    # Compute NIR profiles
    nir_dn, nir_up, beam_flux_nir, nir_sh, nir_sun = nir(
        solar_sine_beta,
        nir_beam,
        nir_diffuse,
        nir_reflect,
        nir_trans,
        nir_soil_refl,
        nir_absorbed,
        dLAIdz,
        exxpdir,
        Gfunc_solar,
    )

    # Compute stomatal conductance for sunlit and
    # shaded leaf fractions as a function of light
    # on those leaves.
    sun_rs, shd_rs = stomata(
        lai,
        pai,
        rcuticle,
        par_sun,
        par_sh,
    )

    # Compute probability of penetration for diffuse
    # radiation for each layer in the canopy
    ir_up, ir_dn = irflux(
        T_Kelvin,
        ratrad,
        sfc_temperature,
        exxpdir,
        sun_T_filter,
        shd_T_filter,
        prob_beam,
        prob_sh,
    )

    # jax.debug.print("ir_up: {a}; ir_dn: {b}", a=ir_up, b=ir_dn)

    # Iteration looping for energy fluxes and scalar fields
    # iterate until energy balance closure occurs or 75 iterations are reached
    def iteration(c, iter_step):
        (
            ir_dn,
            ir_up,
            par_sun,
            nir_sun,
            par_sh,
            nir_sh,
            rnet_sun,
            rnet_sh,
            sun_rs,
            shd_rs,
            sun_gs,
            shd_gs,
            sun_resp,
            shd_resp,
            sun_wj,
            shd_wj,
            sun_wc,
            shd_wc,
            sun_A,
            shd_A,
            sun_rbh,
            shd_rbh,
            sun_rbv,
            shd_rbv,
            sun_rbco2,
            shd_rbco2,
            sun_ci,
            shd_ci,
            sun_cica,
            shd_cica,
            sun_tleaf,
            shd_tleaf,
            dLEdz,
            dHdz,
            dRNdz,
            dPsdz,
            Ci,
            drbv,
            dRESPdz,
            dStomCondz,
            soil_rnet,
            soil_lout,
            soil_heat,
            soil_evap,
            latent,
            sfc_temperature,
            T_soil,
            tair,
            rhov_air,
            tair_filter,
            rhov_filter,
            sun_T_filter,
            shd_T_filter,
            source_co2,
            can_co2_air,
            can_ps_mol,
            canresp,
            sumksi,
            sumlai,
            sumrv,
            sumCi,
            sumh,
            sumle,
            sumrn,
            tavg_sun,
            tavg_shade,
            tleaf_mean,
            H_old,
            zl,
            netrad,
            NIRv,
        ) = c
        # jax.debug.print("tair_filter: {a}", a=tair_filter)
        # Compute net radiation balance on sunlit and shaded leaves
        rnet_sun, rnet_sh = rnet(ir_dn, ir_up, par_sun, nir_sun, par_sh, nir_sh)

        # Compute leaf energy balance, leaf temperature, photosynthesis
        # and stomatal conductance.
        # jax.debug.print("tair_filter: {a}", a=tair_filter)
        (
            sun_rs_update,
            shd_rs_update,
            sun_gs,
            shd_gs,
            sun_resp,
            shd_resp,
            sun_wj,
            shd_wj,
            sun_wc,
            shd_wc,
            sun_A,
            shd_A,
            sun_rbh,
            shd_rbh,
            sun_rbv,
            shd_rbv,
            sun_rbco2,
            shd_rbco2,
            sun_ci,
            shd_ci,
            sun_cica,
            shd_cica,
            sun_tleaf_update,
            shd_tleaf_update,
            dLEdz,
            dHdz,
            dRNdz,
            dPsdz,
            Ci,
            drbv,
            dRESPdz,
            dStomCondz,
        ) = energy_and_carbon_fluxes(
            ht,
            grasshof,
            press_kpa,
            co2,
            ws,
            pr33,
            sc33,
            scc33,
            air_density,
            lai,
            pai,
            pstat273,
            kball,
            tair_filter,
            zzz_ht,
            prob_beam,
            prob_sh,
            rnet_sun,
            rnet_sh,
            quantum_sun,
            quantum_sh,
            can_co2_air,
            rhov_air,
            rhov_filter,
            dLAIdz,
            sun_rs,
            shd_rs,
            sun_tleaf,
            shd_tleaf,
        )
        sun_rs = jnp.concatenate([sun_rs_update, sun_rs[jtot:]])
        shd_rs = jnp.concatenate([shd_rs_update, shd_rs[jtot:]])
        sun_tleaf = jnp.concatenate([sun_tleaf_update, sun_tleaf[jtot:]])
        shd_tleaf = jnp.concatenate([shd_tleaf_update, shd_tleaf[jtot:]])
        # jax.debug.print("prob_beam: {a}", a=prob_beam)
        # jax.debug.print("Ci: {a}", a=Ci)

        # Soil energy balance
        # jax.debug.print("1 - soil_heat: {b}; soil_evap: {a}", a=soil_evap,b=soil_heat)
        (
            soil_rnet,
            soil_lout,
            soil_heat,
            soil_evap,
            latent,
            sfc_temperature,
            T_soil,
        ) = soil_energy_balance(
            epsoil,
            delz,
            ht,
            ws,
            air_density,
            relative_humidity,
            press_Pa,
            ta,
            swc,
            ts,
            beam_flux_par[0],
            par_down[0],
            par_up[0],
            beam_flux_nir[0],
            nir_dn[0],
            nir_up[0],
            ir_dn[0],
            tair_filter[0],
            sfc_temperature,
            rhov_filter[0],
            soil_bulk_density[1],
            soil_lout,
            soil_heat,
            soil_evap,
            soil_mtime,
            # soil_bulk_density[0], soil_lout, soil_heat, soil_evap, soil_mtime,
            iter_step,
            k_conductivity_soil,
            cp_soil,
        )
        # jax.debug.print("2 - soil_heat: {b}; soil_evap: {a}", a=soil_evap,b=soil_heat)

        # Update long wave radiation fluxes with new leaf and air temperatures
        ir_up, ir_dn = irflux(
            T_Kelvin,
            ratrad,
            sfc_temperature,
            exxpdir,
            sun_T_filter,
            shd_T_filter,
            prob_beam,
            prob_sh,
        )

        # Filter temperatures with each interation to minimize numerical instability
        a_filt, b_filt = jax.lax.cond(
            iter_step < 10, lambda: (0.5, 0.5), lambda: (0.85, 0.15)
        )

        # conc, for temperature profiles using source/sinks
        # jax.debug.print("iter: {a}; a_filt: {b}", a=iter_step, b=a_filt)
        tair_update = conc(
            ta,
            soil_heat,
            heatcoef,
            met_zl,
            delz,
            izref,
            ustar_ref,
            ustar,
            dHdz,
            dispersion_matrix,
        )
        # jax.debug.print("heatcoef: {b}; tair_update: {a}", a=tair_update, b=heatcoef)
        tair = jnp.concatenate([tair_update, tair[jtot3:]])
        # jax.debug.print("tair: {a}", a=tair)

        # Filter temperatures to remove numerical instabilities for each iteration
        tair = filter_array(tair, -10.0, 60.0, ta)
        tair_filter = tair * a_filt + tair_filter * b_filt
        # jax.debug.print("tair_filter: {a}", a=tair_filter)

        # Compute filtered sunlit and shaded temperatures
        # these are used to compute iterated longwave emissive energy fluxes
        sun_T_filter = a_filt * sun_tleaf + b_filt * sun_T_filter
        shd_T_filter = a_filt * shd_tleaf + b_filt * shd_T_filter

        # the arrays dLEdz and prof.rhov_air define pointers
        rhov_air_update = conc(
            rhova_kg,
            soil_evap,
            latent,
            met_zl,
            delz,
            izref,
            ustar_ref,
            ustar,
            dLEdz,
            dispersion_matrix,
        )
        rhov_air = jnp.concatenate([rhov_air_update, rhov_air[jtot3:]])

        # Filter humidity computations
        rhov_air = filter_array(rhov_air, 0.0, 0.03, rhova_kg)
        rhov_filter = a_filt * rhov_air + b_filt * rhov_filter

        # Compute soil respiration
        # Convert to umol m-2 s-1
        soil_resp_mole, soil_resp_mg = soil_respiration(T_soil[7], 8.0)

        # sign convention used: photosynthetic uptake is positive
        # respiration is negative
        # prof.dPsdz is net photosynthesis per unit leaf area,
        # the profile was converted to units of mg m-2 s-1 to be
        # consistent with inputs to CONC
        # change sign of dPsdz
        source_co2 = -dPsdz

        # To convert umol m-3 to umol/mol we have to consider
        # Pc/Pa = [CO2]ppm = rhoc ma/ rhoa mc
        co2_fact = (
            mass_air / mass_CO2
        ) * air_density_mole  # CO2 factor, ma/mc * rhoa (mole m-3)  # noqa: E501
        can_co2_air_update = conc(
            co2,
            soil_resp_mole,
            co2_fact,
            met_zl,
            delz,
            izref,
            ustar_ref,
            ustar,
            source_co2,
            dispersion_matrix,
        )
        can_co2_air = jnp.concatenate([can_co2_air_update, can_co2_air[jtot3:]])

        # Integrate source-sink strengths to estimate canopy flux
        sumh = jnp.sum(dHdz[:jtot])
        sumle = jnp.sum(dLEdz[:jtot])
        can_ps_mol = jnp.sum(dPsdz[:jtot])
        canresp = jnp.sum(dRESPdz[:jtot])
        sumksi = jnp.sum(dStomCondz[:jtot])
        sumrn = jnp.sum(dRNdz[:jtot])
        sumlai = jnp.sum(dLAIdz[:jtot])
        sumrv = jnp.sum(drbv[:jtot])
        # need to weight by sun and shaded leaf areas then divide by LAI
        tavg_sun = jnp.sum(sun_tleaf[:jtot] * dLAIdz[:jtot])
        tavg_shade = jnp.sum(shd_tleaf[:jtot] * dLAIdz[:jtot])

        # Mean canopy leaf temperature and CO2
        tleaf_mean = jnp.mean(
            sun_tleaf[:jtot] * prob_beam[:jtot] + shd_tleaf[:jtot] * prob_sh[:jtot]
        )
        sumCi = jnp.mean(Ci[:jtot])

        # Leaf area weighted temperatures
        tavg_sun /= lai
        tavg_shade /= lai

        # Energy exchanges at the soil
        rnet_soil = soil_rnet - soil_lout

        # Canopy scale flux densities, vegetation plus soil
        sumh += soil_heat
        sumle += soil_evap
        sumrn += rnet_soil

        # Re-compute Monin Obuhkov scale length and new ustar values with iterated H
        H_old, zl = friction_velocity(ustar, H_old, sumh, air_density, T_Kelvin)

        # Net radiation balance at the top of the canopy
        netrad = (
            (beam_flux_par[jtot] + par_down[jtot] - par_up[jtot]) / 4.6
            + beam_flux_nir[jtot]
            + nir_dn[jtot]
            - nir_up[jtot]
            + ir_dn[jtot]
            - ir_up[jtot]
        )
        NIRv = nir_up[jtot] - nir_up[0]  # NIR emitted by vegetation, total - soil

        # jax.debug.print("Ci-b: {a}", a=Ci)
        c_new = (
            ir_dn,
            ir_up,
            par_sun,
            nir_sun,
            par_sh,
            nir_sh,
            rnet_sun,
            rnet_sh,
            sun_rs,
            shd_rs,
            sun_gs,
            shd_gs,
            sun_resp,
            shd_resp,
            sun_wj,
            shd_wj,
            sun_wc,
            shd_wc,
            sun_A,
            shd_A,
            sun_rbh,
            shd_rbh,
            sun_rbv,
            shd_rbv,
            sun_rbco2,
            shd_rbco2,
            sun_ci,
            shd_ci,
            sun_cica,
            shd_cica,
            sun_tleaf,
            shd_tleaf,
            dLEdz,
            dHdz,
            dRNdz,
            dPsdz,
            Ci,
            drbv,
            dRESPdz,
            dStomCondz,
            soil_rnet,
            soil_lout,
            soil_heat,
            soil_evap,
            latent,
            sfc_temperature,
            T_soil,
            tair,
            rhov_air,
            tair_filter,
            rhov_filter,
            sun_T_filter,
            shd_T_filter,
            source_co2,
            can_co2_air,
            can_ps_mol,
            canresp,
            sumksi,
            sumlai,
            sumrv,
            sumCi,
            sumh,
            sumle,
            sumrn,
            tavg_sun,
            tavg_shade,
            tleaf_mean,
            H_old,
            zl,
            netrad,
            NIRv,
        )

        return c_new, c_new

    rnet_sun, rnet_sh = jnp.zeros(sze), jnp.zeros(sze)
    sun_gs, shd_gs = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_resp, shd_resp = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_wj, shd_wj = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_wc, shd_wc = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_A, shd_A = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_rbh, shd_rbh = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_rbv, shd_rbv = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_rbco2, shd_rbco2 = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_ci, shd_ci = jnp.zeros(jtot), jnp.zeros(jtot)
    sun_cica, shd_cica = jnp.zeros(jtot), jnp.zeros(jtot)
    dLEdz, dHdz = jnp.zeros(jtot), jnp.zeros(jtot)
    dRNdz, dPsdz = jnp.zeros(jtot), jnp.zeros(jtot)
    Ci, drbv = jnp.zeros(jtot), jnp.zeros(jtot)
    dRESPdz, dStomCondz = jnp.zeros(jtot), jnp.zeros(jtot)
    soil_rnet, latent, sumh, sumle, can_ps_mol = 0.0, 0.0, 0.0, 0.0, 0.0
    canresp, sumksi, sumrn, sumlai, sumrv = 0.0, 0.0, 0.0, 0.0, 0.0
    tavg_sun, tavg_shade, tleaf_mean, sumCi = 0.0, 0.0, 0.0, 0.0
    zl, netrad, NIRv = 0.0, 0.0, 0.0
    init = (
        ir_dn,
        ir_up,
        par_sun,
        nir_sun,
        par_sh,
        nir_sh,
        rnet_sun,
        rnet_sh,
        sun_rs,
        shd_rs,
        sun_gs,
        shd_gs,
        sun_resp,
        shd_resp,
        sun_wj,
        shd_wj,
        sun_wc,
        shd_wc,
        sun_A,
        shd_A,
        sun_rbh,
        shd_rbh,
        sun_rbv,
        shd_rbv,
        sun_rbco2,
        shd_rbco2,
        sun_ci,
        shd_ci,
        sun_cica,
        shd_cica,
        sun_tleaf,
        shd_tleaf,
        dLEdz,
        dHdz,
        dRNdz,
        dPsdz,
        Ci,
        drbv,
        dRESPdz,
        dStomCondz,
        soil_rnet,
        soil_lout,
        soil_heat,
        soil_evap,
        latent,
        sfc_temperature,
        T_soil,
        tair,
        rhov_air,
        tair_filter,
        rhov_filter,
        sun_T_filter,
        shd_T_filter,
        source_co2,
        can_co2_air,
        can_ps_mol,
        canresp,
        sumksi,
        sumlai,
        sumrv,
        sumCi,
        sumh,
        sumle,
        sumrn,
        tavg_sun,
        tavg_shade,
        tleaf_mean,
        H_old,
        zl,
        netrad,
        NIRv,
    )
    carry, _ = jax.lax.scan(
        # iteration, init=init, xs=jnp.arange(2)
        iteration,
        init=init,
        xs=jnp.arange(50),
    )
    (
        ir_dn,
        ir_up,
        par_sun,
        nir_sun,
        par_sh,
        nir_sh,
        rnet_sun,
        rnet_sh,
        sun_rs,
        shd_rs,
        sun_gs,
        shd_gs,
        sun_resp,
        shd_resp,
        sun_wj,
        shd_wj,
        sun_wc,
        shd_wc,
        sun_A,
        shd_A,
        sun_rbh,
        shd_rbh,
        sun_rbv,
        shd_rbv,
        sun_rbco2,
        shd_rbco2,
        sun_ci,
        shd_ci,
        sun_cica,
        shd_cica,
        sun_tleaf,
        shd_tleaf,
        dLEdz,
        dHdz,
        dRNdz,
        dPsdz,
        Ci,
        drbv,
        dRESPdz,
        dStomCondz,
        soil_rnet,
        soil_lout,
        soil_heat,
        soil_evap,
        latent,
        sfc_temperature,
        T_soil,
        tair,
        rhov_air,
        tair_filter,
        rhov_filter,
        sun_T_filter,
        shd_T_filter,
        source_co2,
        can_co2_air,
        can_ps_mol,
        canresp,
        sumksi,
        sumlai,
        sumrv,
        sumCi,
        sumh,
        sumle,
        sumrn,
        tavg_sun,
        tavg_shade,
        tleaf_mean,
        H_old,
        zl,
        netrad,
        NIRv,
    ) = carry

    T_soil_all, Ci_all = jnp.append(T_soil_all, T_soil), jnp.append(Ci_all, Ci)
    sun_tleaf_all = jnp.append(sun_tleaf_all, sun_tleaf)
    shd_tleaf_all = jnp.append(shd_tleaf_all, shd_tleaf)
    rnet_sun_all = jnp.append(rnet_sun_all, rnet_sun)
    rnet_sh_all = jnp.append(rnet_sh_all, rnet_sh)
    sumh_all = jnp.append(sumh_all, sumh)
    sumle_all = jnp.append(sumle_all, sumle)
    sumrn_all = jnp.append(sumrn_all, sumrn)
    Ta_all = jnp.append(Ta_all, ta)
    co2_air_all = jnp.append(co2_air_all, co2)
    time_all = jnp.append(time_all, t_now)

    jax.debug.print("T_soil: {a}; Ci: {b}", a=T_soil, b=Ci)

    # Update the time step
    t_prev = t_now
    t_now = min(t_now + dt, tn)

    # Update the time indices
    tind_prev = tind_now
    tind_now = min(tind_now + 1, time.nt)

nt = tind_now
T_soil_all = T_soil_all.reshape([nt, nsoil + 2])
Ci_all = Ci_all.reshape([nt, jtot])
sun_tleaf_all = sun_tleaf_all.reshape([nt, sze])
shd_tleaf_all = shd_tleaf_all.reshape([nt, sze])
rnet_sun_all = rnet_sun_all.reshape([nt, sze])
rnet_sh_all = rnet_sh_all.reshape([nt, sze])

array_to_save = [
    T_soil_all,
    Ci_all,
    sun_tleaf_all,
    shd_tleaf_all,
    rnet_sun_all,
    rnet_sh_all,
    sumh_all,
    sumle_all,
    sumrn_all,
    Ta_all,
    co2_air_all,
    time_all,
]
array_names = [
    "T_soil",
    "Ci",
    "sun_tleaf",
    "shd_tleaf",
    "rnet_sun",
    "rnet_sh",
    "sumh",
    "sumle",
    "sumrn",
    "Ta",
    "co2_air",
    "time",
]
# Write the data into an h5 file
with h5py.File("simulation.h5", "w") as f:
    for i, array_jnp in enumerate(array_to_save):
        name, shp = array_names[i], array_jnp.shape
        dset = f.create_dataset(name, shp, dtype="float64")
        dset[:] = np.array(array_jnp)
