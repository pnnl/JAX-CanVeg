"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.08.13.
"""

import jax
import jax.numpy as jnp

# import h5py
import numpy as np

import equinox as eqx

# from jax_canoak.subjects import Para

from jax_canoak.subjects import initialize_met, initialize_parameters
from jax_canoak.subjects import initialize_profile, initialize_model_states
from jax_canoak.physics import generate_night_mask, generate_turbulence_mask
from jax_canoak.subjects import update_profile, calculate_veg

from jax_canoak.shared_utilities.utils import dot
from jax_canoak.physics import energy_carbon_fluxes
from jax_canoak.physics.energy_fluxes import diffuse_direct_radiation

# from jax_canoak.physics.energy_fluxes import rad_tran_canopy, sky_ir_v2
from jax_canoak.physics.energy_fluxes import rad_tran_canopy, sky_ir
from jax_canoak.physics.energy_fluxes import compute_qin, ir_rad_tran_canopy
from jax_canoak.physics.energy_fluxes import uz, soil_energy_balance
from jax_canoak.physics.carbon_fluxes import angle, leaf_angle


import matplotlib.pyplot as plt
from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
from jax_canoak.shared_utilities.plot import plot_veg_temp

from jax_canoak.shared_utilities.plot import plot_ir, plot_rad, plot_prof2

# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)

plot = True

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
time_zone = -8
latitude = 38.0991538
longitude = -121.49933
stomata = 2
hypo_amphi = 1
veg_ht = 0.8
leafangle = 1
n_can_layers = 50
meas_ht = 5.0
n_hr_per_day = 48
lai = 5.0


# ---------------------------------------------------------------------------- #
#                     Set the model forcings                                   #
# ---------------------------------------------------------------------------- #
# f_forcing = "../shared_utilities/forcings/AlfMetBouldinInput.csv"
f_forcing = "../shared_utilities/forcings/AlfBouldinMetInput.csv"
forcing_data = np.loadtxt(f_forcing, delimiter=",")
forcing_data = jnp.array(forcing_data)
n_time = forcing_data.shape[0]
zl0 = jnp.zeros(n_time)
forcing_data = jnp.concatenate([forcing_data, jnp.ones([n_time, 1]) * lai], axis=1)
met = initialize_met(forcing_data, n_time, zl0)


# ---------------------------------------------------------------------------- #
#                     Set up model parameter instance                      #
# ---------------------------------------------------------------------------- #
para = initialize_parameters(
    # para = Para(
    time_zone=time_zone,
    latitude=latitude,
    longitude=longitude,
    stomata=stomata,
    hypo_amphi=hypo_amphi,
    veg_ht=veg_ht,
    leafangle=leafangle,
    n_can_layers=n_can_layers,
    meas_ht=meas_ht,
    n_hr_per_day=n_hr_per_day,
    n_time=n_time,
    npart=int(1e6),
)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
# dij = disp_canveg(para, timemax=5000.0)
dij = np.loadtxt("Dij_Alfalfa.csv", delimiter=",")
dij = jnp.array(dij)


# ---------------------------------------------------------------------------- #
#                     Initialize profiles of scalars/sources/sinks             #
# ---------------------------------------------------------------------------- #
prof = initialize_profile(met, para)


# ---------------------------------------------------------------------------- #
#                     Initialize model states                        #
# ---------------------------------------------------------------------------- #
soil, quantum, nir, ir, qin, rnet, sun, shade, veg, lai = initialize_model_states(
    met, para
)
soil_mtime = int(soil.mtime)


# ---------------------------------------------------------------------------- #
#                     Compute sun angles                                       #
# ---------------------------------------------------------------------------- #
sun_ang = angle(para.lat_deg, para.long_deg, para.time_zone, met.day, met.hhour)


# ---------------------------------------------------------------------------- #
#                     Compute leaf angle                                       #
# ---------------------------------------------------------------------------- #
leaf_ang = leaf_angle(sun_ang, para, lai)


# ---------------------------------------------------------------------------- #
#                     Compute direct and diffuse radiations                    #
# ---------------------------------------------------------------------------- #
ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = diffuse_direct_radiation(
    sun_ang.sin_beta, met.rglobal, met.parin, met.P_kPa
)
quantum = eqx.tree_at(
    lambda t: (t.inbeam, t.indiffuse), quantum, (par_beam, par_diffuse)
)
nir = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), nir, (nir_beam, nir_diffuse))


# ---------------------------------------------------------------------------- #
#                     Generate masks for matrix calculations                   #
# ---------------------------------------------------------------------------- #
# Night mask
mask_night_hashable = generate_night_mask(sun_ang)
# Turbulence mask
mask_turbulence_hashable = generate_turbulence_mask(para, met, prof)


# ---------------------------------------------------------------------------- #
#                     Initialize IR fluxes with air temperature                #
# ---------------------------------------------------------------------------- #
ir_in = sky_ir(met.T_air_K, ratrad, para.sigma)
# ir_in = sky_ir_v2(met, ratrad, para.sigma)
ir_dn = dot(ir_in, ir.ir_dn)
ir_up = dot(ir_in, ir.ir_up)
ir = eqx.tree_at(lambda t: (t.ir_in, t.ir_dn, t.ir_up), ir, (ir_in, ir_dn, ir_up))


# ---------------------------------------------------------------------------- #
#                     Compute radiation fields             #
# ---------------------------------------------------------------------------- #
# PAR
quantum = rad_tran_canopy(
    sun_ang, leaf_ang, quantum, para, lai, mask_night_hashable, niter=5
)
# NIR
nir = rad_tran_canopy(sun_ang, leaf_ang, nir, para, lai, mask_night_hashable, niter=25)


# ---------------------------------------------------------------------------- #
#                     Iterations                                               #
# ---------------------------------------------------------------------------- #
# compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
# loop again and apply updated Tsfc info until convergence
# This is where things should be jitted as a whole
def iteration(c, i):
    met, prof, ir, qin, sun, shade, soil, veg = c
    # jax.debug.print("T soil: {a}", a=soil.T_soil[10,:])
    jax.debug.print("T sfc: {a}", a=soil.sfc_temperature[10])

    # Update canopy wind profile with iteration of z/l and use in boundary layer
    # resistance computations
    wind = uz(met, para)
    prof = eqx.tree_at(lambda t: t.wind, prof, wind)

    # Compute IR fluxes with Bonan's algorithms of Norman model
    ir = ir_rad_tran_canopy(leaf_ang, ir, quantum, soil, sun, shade, para)
    # jax.debug.print("ir: {a}", a=ir.ir_dn[10,:])

    # Incoming short and longwave radiation
    qin = compute_qin(quantum, nir, ir, para, qin)

    # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
    # Compute new boundary layer conductances based on new leaf energy balance
    # and delta T, in case convection occurs
    # Different coefficients will be assigned if amphistomatous or hypostomatous
    sun, shade = energy_carbon_fluxes(
        sun, shade, qin, quantum, met, prof, para, mask_turbulence_hashable
    )

    # Compute soil fluxes
    # soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil.mtime)  # type: ignore  # noqa: E501
    soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil_mtime)  # type: ignore  # noqa: E501

    # Compute profiles of C's, zero layer jtot+1 as that is not a dF/dz or
    # source/sink level
    prof = update_profile(met, para, prof, quantum, sun, shade, soil, veg, lai, dij)

    # compute met.zL from HH and met.ustar
    HH = jnp.sum(
        (
            quantum.prob_beam[:, : para.jtot] * sun.H
            + quantum.prob_shade[:, : para.jtot] * shade.H
        )
        * lai.dff[:, : para.jtot],
        axis=1,
    )
    zL = -(0.4 * 9.8 * HH * para.meas_ht) / (
        met.air_density * 1005 * met.T_air_K * jnp.power(met.ustar, 3.0)
    )
    zL = jnp.clip(zL, a_min=-3, a_max=0.25)
    met = eqx.tree_at(lambda t: t.zL, met, zL)

    # Compute canopy integrated fluxes
    veg = calculate_veg(para, lai, quantum, sun, shade)

    cnew = [met, prof, ir, qin, sun, shade, soil, veg]
    return cnew, None


initials = [met, prof, ir, qin, sun, shade, soil, veg]
finals, _ = jax.lax.scan(iteration, initials, xs=None, length=15)

met, prof, ir, qin, sun, shade, soil, veg = finals


# Net radiation budget at top of the canopy
can_rnet = (
    quantum.beam_flux[:, para.jtot] / 4.6
    + quantum.dn_flux[:, para.jtot] / 4.6
    - quantum.up_flux[:, para.jtot] / 4.6
    + nir.beam_flux[:, para.jtot]
    + nir.dn_flux[:, para.jtot]
    - nir.up_flux[:, para.jtot]
    + ir.ir_dn[:, para.jtot]
    + -ir.ir_up[:, para.jtot]
)

# ---------------------------------------------------------------------------- #
#                     Plot                                              #
# ---------------------------------------------------------------------------- #
if plot:
    plot_rad(quantum, para, lai, "par")
    plot_rad(nir, para, lai, "nir")
    plot_ir(ir, para, lai)
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # plot_canopy1(sun, qin, para, "sun", axes[0])
    # plot_canopy1(shade, qin, para, "shade", axes[1])
    # plot_canopy2(sun, para, "sun")
    # plot_canopy2(shade, para, "shade")
    # plot_leafang(leaf_ang, para)
    plot_veg_temp(sun, shade, para)
    plot_dij(dij, para)
    # plot_soil(soil, para)
    # plot_soiltemp(soil, para)
    # plot_totalenergy(soil, veg, can_rnet)
    # plot_prof1(prof)
    plot_prof2(prof, para)
    plot_daily(met, soil, veg, para)
    plt.show()
