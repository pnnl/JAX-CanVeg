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
from jax_canoak.physics.energy_fluxes import diffuse_direct_radiation
from jax_canoak.physics.carbon_fluxes import angle, leaf_angle
from jax_canoak.models import canoak


import matplotlib.pyplot as plt
from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
from jax_canoak.shared_utilities.plot import plot_veg_temp

from jax_canoak.shared_utilities.plot import plot_ir, plot_rad, plot_prof2

# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)

plot = False

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
niter = 15


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
setup, para = initialize_parameters(
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
    niter=niter,
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
prof = initialize_profile(met, para, setup)


# ---------------------------------------------------------------------------- #
#                     Initialize model states                        #
# ---------------------------------------------------------------------------- #
soil, quantum, nir, ir, qin, rnet, sun, shade, veg, lai = initialize_model_states(
    met, para, setup
)
soil_mtime = int(soil.mtime)  # convert it to integer instead from jax array


# ---------------------------------------------------------------------------- #
#                     Compute sun angles                                       #
# ---------------------------------------------------------------------------- #
sun_ang = angle(setup.lat_deg, setup.long_deg, setup.time_zone, met.day, met.hhour)


# ---------------------------------------------------------------------------- #
#                     Compute leaf angle                                       #
# ---------------------------------------------------------------------------- #
leaf_ang = leaf_angle(sun_ang, para, setup, lai)


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
#                     Run CANOAK!                #
# ---------------------------------------------------------------------------- #
initials = [
    prof,
    dij,
    lai,
    sun_ang,
    leaf_ang,
    quantum,
    nir,
    ir,
    qin,
    ratrad,
    sun,
    shade,
    veg,
    soil,
    mask_night_hashable,
    mask_turbulence_hashable,
    int(soil_mtime),
    niter,
]

# Let's use a jitted version
canoak_jit = eqx.filter_jit(canoak)
met1, prof1, ir1, qin1, sun1, shade1, soil1, veg1 = canoak_jit(
    para, setup, met, *initials
)


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
