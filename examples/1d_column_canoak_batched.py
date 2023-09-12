"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.08.13.
"""

import jax

# import jax.numpy as jnp
# import jax.tree_util as jtu

from math import floor

# import h5py
# import numpy as np

# import equinox as eqx

from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import CanoakBase
from jax_canoak.subjects import convert_met_to_batched_met
from jax_canoak.subjects import convert_batchedstates_to_states


# import matplotlib.pyplot as plt
# from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
# from jax_canoak.shared_utilities.plot import plot_veg_temp
# from jax_canoak.shared_utilities.plot import plot_ir, plot_rad, plot_prof2

# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

plot = False

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
time_zone = -8
latitude = 38.0991538
longitude = -121.49933
stomata = 1
veg_ht = 0.8
leafangle = 1
n_can_layers = 50
meas_ht = 5.0
n_hr_per_day = 48
time_batch_size = 2
lai = 5.0
niter = 15
f_forcing = "../data/fluxtower/Alf/AlfBouldinMetInput-yr.csv"


# ---------------------------------------------------------------------------- #
#                     Get the model forcings                                   #
# ---------------------------------------------------------------------------- #
met, n_time = get_met_forcings(f_forcing, lai)


# ---------------------------------------------------------------------------- #
#                     Set up model parameter instance                      #
# ---------------------------------------------------------------------------- #
setup, para = initialize_parameters(
    time_zone=time_zone,
    latitude=latitude,
    longitude=longitude,
    stomata=stomata,
    veg_ht=veg_ht,
    leafangle=leafangle,
    n_can_layers=n_can_layers,
    meas_ht=meas_ht,
    n_hr_per_day=n_hr_per_day,
    n_time=n_time,
    time_batch_size=time_batch_size,
    npart=int(1e6),
    niter=niter,
)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
# dij = get_dispersion_matrix(setup, para)
dij = get_dispersion_matrix(setup, para, "../data/dij/Dij_Alfalfa.csv")


# ---------------------------------------------------------------------------- #
#                     Run CANOAK!                #
# ---------------------------------------------------------------------------- #
# Let's use a jitted version
# canoak = eqx.filter_jit(canoak)
canoak_eqx = CanoakBase(para, setup, dij)
n_batch, batch_size = floor(setup.ntime / time_batch_size), time_batch_size
batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
# jax.vmap(canoak_eqx)(batched_met)
(
    _,
    batched_prof,
    batched_quantum,
    batched_nir,
    batched_ir,
    batched_rnet,
    batched_qin,
    batched_sun_ang,
    batched_leaf_ang,
    batched_lai,
    batched_sun,
    batched_shade,
    batched_soil,
    batched_veg,
    batched_can,
) = jax.vmap(canoak_eqx)(batched_met)


# ---------------------------------------------------------------------------- #
#                     Reshape the results                #
# ---------------------------------------------------------------------------- #
(  # pyright: ignore
    prof,
    can,
    veg,
    shade,
    sun,
    qin,
    rnet,
    sun_ang,
    ir,
    nir,
    quantum,
    lai,
    leaf_ang,
    soil,
) = convert_batchedstates_to_states(
    batched_prof,
    batched_can,
    batched_veg,
    batched_shade,
    batched_sun,
    batched_qin,
    batched_rnet,
    batched_sun_ang,
    batched_ir,
    batched_nir,
    batched_quantum,
    batched_lai,
    batched_leaf_ang,
    batched_soil,
)

# ---------------------------------------------------------------------------- #
#                     Read observations                #
# ---------------------------------------------------------------------------- #
# f_obs = ''


# ---------------------------------------------------------------------------- #
#                     Plot                                              #
# ---------------------------------------------------------------------------- #
# if plot:
#     plot_rad(quantum, setup, lai, "par")
#     plot_rad(nir, setup, lai, "nir")
#     plot_ir(ir, setup, lai)
#     # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
#     # plot_canopy1(sun, qin, para, "sun", axes[0])
#     # plot_canopy1(shade, qin, para, "shade", axes[1])
#     # plot_canopy2(sun, para, "sun")
#     # plot_canopy2(shade, para, "shade")
#     # plot_leafang(leaf_ang, para)
#     plot_veg_temp(sun, shade, para, met)
#     plot_dij(dij, para)
#     # plot_soil(soil, para)
#     # plot_soiltemp(soil, para)
#     # plot_totalenergy(soil, veg, can_rnet)
#     # plot_prof1(prof)
#     plot_prof2(prof, para)
#     plot_daily(met, soil, veg, para)
#     plt.show()
