"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.08.13.
"""

import jax

# import jax.numpy as jnp

# import h5py
# import numpy as np

import equinox as eqx

from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import canoak

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
stomata = 1
veg_ht = 0.8
leafangle = 1
n_can_layers = 50
meas_ht = 5.0
n_hr_per_day = 48
lai = 5.0
niter = 15
f_forcing = "../shared_utilities/forcings/AlfBouldinMetInput.csv"


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
    npart=int(1e6),
    niter=niter,
)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
# dij = get_dispersion_matrix(setup, para)
dij = get_dispersion_matrix(setup, para, "Dij_Alfalfa.csv")


# ---------------------------------------------------------------------------- #
#                     Run CANOAK!                #
# ---------------------------------------------------------------------------- #
# Let's use a jitted version
canoak = eqx.filter_jit(canoak)
(
    met,
    prof,
    quantum,
    nir,
    ir,
    rnet,
    qin,
    sun_ang,
    leaf_ang,
    lai,
    sun,
    shade,
    soil,
    veg,
    can,
) = canoak(para, setup, met, dij, setup.soil_mtime, setup.niter)


# ---------------------------------------------------------------------------- #
#                     Read observations                #
# ---------------------------------------------------------------------------- #
# f_obs = ''


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
