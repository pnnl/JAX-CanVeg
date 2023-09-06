"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.08.29.
"""

import jax

# import jax.numpy as jnp

# import h5py
# import numpy as np

from functools import partial

import equinox as eqx

from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import CanoakBase

import matplotlib.pyplot as plt
from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
from jax_canoak.shared_utilities.plot import plot_veg_temp

from jax_canoak.shared_utilities.plot import plot_ir, plot_rad, plot_prof2

# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_debug_infs", False)

plot = False

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
time_zone = -8
latitude = 46.4089
longitude = -119.2750
stomata = 0
veg_ht = 1.2
leafangle = 2  # erectophile
n_can_layers = 50
meas_ht = 5.0
soil_depth = 0.15
n_hr_per_day = 48
niter = 15
f_forcing = "../data/fluxtower/US-Hn1/US-Hn1-forcings.csv"


# ---------------------------------------------------------------------------- #
#                     Get the model forcings                                   #
# ---------------------------------------------------------------------------- #
met, n_time = get_met_forcings(f_forcing)


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
dij = get_dispersion_matrix(setup, para, "../data/dij/Dij_US-Hn1.csv")


# ---------------------------------------------------------------------------- #
#                     Run CANOAK!                #
# ---------------------------------------------------------------------------- #
# Let's use a jitted version
canoak_eqx = CanoakBase(para, setup, dij)
print("Performing forward runs ...")
(
    _,
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
) = canoak_eqx(met)


# ---------------------------------------------------------------------------- #
#                     Sensitivity analysis                #
# ---------------------------------------------------------------------------- #
print("Performing local sensitivity analysis ...")
jax.clear_caches()


@eqx.filter_jit
@partial(jax.grad, argnums=0)
def df_canoak_le(para, met, canoak_eqx):
    canoak_eqx = eqx.tree_at(lambda t: t.para, canoak_eqx, para)
    (
        _,
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
    ) = canoak_eqx(met)
    return can.LE.sum()


# start, ntime = 20, 1
start, ntime = 20, 1000
canoak_eqx2 = eqx.tree_at(lambda t: (t.ntime), canoak_eqx, replace=(ntime))
met2 = jax.tree_util.tree_map(lambda x: x[start : start + ntime], met)

jax.profiler.start_trace("./memory_us-hn1")
gradients = df_canoak_le(para, met2, canoak_eqx2)
jax.profiler.stop_trace()

# jax.profiler.save_device_memory_profile("memory.prof")


# ---------------------------------------------------------------------------- #
#                     Read observations                #
# ---------------------------------------------------------------------------- #
# f_obs = ''


# ---------------------------------------------------------------------------- #
#                     Plot                                              #
# ---------------------------------------------------------------------------- #
if plot:
    plot_rad(quantum, setup, lai, "par")
    plot_rad(nir, setup, lai, "nir")
    plot_ir(ir, setup, lai)
    plot_veg_temp(sun, shade, para, met)
    plot_dij(dij, para)
    plot_prof2(prof, para)
    plot_daily(met, soil, veg, para)
    plt.show()
