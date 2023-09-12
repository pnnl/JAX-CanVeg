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

from math import floor

from functools import partial

import equinox as eqx

from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import CanoakBase
from jax_canoak.subjects import convert_met_to_batched_met
from jax_canoak.subjects import convert_batchedstates_to_states

import matplotlib.pyplot as plt
from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
from jax_canoak.shared_utilities.plot import plot_veg_temp

from jax_canoak.shared_utilities.plot import plot_ir, plot_rad, plot_prof2

# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

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
# n_can_layers = 10
# n_atmos_layers = 10
n_can_layers = 50
n_atmos_layers = 50
meas_ht = 5.0
soil_depth = 0.15
n_hr_per_day = 48
time_batch_size = 2
# time_batch_size = 1
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
    n_atmos_layers=n_atmos_layers,
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
dij = get_dispersion_matrix(setup, para, "../data/dij/Dij_US-Hn1.csv")


# ---------------------------------------------------------------------------- #
#                     Run CANOAK!                #
# ---------------------------------------------------------------------------- #
# Let's use a jitted version
canoak_eqx = CanoakBase(para, setup, dij)
n_batch, batch_size = floor(setup.ntime / time_batch_size), time_batch_size
batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
print("Performing forward runs ...")
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


@eqx.filter_jit
def compute_grad(c, met):
    para = c
    grad = df_canoak_le(para, met, canoak_eqx)
    return c, grad


jax.debug.print("canoak_eqx: {a}", a=canoak_eqx)

_, gradients = jax.lax.scan(compute_grad, para, xs=batched_met)

# gradients = jax.vmap(df_canoak_le, in_axes=[None,0,None])(
#     para, batched_met, canoak_eqx
# )

# gradients4 = df_canoak_le(para, batched_met, canoak_eqx)

# start, ntime = 20, 1000
# canoak_eqx2 = eqx.tree_at(lambda t: (t.ntime), canoak_eqx, replace=(ntime))
# met2 = jax.tree_util.tree_map(lambda x: x[start : start + ntime], met)

# # jax.profiler.start_trace("./memory_us-hn1")
# gradients = df_canoak_le(para, met2, canoak_eqx2)
# # jax.profiler.stop_trace()

# # jax.profiler.save_device_memory_profile("memory.prof")


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
