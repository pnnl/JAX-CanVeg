"""
Outputing the make_jaxp for debugging.

Author: Peishi Jiang
Date: 2023.9.11.
"""

import jax

from functools import partial

import equinox as eqx

from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import CanoakBase
from jax_canoak.subjects import convert_met_to_batched_met

# from jax_canoak.subjects import convert_batchedstates_to_states


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
    npart=int(1e6),
    niter=niter,
)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
# dij = get_dispersion_matrix(setup, para)
dij = get_dispersion_matrix(setup, para, "../data/dij/Dij_Alfalfa.csv")


# ---------------------------------------------------------------------------- #
#                     Forward function and gradient function  #
# ---------------------------------------------------------------------------- #
canoak_eqx = CanoakBase(para, setup, dij)


@eqx.filter_jit
@partial(jax.grad, argnums=0)
def df_canoak_le(para, batched_met, canoak_eqx):
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
    ) = canoak_eqx(batched_met)
    return can.LE.sum()


# ---------------------------------------------------------------------------- #
#                     Batch size: 1  #
# ---------------------------------------------------------------------------- #
batch_size = 1
n_batch = setup.ntime
batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
jaxpr = eqx.filter_make_jaxpr(canoak_eqx)(batched_met)
jaxpr_grad = eqx.filter_make_jaxpr(df_canoak_le)(para, batched_met, canoak_eqx)

# Save
with open("jaxpr_bs1", "w") as file:
    jaxpr_str = str(jaxpr)
    file.write(jaxpr_str)
with open("jaxpr_grad_bs1", "w") as file:
    jaxpr_str = str(jaxpr_grad)
    file.write(jaxpr_str)


# ---------------------------------------------------------------------------- #
#                     Batch size: 816  #
# ---------------------------------------------------------------------------- #
batch_size = setup.ntime
n_batch = 1
batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
jaxpr = eqx.filter_make_jaxpr(canoak_eqx)(batched_met)
jaxpr_grad = eqx.filter_make_jaxpr(df_canoak_le)(para, batched_met, canoak_eqx)

# Save
with open("jaxpr_bs816", "w") as file:
    jaxpr_str = str(jaxpr)
    file.write(jaxpr_str)
with open("jaxpr_grad_bs816", "w") as file:
    jaxpr_str = str(jaxpr_grad)
    file.write(jaxpr_str)
