"""
Scripts for benchmarking time on batch implementation.

Author: Peishi Jiang
Date: 2023.09.12.
"""

from functools import partial

from time import time
from math import floor

# import numpy as np
import jax
import equinox as eqx

from jax_canoak.physics.energy_fluxes import get_dispersion_matrix
from jax_canoak.models import CanoakBase
from jax_canoak.subjects import convert_met_to_batched_met
from jax_canoak.subjects import get_met_forcings, initialize_parameters
from jax_canoak.models import run_canoak_in_batch

import matplotlib.pyplot as plt

# Parameters and setup
time_zone = -8
latitude = 46.4089
longitude = -119.2750
stomata = 0
veg_ht = 1.2
leafangle = 2  # erectophile
n_can_layers = 50
n_atmos_layers = 50
meas_ht = 5.0
soil_depth = 0.15
n_hr_per_day = 48
niter = 15
f_forcing = "../data/fluxtower/US-Hn1/US-Hn1-forcings.csv"


# Load model forcings
met, n_time = get_met_forcings(f_forcing)


# Create the model parameter instance
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
    soil_depth=soil_depth,
    n_hr_per_day=n_hr_per_day,
    n_time=n_time,
    npart=int(1e6),
    niter=niter,
)


# Generate or read the dispersion matrix
dij = get_dispersion_matrix(setup, para)


# Run canoak through different batch sizes
print("Forward runs ...")
canoak_eqx = CanoakBase(para, setup, dij)
batch_size_set = [1, 2, 4, 8, 16, 48, 100, 240, 1274, 3792, 18960]
batch_size_set.reverse()
time_set = []
for i, batch_size in enumerate(batch_size_set):
    print(f"Batch size: {batch_size}")
    n_batch = floor(n_time / batch_size)
    batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
    run_canoak_in_batch(batched_met, canoak_eqx)
    start = time()
    run_canoak_in_batch(batched_met, canoak_eqx)
    end = time()
    simulation_time = end - start
    print(f"Simulation time: {simulation_time} sec")
    time_set.append(simulation_time)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(batch_size_set, time_set, "k*-")
ax.set(
    ylabel="Execution time [s]",
    xlabel="Batch size",
    title="Forward run (US-Hn1)",
    yscale="log",
    xscale="log",
)
plt.savefig("Benchmark_time_batch_forwardrun.png", dpi=150)


# Calculate gradients through different batch sizes
# jax.clear_caches()
@eqx.filter_jit
@partial(jax.grad, argnums=0)
def df_canoak_le(para, met, canoak_eqx):
    canoak_eqx = eqx.tree_at(lambda t: t.para, canoak_eqx, para)
    results = canoak_eqx(met)
    can = results[-1]
    return can.LE.sum()


@eqx.filter_jit
def compute_grad(para, met):
    grad = df_canoak_le(para, met, canoak_eqx)
    return para, grad


@eqx.filter_jit
def df_canoak_le_batched(para, batched_met, canoak_eqx):
    def compute_grad(para, met):
        grad = df_canoak_le(para, met, canoak_eqx)
        return para, grad

    _, gradients = jax.lax.scan(compute_grad, para, xs=batched_met)
    return gradients


print("Compute gradients ...")
batch_size_set = [1, 2, 4, 8, 16, 48, 100, 240, 1274, 3792]
batch_size_set.reverse()
time_set = []
for i, batch_size in enumerate(batch_size_set):
    print(f"Batch size: {batch_size}")
    n_batch = floor(n_time / batch_size)
    batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
    df_canoak_le_batched(para, batched_met, canoak_eqx)
    start = time()
    df_canoak_le_batched(para, batched_met, canoak_eqx)
    end = time()
    simulation_time = end - start
    print(f"Simulation time: {simulation_time} sec")
    time_set.append(simulation_time)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(batch_size_set, time_set, "k*-")
ax.set(
    ylabel="Execution time [s]",
    xlabel="Batch size",
    title="Gradients calculation (US-Hn1)",
    yscale="log",
    xscale="log",
)
plt.savefig("Benchmark_time_batch_gradients.png", dpi=150)
