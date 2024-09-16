"""Performing sensitivity analysis of JAX-CanVeg at the four sites."""

import os
import itertools
from pathlib import Path
from math import floor

from tqdm import tqdm

import numpy as np
import pandas as pd

from jax_canveg import load_model
from jax_canveg.subjects import convert_met_to_batched_met
from jax_canveg.models import get_canle, update_canle, get_cannee, update_cannee
from jax_canveg.shared_utilities import sensitivity as st
from jax_canveg.shared_utilities.plot import get_time

import jax

jax.config.update("jax_debug_nans", False)
jax.config.update("jax_traceback_filtering", "off")


# Current directory
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))

################################################################
# Some parameters/configurations
################################################################
# sites = ['US-Bi1', 'US-Hn1', 'US-Me2', 'US-Whs']
# cases = ['test-model', 'PB-1L-0.5']
sites = ["US-Bi1"]
cases = ["test-model"]
# n_time = None
batch_size = 1

# Parameters to be estimated
subpara = [
    "bprime",
    "ep",
    "lleaf",
    "qalpha",
    "kball",
    "leaf_clumping_factor",
    "vcopt",
    "jmopt",
    "rd25",
    "toptvc",
    "toptjm",
    "epsoil",
    "par_reflect",
    "par_trans",
    "par_soil_refl",
    "nir_reflect",
    "nir_trans",
    "nir_soil_refl",
    "q10a",
    "q10b",
    "q10c",
]

# The functions for updating and getting substates for derivative evaluation
update_get_funcs = [[update_canle, get_canle], [update_cannee, get_cannee]]
update_get_labels = ["LE", "NEE"]


################################################################
# Loop through sites on the test datasets
################################################################
# for site in tqdm(sites):
for site, case in tqdm(itertools.product(sites, cases)):
    # Stay in the current directory
    os.chdir(dir_mother)

    f_configs = Path(f"{site}/{case}/configs.json")
    print(f_configs)

    f_sensitivities = []
    cond = True
    for i, label in enumerate(update_get_labels):
        f_sens = dir_mother / f"{site}/{case}/sensitivity-{label}.csv"
        f_sensitivities.append(f_sens)
        cond &= f_sens.is_file()
    # if cond:
    #     continue

    # Load the model, forcings, and observations
    model, met_train, _, _, _ = load_model(f_configs)
    timesteps = get_time(met_train)
    n_time = timesteps.size
    # n_time = 3
    # timesteps = timesteps[:n_time]
    n_batch = floor(n_time / batch_size)
    batched_met = convert_met_to_batched_met(met_train, n_batch, batch_size)
    para = model.para

    # Perform sensitivity analysis
    gradients_set = []
    for funcs in update_get_funcs:
        update_func, get_func = funcs[0], funcs[1]
        gradients = st.df_canveg_le_batched(
            para, batched_met, model, update_func, get_func
        )
        gradients_set.append(gradients)

    # Calculate directional derivative using Implicit Function Theorem-based AD
    para_tangents = st.get_partial_directional_vector(subpara, para)
    dir_grad_ad_set = []
    for funcs in update_get_funcs:
        update_func, get_func = funcs[0], funcs[1]
        dir_grad_ad = st.AD_directional_derivative_batched(
            para, batched_met, model, para_tangents, update_func, get_func
        )
        dir_grad_ad_set.append(dir_grad_ad)

    # Calculate directional derivative using Finite Difference-based differentiation
    dir_grad_fd_set = []
    for funcs in update_get_funcs:
        update_func, get_func = funcs[0], funcs[1]
        dir_grad_fd = st.FD_directional_derivative_batched(
            para, batched_met, model, para_tangents, update_func, get_func, h=0.015
        )
        dir_grad_fd_set.append(dir_grad_fd)

    # Save the results
    keys = subpara + ["IFT-AD", "FD"]
    for i, f_sens in enumerate(f_sensitivities):
        grad = gradients_set[i]
        dir_grad_ad = dir_grad_ad_set[i]
        dir_grad_fd = dir_grad_fd_set[i]

        sensitivity = []
        for p in subpara:
            s = getattr(grad, p)
            sensitivity.append(s)
        sensitivity.append(dir_grad_ad)
        sensitivity.append(dir_grad_fd)
        sensitivity = np.array(sensitivity).T
        sens_df = pd.DataFrame(sensitivity, index=timesteps, columns=keys)
        # sens_df.to_csv(dir_mother / f"{site}/{case}/sensitivity-{label}.csv")
        sens_df.to_csv(f_sens)
