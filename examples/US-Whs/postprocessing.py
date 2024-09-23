"""Postprocess the trained models."""

import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_canveg import load_model
from jax_canveg.shared_utilities import compute_metrics, get_time

from tqdm import tqdm  # pyright: ignore

jax.config.update("jax_enable_x64", True)

# Current directory
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))

################################################################
# JAX-CanVeg
################################################################
canopy_layers = ["1L", "ML"]
model_types = ["PB", "Hybrid"]
multi_optim_le_weight = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
combinations = list(
    itertools.product(canopy_layers, model_types, multi_optim_le_weight)
)

for cl, mt, mow in tqdm(combinations):
    # Step 0: Stay in the current directory
    os.chdir(dir_mother)

    # Step 1: Case folder name
    dir_case = dir_mother / f"{mt}-{cl}-{mow}"
    f_configs = dir_case / "configs.json"
    if not f_configs.is_file():
        print(f"The case does not exist: {dir_case}")
        continue
    else:
        print(f"Processing the case: {dir_case} ...")
    
    # if f"{mt}-{cl}-{mow}" != 'PB-1L-0.3':
    #     continue

    # Step 2: Load the model, forcings, and observations
    model, met_train, met_test, obs_train, obs_test = load_model(f_configs)
    timesteps_train, timesteps_test = get_time(met_train), get_time(met_test)

    # Step 3: Run the model on both training and test datasets
    states_train, drivers_train = model(met_train)
    states_test, drivers_test = model(met_test)
    can_train, can_test = states_train[-1], states_test[-1]
    veg_train, veg_test = states_train[-2], states_test[-2]
    soil_train, soil_test = states_train[-3], states_test[-3]

    # Step 3-a: Remove some large numbers due to the numerical instability
    # def convert_instability_to_nan(d):
    #     d.at[d>10000.].set(jnp.nan)
    #     d.at[d<-10000.].set(jnp.nan)
    #     return d
    @jnp.vectorize
    def convert_instability_to_nan(d_e):
        return jax.lax.cond(
            jnp.abs(d_e) > 10000,
            lambda: jnp.nan,
            lambda: d_e,
        )
        # d.at[d>10000.].set(jnp.nan)
        # d.at[d<-10000.].set(jnp.nan)
        # return d
    can_train = jtu.tree_map(convert_instability_to_nan, can_train)
    can_test = jtu.tree_map(convert_instability_to_nan, can_test)
    veg_train = jtu.tree_map(convert_instability_to_nan, veg_train)
    veg_test = jtu.tree_map(convert_instability_to_nan, veg_test)
    soil_train = jtu.tree_map(convert_instability_to_nan, soil_train)
    soil_test = jtu.tree_map(convert_instability_to_nan, soil_test)

    # Step 4: Assemble key simulations
    sim_train = np.array(
        [
            met_train.soilmoisture,
            obs_train.LE,
            obs_train.Fco2,
            obs_train.H,
            obs_train.rnet,
            obs_train.gsoil,
            can_train.LE,
            can_train.NEE,
            can_train.H,
            can_train.rnet,
            can_train.gsoil,
            veg_train.gs,
            veg_train.Tsfc,
            veg_train.Ps,
            veg_train.GPP,
            soil_train.resp,
        ]
    ).T
    sim_test = np.array(
        [
            met_test.soilmoisture,
            obs_test.LE,
            obs_test.Fco2,
            obs_test.H,
            obs_test.rnet,
            obs_test.gsoil,
            can_test.LE,
            can_test.NEE,
            can_test.H,
            can_test.rnet,
            can_test.gsoil,
            veg_test.gs,
            veg_test.Tsfc,
            veg_test.Ps,
            veg_test.GPP,
            soil_test.resp,
        ]
    ).T
    sim_train_df = pd.DataFrame(
        sim_train,
        index=timesteps_train,
        columns=[
            "SWC-obs",
            "LE-obs",
            "NEE-obs",
            "H-obs",
            "Rn-obs",
            "G-obs",
            "LE",
            "NEE",
            "H",
            "Rn",
            "G",
            "gs",
            "Tsfc",
            "Ps",
            "GPP",
            "Rsoil",
        ],
    )
    sim_test_df = pd.DataFrame(
        sim_test,
        index=timesteps_test,
        columns=[
            "SWC-obs",
            "LE-obs",
            "NEE-obs",
            "H-obs",
            "Rn-obs",
            "G-obs",
            "LE",
            "NEE",
            "H",
            "Rn",
            "G",
            "gs",
            "Tsfc",
            "Ps",
            "GPP",
            "Rsoil",
        ],
    )

    # Step 5: Calculate the metrics of LE and NEE
    value_pairs = [
        [can_train.LE, obs_train.LE],
        [can_test.LE, obs_test.LE],
        [can_train.NEE, obs_train.Fco2],
        [can_test.NEE, obs_test.Fco2],
        [can_train.H, obs_train.H],
        [can_test.H, obs_test.H],
        [can_train.rnet, obs_train.rnet],
        [can_test.rnet, obs_test.rnet],
        [can_train.gsoil, obs_train.gsoil],
        [can_test.gsoil, obs_test.gsoil],
    ]
    metric_values = []
    for pred, true in value_pairs:
        # print(pred.mean(), true.mean())
        metrics = compute_metrics(pred, true, mask_naninf=True)
        metric_values.append(list(metrics.values()))
        metric_keys = list(metrics.keys())
    metric_values = np.array(metric_values)
    metric_df = pd.DataFrame(metric_values, columns=metric_keys)  # pyright: ignore
    metric_df.index = [
        "LE-train", "LE-test", "NEE-train", "NEE-test", "H-train", "H-test",
        "Rn-train", "Rn-test", "G-train", "G-test"
    ]

    # Step 6: Save the metrics and simulations
    f_metrics = dir_case / "metrics.csv"
    metric_df.to_csv(f_metrics)
    f_sim_train = dir_case / "predictions_train.csv"
    sim_train_df.to_csv(f_sim_train)
    f_sim_test = dir_case / "predictions_test.csv"
    sim_test_df.to_csv(f_sim_test)


################################################################
# Pure deep learning model
################################################################
w_set = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# w_set = [0., 0.5, 1.0]
for w in tqdm(w_set):
    # Step 0: Stay in the current directory
    os.chdir(dir_mother)

    # Step 1: Case folder name
    dir_case = dir_mother / f"DNN_LE-GPP-{w}"
    if not dir_case.is_dir():
        print(f"The case does not exist: {dir_case}")
        continue
    else:
        print(f"Processing the case: {dir_case} ...")

    # Step 2: Load the predictions
    f_sim_train = dir_case / "predictions_train.txt"
    f_sim_test = dir_case / "predictions_test.txt"
    sim_train, sim_test = np.loadtxt(f_sim_train), np.loadtxt(f_sim_test)
    le_train, nee_train = sim_train[:, 0], sim_train[:, 1]
    le_test, nee_test = sim_test[:, 0], sim_test[:, 1]
    # print(le_train.shape, le_test.shape, nee_train.shape)
    # print(obs_train.LE.shape)

    # Step 3: Compute the metrics
    value_pairs = [
        [le_train, obs_train.LE],  # pyright: ignore
        [le_test, obs_test.LE],  # pyright: ignore
        [nee_train, obs_train.Fco2],  # pyright: ignore
        [nee_test, obs_test.Fco2],  # pyright: ignore
    ]
    metric_values = []
    for pred, true in value_pairs:
        metrics = compute_metrics(pred, true, mask_naninf=True)
        metric_values.append(list(metrics.values()))
        metric_keys = list(metrics.keys())
    metric_values = np.array(metric_values)
    metric_df = pd.DataFrame(metric_values, columns=metric_keys)  # pyright: ignore
    metric_df.index = ["LE-train", "LE-test", "NEE-train", "NEE-test"]

    # Step 4: Save the metrics
    f_metrics = dir_case / "metrics.csv"
    metric_df.to_csv(f_metrics)
