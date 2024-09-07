"""
Train a JAX-CanVeg model based on a configuration file.

Author: Peishi Jiang
Date: 2024.8.30.
"""

# TODO: Need a documentation for the configuration file
# stomata_type
# leaf angle type
# leaf relative humidity module
# soil respiration module
# See the comments of jax_canveg.subjects.parameters.Para

import os
import json

# import pickle
import logging

import time
from datetime import datetime

from math import floor
from pathlib import PosixPath
from typing import Any, Optional, List, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pandas as pd

from .physics.energy_fluxes import get_dispersion_matrix
from .subjects import initialize_parameters
from .subjects import convert_obs_to_batched_obs, convert_met_to_batched_met

# from .subjects import convert_batchedmet_to_met
from .subjects import Met, Obs, BatchedObs, get_met_forcings, get_obs
from .subjects import get_filter_para_spec
from .shared_utilities.optim import get_loss_function, get_optimzer
from .shared_utilities.optim import perform_optimization_batch
from .shared_utilities.scaler import identity_scaler
from .shared_utilities.scaler import standardizer_1d, standardizer_nd
from .shared_utilities.scaler import minmax_1d, minmax_nd
from .models import get_canveg_eqx_class, get_output_function
from .models import CanvegBase, save_model


def train_model(f_configs: PosixPath | str, save_log_local: bool = False):
    """Parse the configuration file and train the model

    Args:
        f_configs (PosixPath | str): the configuration file in JSON format
    """
    # Go to the folder where the configuration resides
    parent_directory = os.path.dirname(f_configs)
    f_configs = os.path.basename(f_configs)
    os.chdir(parent_directory)

    if save_log_local:
        ts = time.time()
        time_label = datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H:%M:%S")
        logging.basicConfig(
            filename=f"train{time_label}.log",
            # filename="train.log",
            filemode="w",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        )
    logging.info(
        f"Start training JAX-CanVeg with the configuration file {str(f_configs)} under {parent_directory}"  # noqa: E501
    )

    # Parse the configuration file
    (
        model,
        filter_model_spec,
        batched_met,
        batched_y,
        hyperparams,
        para_min,
        para_max,
        output_funcs,
        loss_func,
        optim,
        nsteps,
        configs,
    ) = parse_config(f_configs)

    batched_y_train, batched_y_test = batched_y[0], batched_y[1]
    batched_met_train, batched_met_test = batched_met[0], batched_met[1]

    # met = convert_batchedmet_to_met(batched_met)[0]
    # model(met=met)
    # return

    # Train the model
    (
        model_new,
        loss_train_set,
        loss_test_set,
    ) = perform_optimization_batch(  # pyright: ignore
        model.get_fixed_point_states,  # pyright: ignore
        filter_model_spec.get_fixed_point_states,
        optim,
        nsteps,
        loss_func,
        batched_y_train,
        batched_met_train,
        batched_y_test,
        batched_met_test,
        para_min,
        para_max,
        *output_funcs,
    )
    # canveg_eqx_new = canveg_eqx_new.args[0]
    model_new = model_new.__self__  # pyright: ignore

    # Save the loss values
    logging.info("Saving the loss values ...")
    save_configs = check_and_get_keyword(
        configs, "saving configurations", "saving", True, {}
    )
    f_loss = check_and_get_keyword(
        save_configs, "loss values", "saving", True, "./loss.csv"
    )
    loss_df = pd.DataFrame(
        jnp.array([loss_train_set, loss_test_set]).T, columns=["training", "test"]
    )
    loss_df.to_csv(f_loss, index=False)
    # loss = {"train": loss_train_set, "test": loss_test_set}
    # pickle.dump(loss, open(f_loss, "wb"), pickle.HIGHEST_PROTOCOL)

    # Save the model
    logging.info("Saving the trained model ...")
    f_model = check_and_get_keyword(
        save_configs, "new model", "saving", True, "./new_model.eqx"
    )
    save_model(f_model, hyperparams, model_new)


################### A few utility functions go here ###################
def parse_config(f_configs: PosixPath | str):
    """Parse the configuration file and return the intialized model,
       training data, and the optimizer.

    Args:
        f_configs (PosixPath | str): the configuration file in JSON format
    """
    # Read the configuration file
    with open(f_configs, "r") as f:
        configs = json.load(f)

    # Load the forcing and flux data
    data_configs = configs["data"]
    (
        met_train,
        obs_train,
        n_time_train,
        met_test,
        obs_test,
        n_time_test,
    ) = get_forcing_flux(data_configs)

    # Get the model
    model_configs = configs["model configurations"]
    model, hyperparams, para, para_min, para_max = get_model(
        model_configs, met_train, obs_train, n_time_train
    )

    # Get the learning elements
    learn_configs = configs["learning configurations"]
    (
        output_funcs,
        batched_met,
        _,
        batched_y_scaled,
        filter_model_spec,
        loss_func,
        optim,
        nsteps,
    ) = get_learning_elements(
        learn_configs,
        model,
        met_train,
        obs_train,
        n_time_train,
        met_test,
        obs_test,
        n_time_test,
    )

    return (
        model,
        filter_model_spec,
        batched_met,
        batched_y_scaled,
        hyperparams,
        para_min,
        para_max,
        output_funcs,
        loss_func,
        optim,
        nsteps,
        configs,
    )


def get_forcing_flux(data_configs: dict):
    """Get the forcing and flux data."""
    logging.info("Loading training forcings and fluxes...")
    f_training_forcing = check_and_get_keyword(
        data_configs, "training forcings", "data"
    )
    f_training_obs = check_and_get_keyword(data_configs, "training fluxes", "data")
    met_train, n_time_train = get_met_forcings(f_training_forcing)
    obs_train = get_obs(f_training_obs)

    logging.info("Loading test forcings and fluxes if any...")
    f_test_forcing = check_and_get_keyword(
        data_configs, "test forcings", "data", True, None
    )
    f_test_obs = check_and_get_keyword(data_configs, "test fluxes", "data", True, None)
    if f_test_forcing is None or f_test_obs is None:
        met_test, obs_test, n_time_test = None, None, None
    else:
        met_test, n_time_test = get_met_forcings(f_test_forcing)
        obs_test = get_obs(f_test_obs)

    return met_train, obs_train, n_time_train, met_test, obs_test, n_time_test


def get_model(model_configs: dict, met: Met, obs: Obs, n_time: int):
    """Get the canveg equinox model."""
    time_zone = check_and_get_keyword(model_configs, "time zone", "model")
    latitude = check_and_get_keyword(model_configs, "latitude", "model")
    longitude = check_and_get_keyword(model_configs, "longitude", "model")
    n_can_layers = check_and_get_keyword(
        model_configs, "number of canopy layers", "model", True, 10
    )
    n_atmos_layers = check_and_get_keyword(
        model_configs, "number of atmospheric layers", "model", True, 10
    )
    veg_ht = check_and_get_keyword(model_configs, "canopy height", "model")
    meas_ht = check_and_get_keyword(model_configs, "measurement height", "model")
    soil_depth = check_and_get_keyword(model_configs, "soil depth", "model")
    n_hr_per_day = check_and_get_keyword(
        model_configs, "number of observed steps per day", "model"
    )
    niter = check_and_get_keyword(model_configs, "number of solver iterations", "model")
    npart = check_and_get_keyword(
        model_configs, "number of lagrangian particles", "model", True, int(1e6)
    )

    # TODO: more descriptions are needed for the stomatal type and leaf angle type
    stomata = check_and_get_keyword(model_configs, "stomata type", "model", True, 1)
    leafangle = check_and_get_keyword(
        model_configs, "leaf angle type", "model", True, 1
    )
    leafrh = check_and_get_keyword(
        model_configs, "leaf relative humidity module", "model", True, 0
    )
    soilresp = check_and_get_keyword(
        model_configs, "soil respiration module", "model", True, 0
    )

    # Get the model setup and parameters
    logging.info("Loading the model setup and parameters ...")
    setup, para, para_min, para_max = initialize_parameters(  # pyright: ignore
        time_zone=time_zone,
        latitude=latitude,
        longitude=longitude,
        stomata=stomata,
        leafrh=leafrh,
        soilresp=soilresp,
        veg_ht=veg_ht,
        leafangle=leafangle,
        n_can_layers=n_can_layers,
        n_atmos_layers=n_atmos_layers,
        meas_ht=meas_ht,
        soil_depth=soil_depth,
        n_hr_per_day=n_hr_per_day,
        n_time=n_time,
        npart=npart,
        obs=obs,
        met=met,
        niter=niter,
        get_para_bounds=True,
    )
    hyperparams = dict(
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
        npart=npart,
        niter=niter,
    )

    # Get the dispersion matrix
    f_dij = check_and_get_keyword(model_configs, "dispersion matrix", "data", True)
    if f_dij is not None:
        logging.info("Loading the disperion matrix ...")
        dij = get_dispersion_matrix(setup, para, f_dij)
    else:
        logging.info("Generating the disperion matrix ...")
        dij = get_dispersion_matrix(setup, para)

    # Get the model
    model_type = check_and_get_keyword(
        model_configs, "model type", "model", True, "pure physics"
    )
    canveg_eqx_class = get_canveg_eqx_class(model_type)  # pyright: ignore
    canveg_eqx = canveg_eqx_class(para, setup, dij)  # pyright: ignore

    return canveg_eqx, hyperparams, para, para_min, para_max


def get_learning_elements(
    learn_configs: dict,
    model: CanvegBase,
    met_train: Met,
    obs_train: Obs,
    n_time_train: int,
    met_test: Optional[Met] = None,
    obs_test: Optional[Obs] = None,
    n_time_test: Optional[int] = None,
):
    """Get the essential training elements."""
    # Get the output function arguments
    output_func_args = check_and_get_keyword(
        learn_configs, "output function", "learning", True, "LE"
    )
    update_func, get_func = get_output_function(output_func_args)

    # Get the output scaler
    output_scaler_args = check_and_get_keyword(
        learn_configs, "output scaler", "learning", True, None
    )
    output_scaler = get_output_scaler(output_scaler_args, output_func_args, obs_train)
    output_funcs = [update_func, get_func, output_scaler]

    # Create the batched training data
    logging.info("Converting the obs and met to batched dataset ...")
    batch_size = check_and_get_keyword(learn_configs, "batch size", "learning")
    batched_met_train, batched_y_train, batched_y_train_scaled = get_batched_met_obs(
        met_train, obs_train, batch_size, n_time_train, output_func_args, output_scaler
    )
    if obs_test is not None:
        batched_met_test, batched_y_test, batched_y_test_scaled = get_batched_met_obs(
            met_test,  # pyright: ignore
            obs_test,
            1,  # pyright: ignore
            n_time_test,  # pyright: ignore
            output_func_args,
            output_scaler,
        )
    else:
        batched_met_test, batched_y_test, batched_y_test_scaled = None, None, None
    batched_met = [batched_met_train, batched_met_test]
    batched_y = [batched_y_train, batched_y_test]
    batched_y_scaled = [batched_y_train_scaled, batched_y_test_scaled]

    # Filter the parameters to be estimated
    logging.info("Getting the filtered model spec for the tunable parameters ...")
    tunable_para = check_and_get_keyword(
        learn_configs, "tunable parameters", "learning", True, None
    )
    filter_model_spec = get_filter_model_spec(model, tunable_para)

    # Get the loss function
    logging.info("Getting the loss function ...")
    loss_func_arg = check_and_get_keyword(
        learn_configs, "loss function", "learning", True, "mse"
    )
    loss_func = get_loss_function(loss_func_arg)

    # Get the optimizer and training epochs
    logging.info("Getting the optimizer and training epochs ...")
    nsteps = check_and_get_keyword(learn_configs, "number of epochs", "learning")
    optim_configs = check_and_get_keyword(
        learn_configs, "optimizer", "learning", True, None
    )
    optim = get_optimzer(optim_configs)

    return (
        output_funcs,
        batched_met,
        batched_y,
        batched_y_scaled,
        filter_model_spec,
        loss_func,
        optim,
        nsteps,
    )


def get_batched_met_obs(
    met: Met,
    obs: Obs,
    batch_size: int | None,
    n_time: int,
    output_args: str,
    output_scaler: Callable,
):
    """Get the batched observation data for training"""
    if batch_size is None:
        batch_size = n_time
    n_batch = floor(n_time / batch_size)
    batched_met = convert_met_to_batched_met(met, n_batch, batch_size)
    batched_obs = convert_obs_to_batched_obs(obs, n_batch, batch_size)
    batched_y = downselect_obs(batched_obs, output_args)
    batched_y_scaled = output_scaler(batched_y)
    return batched_met, batched_y, batched_y_scaled


def downselect_obs(batched_obs: BatchedObs, output_args: str):
    """Select the particular observation output for training"""
    if output_args.lower() == "canle" or output_args.lower() == "canopy le":
        return batched_obs.LE
    elif output_args.lower() == "cangpp" or output_args.lower() == "canopy gpp":
        return batched_obs.GPP
    elif output_args.lower() == "cannee" or output_args.lower() == "canopy nee":
        return batched_obs.Fco2
    elif output_args.lower() == "canlenee" or output_args.lower() == "canopy le nee":
        le, nee = batched_obs.LE, batched_obs.Fco2
        # The returned shape is [nbatch, batch_size, 2]
        return jnp.stack([le, nee], axis=-1)
    else:
        raise Exception("Unknown output arguments: %s", output_args)


def get_output_scaler(output_scaler: str | None, output_args: str, obs: Obs):
    """Get the output scaler function"""
    # Get the observation data for scaling
    if output_args.lower() == "canle" or output_args.lower() == "canopy le":
        data, dim = obs.LE, 1
    elif output_args.lower() == "cangpp" or output_args.lower() == "canopy gpp":
        data, dim = obs.GPP, 1
    elif output_args.lower() == "cannee" or output_args.lower() == "canopy nee":
        data, dim = obs.Fco2, 1
    elif output_args.lower() == "canlenee" or output_args.lower() == "canopy le nee":
        le, nee = obs.LE, obs.Fco2
        data = jnp.array([le, nee]).T  # shape (Nt, 2)
        dim = 2
    else:
        raise Exception("Unknown output arguments: %s", output_args)

    # Get the scaler
    if output_scaler is None:
        scaler = identity_scaler
    elif output_scaler.lower() == "standard":
        if dim == 1:
            mu, std = data.mean(), data.std()
            scaler = lambda x: standardizer_1d(x, mu, std)  # noqa: E731
        else:
            mu, std = data.mean(axis=0), data.std(axis=0)
            scaler = lambda x: standardizer_nd(x, mu, std)  # noqa: E731
    elif output_scaler.lower() == "minmax":
        if dim == 1:
            xmin, xmax = data.min(), data.max()
            scaler = lambda x: minmax_1d(x, xmin, xmax)  # noqa: E731
        else:
            xmin, xmax = data.min(axis=0), data.max(axis=0)
            scaler = lambda x: minmax_nd(x, xmin, xmax)  # noqa: E731
    else:
        raise Exception("Unknown output scaler: %s", output_scaler)

    return scaler


def get_filter_model_spec(model: CanvegBase, tunable_para: Optional[List] = None):
    """Get the filtered model specification based on the tunable parameter list."""
    filter_model_spec = jtu.tree_map(lambda _: False, model)
    filter_para = get_filter_para_spec(model.para, tunable_para)  # pyright: ignore
    filter_model_spec = eqx.tree_at(
        lambda t: t.para, filter_model_spec, replace=filter_para
    )
    return filter_model_spec


def check_and_get_keyword(
    configs: dict,
    key: str,
    config_type: str = "Unknown",
    return_default: bool = False,
    default: Any = None,
):
    if key in configs:
        return configs[key]
    else:
        if return_default:
            logging.info(
                f"{key} is not found in configuration of {config_type} and return {default}."  # noqa: E501
            )
            return default
        else:
            raise Exception(f"{key} is not found in configuration of {config_type}.")
