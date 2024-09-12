import json
import equinox as eqx
import jax.numpy as jnp

from typing import Dict, Any
from .canveg_eqx import CanvegBase

import logging
from ..subjects.meterology import Met
from ..subjects.states import Obs
from ..subjects.initialization_update import initialize_parameters

# Function for saving a model
def save_model(filename: str, hyperparams: Dict, model: CanvegBase) -> None:
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


# Function for loading a model
def load_model(filename: str, modelclass) -> CanvegBase:
    def initialize_empty_met_obs():
        met_zeros = [jnp.zeros(5)] * 15
        obs_zeros = [jnp.zeros(5)] * 9
        return Met(*met_zeros), Obs(*obs_zeros)

    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        met, obs = initialize_empty_met_obs()
        setup, para = initialize_parameters(  # pyright: ignore
            met=met, obs=obs, **hyperparams
        )
        dij = jnp.zeros([setup.n_total_layers, setup.n_can_layers])
        model_skeleton = modelclass(para, setup, dij)
        model = eqx.tree_deserialise_leaves(f, model_skeleton)

    return model


# Function for loading a model by checking its configuration file
# TODO: This function should not be used ...
def load_model_check(filename: str, modelclass, model_configs: Dict) -> CanvegBase:
    def initialize_empty_met_obs():
        met_zeros = [jnp.zeros(5)] * 15
        obs_zeros = [jnp.zeros(5)] * 9
        return Met(*met_zeros), Obs(*obs_zeros)

    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        if "leafrh" not in hyperparams:
            leafrh = check_and_get_keyword(
                model_configs, "leaf relative humidity module", "model", True, 0
            )
            hyperparams["leafrh"] = leafrh

        if "soilresp" not in hyperparams:
            soilresp = check_and_get_keyword(
                model_configs, "soil respiration module", "model", True, 0
            )
            hyperparams["soilresp"] = soilresp

        if "stomata" not in hyperparams:
            stomata = check_and_get_keyword(
                model_configs, "stomata type", "model", True, 1
            )
            hyperparams["stomata"] = stomata

        if "leafangle" not in hyperparams:
            leafangle = check_and_get_keyword(
                model_configs, "leaf angle type", "model", True, 1
            )
            hyperparams["leafangle"] = leafangle

        met, obs = initialize_empty_met_obs()
        setup, para = initialize_parameters(  # pyright: ignore
            met=met, obs=obs, **hyperparams
        )
        dij = jnp.zeros([setup.n_total_layers, setup.n_can_layers])
        model_skeleton = modelclass(para, setup, dij)
        model = eqx.tree_deserialise_leaves(f, model_skeleton)

    return model


# TODO: I should remove this function
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
