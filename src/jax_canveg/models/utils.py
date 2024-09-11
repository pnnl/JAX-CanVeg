import json
import equinox as eqx
import jax.numpy as jnp

from typing import Dict
from .canveg_eqx import CanvegBase

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
