"""Run one model."""

import os
import jax
from pathlib import Path
from jax_canveg import train_model, load_model
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

################################################################
# Run the model
################################################################
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))
f_configs = dir_mother / "./PB-1L-0.5/configs.json"

# Load the model, forcings, and observations
model, met_train, met_test, obs_train, obs_test = load_model(f_configs)
# timesteps_train, timesteps_test = get_time(met_train), get_time(met_test)

# Run the model on both training and test datasets
states_train, drivers_train = model(met_train)
# states_test, drivers_test = model(met_test)
os.chdir(dir_mother)
