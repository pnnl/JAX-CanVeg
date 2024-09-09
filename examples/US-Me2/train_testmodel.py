"""Train the models."""

import os
import jax
from pathlib import Path
from jax_canveg import train_model
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

################################################################
# Run the model
################################################################
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))
f_config = dir_mother / "./test-model/configs.json"
train_model(f_config, save_log_local=True)  # pyright: ignore