"""Train the models."""

# import os
import jax
from pathlib import Path
from jax_canveg import train_model
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(True)

################################################################
# test model
################################################################
# f_config = Path("./test-model/configs.json")
f_config = Path("./PB-ML-0.0/configs.json")
train_model(f_config)
