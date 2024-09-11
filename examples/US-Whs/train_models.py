"""Train the models."""

import os
import json
import logging

import time
from datetime import datetime

import jax
from pathlib import Path
from jax_canveg import train_model
from jax_canveg.shared_utilities import tune_jax_naninfs_for_debug

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")
tune_jax_naninfs_for_debug(False)

# Start logging information
ts = time.time()
time_label = datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H:%M:%S")
logging.basicConfig(
    filename=f"train-Whs-{time_label}.log",
    filemode="w",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
)

# Current directory
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))

################################################################
# General configuration
################################################################
f_configs_template = dir_mother / "./test-model/configs.json"
model_configs = {
    "time zone": -7,
    "latitude": 31.7438,
    "longitude": -110.0522,
    "stomata type": 1,
    "leaf angle type": 4,
    "canopy height": 1.0,
    "measurement height": 6.5,
    "soil respiration module": 1,
}
learning_config = {
    "batch size": 1024,
    "number of epochs": 300,
    # "number of epochs": 2,
    "output scaler": "standard",
}
data_config = {
    "training forcings": "../../../data/fluxtower/US-Whs/US-Whs-forcings.csv",
    "training fluxes": "../../../data/fluxtower/US-Whs/US-Whs-fluxes.csv",
    "test forcings": "../../../data/fluxtower/US-Whs/US-Whs-forcings-test.csv",
    "test fluxes": "../../../data/fluxtower/US-Whs/US-Whs-fluxes-test.csv",
}

################################################################
#  Configurations for
#  - canopy layers
#  - hybrid model
#  - multiobjective optimization
################################################################
canopy_layers = ["1L", "ML"]
model_types = ["PB", "Hybrid"]
multi_optim_le_weight = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# multi_optim_le_weight = [0.0, 0.5, 1.0]
canopy_layers_config = {
    "1L": {
        "number of canopy layers": 1,
        "dispersion matrix": "../../../data/dij/Dij_US-Whs_1L.csv",
    },
    "ML": {
        "number of canopy layers": 50,
        "dispersion matrix": "../../../data/dij/Dij_US-Whs_50L.csv",
    },
}
model_types_config = {
    "PB": {"leaf relative humidity module": 0},
    "Hybrid": {"leaf relative humidity module": 1},
}


################################################################
#  Load the default configuration
################################################################
with open(f_configs_template, "r") as f:
    configs = json.load(f)


################################################################
#  Train the models
################################################################
for cl in canopy_layers:
    cl_config = canopy_layers_config[cl]

    for mt in model_types:
        mt_config = model_types_config[mt]

        for mow in multi_optim_le_weight:
            # Step 0: Stay in the current directory
            os.chdir(dir_mother)

            # Step 1: Case folder name
            dir_name = dir_mother / f"{mt}-{cl}-{mow}"
            f_configs = dir_name / "configs.json"
            logging.info("")
            logging.info(f"The model: {f_configs}.")

            # Step 2-a: Create the folder if not existed
            if not dir_name.is_dir():
                dir_name.mkdir()
            # Step 2-b: Continue to the next loop if the folder and results exist
            else:
                files = dir_name.glob("**/*")
                check_config, check_model = False, False
                for f in files:
                    if f == f_configs:
                        check_config = True
                    if f.suffix == ".eqx":
                        check_model = True
                if check_model and check_config:
                    logging.info(
                        f"The model has been trained in {f_configs}. Continue to the next model."  # noqa: E501
                    )
                    continue

            # Step 3: Create the configuration file
            cfg = configs.copy()
            for key, value in model_configs.items():
                cfg["model configurations"][key] = value
            for key, value in learning_config.items():
                cfg["learning configurations"][key] = value
            for key, value in data_config.items():
                cfg["data"][key] = value
            for key, value in cl_config.items():
                cfg["model configurations"][key] = value
            for key, value in mt_config.items():
                cfg["model configurations"][key] = value
            cfg["learning configurations"]["loss function"]["weights"] = [mow, 1 - mow]

            # Step 4: Save it to the designated folder
            with open(f_configs, "w") as f:
                json.dump(cfg, f, indent=4)

            # Step 5: Launch training!
            logging.info("Start training!")
            train_model(f_configs)
