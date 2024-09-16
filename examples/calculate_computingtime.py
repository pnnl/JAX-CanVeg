"""Log the computation time of JAX-CanVeg at the four sites."""

import os
import sys
import timeit
import logging
from pathlib import Path

import jax

################################################################
# Take argument from user input to indicate whether jax is set up
# at GPU or CPU
################################################################
device = sys.argv[1]
device = device.lower()
if device not in ["cpu", "cuda"]:
    print(f"Unknown device: {device}. Set it to cpu.")
    device = "cpu"

logging.basicConfig(
    filename=f"./computation_time-{device}.log",
    filemode="w",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
)

jax.config.update("jax_platform_name", device)

# Current directory
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))

################################################################
# Loop through sites on the test datasets
################################################################
from jax_canveg import load_model  # noqa: E402

# sites = ['US-Bi1', 'US-Hn1', 'US-Me2', 'US-Whs']
sites = ["US-Bi1", "US-Hn1", "US-Me2", "US-Whs"]

number_of_trials = 10
for site in sites:
    # Stay in the current directory
    os.chdir(dir_mother)

    logging.info(
        f"Calculating the computing time of JAX-CanVeg at {site} on {device}..."
    )
    f_configs = Path(f"{site}/test-model/configs.json")

    # Load the model, forcings, and observations
    model, met_train, _, _, _ = load_model(f_configs)

    # Pre-compile the function before timing...
    model(met_train)

    # Calculating the computing time at the training dataset
    # time = timeit.timeit('model(met_train)', number=number_of_trials)
    time = timeit.timeit(lambda: model(met_train), number=number_of_trials)
    time = time / number_of_trials
    logging.info(f"The average time of {number_of_trials} trials is {time} seconds ...")
    logging.info("")
