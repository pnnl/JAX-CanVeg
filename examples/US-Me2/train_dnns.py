"""Train DNNs."""

import os
from pathlib import Path

import numpy as np

import logging
import time
from datetime import datetime

# import jax
# import equinox as eqx
import jax.numpy as jnp
import optax
from jax_canveg.subjects import get_met_forcings, get_obs
from jax_canveg.shared_utilities.plot import get_time
from jax_canveg.shared_utilities.dnn import MLP, train_dnn
from jax_canveg.shared_utilities.optim import mse, weighted_loss


##################################################
# General configurations
##################################################
# Current directory
dir_mother = Path(os.path.dirname(os.path.realpath(__file__)))

# Files and directories
site, key = "US-Me2", "dl"
f_forcing_train = f"../../data/fluxtower/{site}/{site}-forcings.csv"
f_obs_train = f"../../data/fluxtower/{site}/{site}-fluxes.csv"
f_forcing_test = f"../../data/fluxtower/{site}/{site}-forcings-test.csv"
f_obs_test = f"../../data/fluxtower/{site}/{site}-fluxes-test.csv"

# Input variables
in_varns = ["T_air", "rglobal", "eair", "wind", "CO2",
            "P_kPa", "ustar", "soilmoisture", "lai"]

# DNN hyperparameters
batch_size = 64
# batch_size = 1024
initial_lr = 2e-1
nsteps = 300
# nsteps = 10
seed = 5678
scaler_type = 'standard'
model_type = MLP
model_args = {
    "depth": 2, "width_size": 6, "model_seed": seed,
    "out_size": 2, "hidden_activation": "tanh", 
    "final_activation": "identity", "in_size": len(in_varns)
}

# Start logging information
ts = time.time()
time_label = datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H:%M:%S")
logging.basicConfig(
    filename=f"train-{site}-{time_label}.log",
    filemode="w",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
)


##################################################
# Read data and scaler
##################################################
# Get inputs from forcings
met_train, n_time_train = get_met_forcings(f_forcing_train)
timesteps_train = get_time(met_train)
x_train = np.array([getattr(met_train, varn) for varn in in_varns]).T
met_test, n_time_test = get_met_forcings(f_forcing_test)
timesteps_test = get_time(met_test)
x_test = np.array([getattr(met_test, varn) for varn in in_varns]).T

# Get the observations
obs_train, obs_test = get_obs(f_obs_train), get_obs(f_obs_test)

# Get the observed outputs
# y_train, y_test = obs_train.LE, obs_test.LE
y_train = np.array([obs_train.LE, obs_train.Fco2]).T
y_test = np.array([obs_test.LE, obs_test.Fco2]).T


##################################################
# Initialize optimizer and scheduler
##################################################
scheduler = optax.piecewise_constant_schedule(
    initial_lr, boundaries_and_scales={50: 0.1, 100: 0.1, 200: 0.1}
)
# scheduler = optax.constant_schedule(initial_lr)
optim = optax.adam(learning_rate=scheduler)  # Adam optimizer


##################################################
# Train DNNs
##################################################
w_set = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# w_set = [0., 0.5, 1.0]
for w in w_set:
    # Get the saving dir
    dir_save = dir_mother / f"DNN_LE-GPP-{w}"

    # Define the weighted normalized function
    weights = jnp.array([w, 1-w])
    def loss_func(y, pred_y):
        return weighted_loss(y, pred_y, mse, weights)

    # Train the model
    train_dnn(
        dir_save, model_type, model_args,
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        batch_size=batch_size, nsteps=nsteps, scaler_type=scaler_type, 
        optim=optim, loss_func=loss_func, save_log_local=False
    )


# ################################################
# # Train a DNN for LE only
# ##################################################
# # Get the observed outputs
# y_train, y_test = obs_train.LE, obs_test.LE

# # Get the saving dir
# dir_save = dir_mother / "DNN_LE"

# # Train the models
# model_args = mlp_configs.copy()
# model_args['in_size'] = len(in_varns)
# model_args['out_size'] = 2
# train_dnn(
#     dir_save, model_type, model_args,
#     x_train = x_train, y_train=y_train, x_test=x_test, y_test=y_test,
#     batch_size=batch_size, nsteps=nsteps, scaler=scaler, optim=optim,
#     loss_func=mse, save_log_local=True
# )


##################################################
# Train a DNN for NEE only
##################################################

##################################################
# Train a DNN for both LE and NEE
##################################################
