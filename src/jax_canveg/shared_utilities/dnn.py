"""
Training a DNN model. Since training a pure DNN is almost irrelevant with
training a canveg model, I store the codes here but need to figure a better
way to document/write the codes.

Author: Peishi Jiang
Date: 2024.09.10
"""

# TODO: This piece of code needs to be integrated with ..subjects.dnn.

import os
import json
import time
import logging
from datetime import datetime
from pathlib import PosixPath

import numpy as np
import pandas as pd
from sklearn import preprocessing as pp

import torch
from torch.utils.data import Dataset

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

from typing import Callable, Dict
from jaxtyping import Array


class MLP(eqx.nn.MLP):

    def __init__(
        self,
        in_size: int=2,
        out_size: int=1,
        width_size: int=3,
        depth: int=3,
        model_seed: int=1024,
        hidden_activation: str='tanh',
        final_activation: str='tanh',
        **kwargs
    ):
        key = jrandom.key(model_seed)
        # Get the activation functions
        hidden_activation = get_activation(hidden_activation)
        final_activation = get_activation(final_activation)

        super().__init__(
            in_size=in_size, out_size=out_size, width_size=width_size,
            depth=depth, activation=hidden_activation, final_activation=final_activation,
            key=key,
            **kwargs
        )


def train_dnn(
    dir_save: PosixPath, 
    model_type: eqx.Module, 
    model_args: Dict,
    x_train: Array, 
    y_train: Array, 
    x_test: Array, 
    y_test: Array,
    batch_size: int,
    nsteps: int,
    scaler_type: str, 
    optim: optax._src.base.GradientTransformation, 
    loss_func: Callable,
    save_log_local: bool=False
):
    # Go to the folder where the results should be saved
    if not dir_save.is_dir():
        dir_save.mkdir()
    os.chdir(dir_save)

    if save_log_local:
        ts = time.time()
        time_label = datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H:%M:%S")
        logging.basicConfig(
            filename=f"train{time_label}.log",
            filemode="w",
            datefmt="%H:%M:%S",
            level=logging.INFO,
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        )
    logging.info(f"Start training a DNN model ...")

    @eqx.filter_jit
    def make_step(model, opt_state, x_batch, y_batch):
        loss_value, grads = eqx.filter_value_and_grad(loss_func_batch)(
            model, x_batch, y_batch, loss_func)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    # Loss function
    @eqx.filter_jit
    def loss_func_batch(model, x_batch, y_batch, loss_func):
        pred_y_batch = jax.vmap(model)(x_batch)
        return loss_func(y_batch, pred_y_batch)
    
    model = model_type(**model_args)

    # Establish the scaler for x and y
    # Get scaler
    if scaler_type.lower() == 'standard':
        scaler_x = pp.StandardScaler().fit(x_train)
        scaler_y = pp.StandardScaler().fit(y_train)
    elif scaler_type.lower() == 'minmax':
        scaler_x = pp.MinMaxScaler().fit(x_train)
        scaler_y = pp.MinMaxScaler().fit(y_train)
    else:
        raise Exception(f"Unknown scaler type: {scaler_type}.")
    x_train_norm = scaler_x.transform(x_train)
    y_train_norm = scaler_y.transform(y_train)
    x_test_norm = scaler_x.transform(x_test)
    y_test_norm = scaler_y.transform(y_test)

    # Create a dataloader
    train_dataset = CustomDataset(np.array(x_train_norm), np.array(y_train_norm))
    # test_dataset = CustomDataset(np.array(x_test_norm), np.array(y_test_norm))
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False
    # )

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    loss_train_all = []
    loss_test_all = []
    # Train the model
    for step in range(nsteps):
        start_time = time.time()
        for x_batch_tensor, y_batch_tensor in trainloader:
            x_batch, y_batch = jnp.array(x_batch_tensor), jnp.array(y_batch_tensor)
            model, opt_state, loss_batch = make_step(model, opt_state, x_batch, y_batch)
        epoch_time = time.time() - start_time

        loss_train = loss_func_batch(model, x_train_norm, y_train_norm, loss_func)
        loss_test = loss_func_batch(model, x_test_norm, y_test_norm, loss_func)

        loss_train_all.append(loss_train)
        loss_test_all.append(loss_test)
        logging.info(
            f"The training loss of step {step}: {loss_train}; the test loss of step {step}: {loss_test}."
        )
    
    # Make predictions on both training and test data
    pred_y_norm_train = jax.vmap(model)(x_train_norm)
    pred_y_norm_test = jax.vmap(model)(x_test_norm)

    pred_y_train = scaler_y.inverse_transform(pred_y_norm_train)
    pred_y_test = scaler_y.inverse_transform(pred_y_norm_test)

    # Save the loss values
    logging.info("Saving the loss values ...")
    f_loss = dir_save / 'loss.csv'
    loss_df = pd.DataFrame(
        jnp.array([loss_train_all, loss_test_all]).T, columns=["training", "test"]
    )
    loss_df.to_csv(f_loss, index=False)

    # Save the predictions
    logging.info("Saving the predictions ...")
    f_pred_train = dir_save / 'predictions_train.txt'
    f_pred_test = dir_save / 'predictions_test.txt'
    np.savetxt(f_pred_train, pred_y_train)
    np.savetxt(f_pred_test, pred_y_test)

    # Save the model
    logging.info("Saving the trained model ...")
    f_model = dir_save / 'model.eqx'
    save_model(f_model, model_args, model)

    return model, loss_train_all, loss_test_all, pred_y_train, pred_y_test


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        # Return the size of the dataset, which is the length of one of the arrays
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve and return the corresponding elements from both arrays
        sample1 = self.x[idx]
        sample2 = self.y[idx]
        return sample1, sample2


# Function for saving a model
def save_model(filename: str, hyperparams: Dict, model: eqx.Module) -> None:
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def get_activation(activation: str='tanh'):
    if activation.lower() == 'tanh':
        return jax.nn.tanh

    elif activation.lower() == 'leaky_relu':
        return jax.nn.leaky_relu

    elif activation.lower() == 'sigmoid':
        return jax.nn.sigmoid

    elif activation.lower() == 'relu':
        return jax.nn.relu

    elif activation.lower() == 'identity':
        return lambda x: x
    
    else:
        raise Exception('Unknown activation: %s' % activation)