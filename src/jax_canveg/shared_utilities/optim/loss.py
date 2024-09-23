"""
Loss functions.

Author: Peishi Jiang
Date: 2024.09.04.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array


# Define a mean squared error function
def mse(y: Array, pred_y: Array):
    """Function for calculating mean square error function"""
    return jnp.mean((y - pred_y) ** 2)


# Define a relative mean squared error function
def relative_mse(y: Array, pred_y: Array):
    """Function for calculating relative square error function"""
    y_std = y.std()
    return jnp.mean(((y - pred_y) / y_std) ** 2)


# Define the percentage error
def mspe(y: Array, pred_y: Array):
    """Function for calculating relative square error function"""
    return jnp.mean(((y - pred_y) / y) ** 2)


# Define a weighted loss function
def weighted_loss(y: Array, pred_y: Array, loss: Callable, weights: Array):
    """Weighted loss function

    Args:
        y (Array): the true values with shape (n_samples, n_features)
        pred_y (Array): the predicted values with shape (n_samples, n_features
        loss (Callable): the loss function
        weights (Array): the weights with shape (n_features,)
    """
    weights = weights / weights.sum()
    loss_all = jax.vmap(loss, in_axes=(1, 1))(y, pred_y)
    return (loss_all * weights).sum()
