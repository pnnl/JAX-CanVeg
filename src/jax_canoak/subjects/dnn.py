"""
This file defines multiple deep neural networks.

Author: Peishi Jiang
Date: 2023.10.20
"""

import jax
import jax.random as jrandom

import equinox as eqx
from equinox.nn import Linear
from jaxtyping import PRNGKeyArray, Array
from typing import Literal, Union


class MLP(eqx.Module):
    layers: tuple[Linear, ...]

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: int,
        depth: int,
        key: PRNGKeyArray,
        **kwargs
    ):
        super().__init__(**kwargs)
        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, True, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, True, key=keys[0]))
            for i in range(depth - 1):
                layers.append(Linear(width_size, width_size, True, key=keys[i + 1]))
            layers.append(Linear(width_size, out_size, True, key=keys[-1]))
        self.layers = tuple(layers)

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = layer(x)
            # x = jax.nn.relu(x)
            x = jax.nn.tanh(x)
        x = self.layers[-1](x)
        x = jax.nn.tanh(x)
        return x
