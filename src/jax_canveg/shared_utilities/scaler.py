"""
A couple of scaler function.

Author: Peishi Jiang
Date: 2024.09.04.
"""

import jax
from jaxtyping import Array

from .types import Float_0D, Float_1D, Float_2D


def identity_scaler(x: Array):
    return x


def standardizer_1d(x: Float_1D, xmean: Float_0D, xstd: Float_0D):
    return (x - xmean) / xstd


def standardizer_nd(x: Float_2D, xmean: Float_1D, xstd: Float_1D):
    def standardize(x_1d: Float_1D, xmean: Float_1D, xstd: Float_1D):
        return (x_1d - xmean) / xstd

    return jax.vmap(standardize, in_axes=(0, None, None))(x, xmean, xstd)


def minmax_1d(x: Float_1D, xmin: Float_0D, xmax: Float_0D):
    return (x - xmin) / (xmax - xmin)


def minmax_nd(x: Float_2D, xmin: Float_1D, xmax: Float_1D):
    def minmax(x_1d: Float_1D, xmin: Float_1D, xmax: Float_1D):
        return (x_1d - xmin) / (xmax - xmin)

    return jax.vmap(minmax, in_axes=(0, None, None))(x, xmin, xmax)
