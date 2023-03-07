"""This implements the basic properties of soil using equinox."""

import jax
import jax.numpy as jnp

# from jaxtyping import Array, Float, Int
from typing import Union
from ..types import Float_0D, Float_general
from typeguard import typechecked

import equinox as eqx
from equinox.module import static_field

# @typechecked
class Canopy(eqx.Module):

    rsmin: Float_0D
    tamin: Float_0D
    tamax: Float_0D
    taopt: Float_0D
    w: Float_0D

    dh: Float_0D = static_field
    zh: Float_0D = static_field
    zm: Float_0D = static_field
    zoh: Float_0D = static_field
    zom: Float_0D = static_field

    def __init__(
        self, 
        rsmin: Float_0D,
        tamin: Float_0D,
        tamax: Float_0D,
        taopt: Float_0D,
        w: Float_0D,
        dh: Float_0D,
        zh: Float_0D,
        zm: Float_0D,
        zoh: Float_0D,
        zom: Float_0D,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.rsmin = rsmin
        self.tamin = tamin
        self.tamax = tamax
        self.taopt = taopt
        self.w     = w
        self.dh    = dh
        self.zh    = zh
        self.zm    = zm
        self.zoh   = zoh
        self.zom   = zom


def calculate_canopy_resistance(
    lai: Float_general, theta: Float_general, ta: Float_general, vpd: Float_general, 
    rsmin: Float_general, theta_wp: Float_general, theta_lim: Float_general, 
    tamin: Float_general, tamax: Float_general, taopt: Float_general, 
    w: Float_general
) -> Float_general:
    """Calculate the canopy resistance"""
    # TODO: We might need soil moisture in the canopy
    f1 = calculate_f1(theta, theta_wp, theta_lim)
    # f1 = 1.
    f2 = calculate_f2(ta, tamin, tamax, taopt)
    f3 = calculate_f3(vpd, w)
    return rsmin / lai * (f1*f2*f3)


# Functions for calculating the canopy resistance
def calculate_f1(
    theta: Float_general, 
    theta_wp: Float_general, 
    theta_lim: Float_general
    ) -> Float_general:
    """Dependencies on soil moisture"""
    # index = jnp.digitize(c, jnp.array([0, 3, 5]))
    index = jnp.digitize(theta, jnp.array([theta_wp, theta_lim, ]))
    
    branches = [
        lambda x:0.,
        lambda x:(theta - theta_wp) / (theta_lim - theta_wp),
        lambda x:1.
    ]
    result = jax.lax.switch(index, branches, theta)
    return result

def calculate_f2(
    ta: Float_general, 
    tamin: Float_general, 
    tamax: Float_general, 
    taopt: Float_general
    ) -> Float_general:
    """Dependencies on temperature"""
    index = jnp.digitize(ta, jnp.array([tamin, taopt, tamax, ]))
    
    branches = [
        lambda x:0.,
        lambda x:1. - (taopt-ta)/(taopt-tamin),
        lambda x:1.,
        lambda x:0.
    ]
    result = jax.lax.switch(index, branches, ta)
    return result

def calculate_f3(
    vpd: Float_general, 
    w: Float_general
    ) -> Float_general:
    """Dependencies on vapor pressure deficit and wind speed"""
    return 1 - w*vpd
