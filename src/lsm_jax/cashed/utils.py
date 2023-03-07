# TODO: Add units

import jax
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx
from equinox.module import static_field

class CanopyResistance(eqx.Module):
    rsmin: Array
    theta_wp: Array
    theta_lim: Array
    tamin: Array
    tamax: Array
    taopt: Array
    w: Array

    def __init__(
        self,
        rsmin: jnp.array,
        theta_wp: jnp.array,
        theta_lim: jnp.array,
        tamin: jnp.array,
        tamax: jnp.array,
        taopt: jnp.array,
        w: jnp.array,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.rsmin, self.w = rsmin, w
        self.theta_wp, self.theta_lim = theta_wp, theta_lim
        self.tamin, self.tamax = tamin, tamax
        self.taopt = taopt
    
    def __call__(
        self,
        lai: Array,
        theta: Array,
        ta: Array,
        vpd: Array
    ):
        f1 = calculate_f1(theta, self.theta_wp, self.theta_lim)
        f2 = calculate_f2(ta, self.tamin, self.tamax, self.taopt)
        f3 = calculate_f3(vpd, self.w)
        return self.rsmin / lai * (f1*f2*f3)


def calculate_canopy_resistance(rsmin, lai, theta, ta, vpd, theta_wp, theta_lim, tamin, tamax, taopt, w):
    """Calculate the canopy resistance"""
    f1 = calculate_f1(theta, theta_wp, theta_lim)
    f2 = calculate_f2(ta, tamin, tamax, taopt)
    f3 = calculate_f3(vpd, w)
    return rsmin / lai * (f1*f2*f3)


# Functions for calculating the canopy resistance
def calculate_f1(theta, theta_wp, theta_lim):
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

def calculate_f2(ta, tamin, tamax, taopt):
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

def calculate_f3(vpd, w):
    """Dependencies on vapor pressure deficit and wind speed"""
    return 1 - w*vpd
