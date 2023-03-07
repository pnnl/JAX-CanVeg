"""Class for computing infiltration rate."""
# TODO: Add units and documentation

import jax
import jax.numpy as jnp
# from jaxtyping import Array

from ..types import Float_general

import equinox as eqx
from equinox.module import static_field

# Philips methods
def calculate_infiltration_philips(
    t: Float_general, 
    theta: Float_general, 
    theta_sat: Float_general, 
    ksat: Float_general, 
    psi: Float_general
    ) -> Float_general:
    a0    = 0.5 * ksat   # [m d-1]
    # psi_f = psi_a * (2*r+2.5) / (r+2.5)              # [m] (Clapp and Hornberger 1978)
    sorp  = (2*ksat*(theta_sat - theta)*psi) ** 0.5  # [m d-1/2] (Rawls et al, 1982)
    i = 0.5 * sorp / t**0.5 + a0  # [m d-1]
    return i 

# Green-Ampt method
def calculate_infiltration_greenampt(
    ci: Float_general, 
    k: Float_general, 
    psi: Float_general, 
    theta: Float_general, 
    theta_sat: Float_general
    ) -> Float_general:
    # Calculate infiltration capacity 
    i = k * (psi*(theta_sat-theta)+ci) / ci
    return i

# class InfiltrationBase(eqx.Module):
#     theta_sat: Array

#     def __init__(self, theta_sat: jnp.array, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.theta_sat = theta_sat
    

# class Philips(InfiltrationBase):
#     ksat: Array

#     def __init__(self, theta_sat: jnp.array, ksat: jnp.array, **kwargs) -> None:
#         super().__init__(theta_sat, **kwargs)
#         self.ksat = ksat
    
#     def __call__(self, t: jnp.array, theta: jnp.array, psi_f: jnp.array):
#         return calculate_infiltration_philips(t, theta, self.theta_sat, self.ksat, psi_f)


# class GreenAmpt(InfiltrationBase):

#     def __init__(self, theta_sat: jnp.array, **kwargs) -> None:
#         super().__init__(theta_sat, **kwargs)
    
#     def __call__(self, ci: jnp.array, k: jnp.array, psi: jnp.array, theta: jnp.array):
#         return calculate_infiltration_greenampt(ci, k, psi, theta, self.theta_sat)

