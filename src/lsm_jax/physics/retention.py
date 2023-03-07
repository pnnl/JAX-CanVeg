# TODO: Add units

# import jax
# import jax.numpy as jnp
# from jaxtyping import Array

from ..types import Float_general

import equinox as eqx
from equinox.module import static_field

# Functions for soil water retention curves
def van_genuchten_model(
    theta    : Float_general,
    theta_sat: Float_general,
    theta_r  : Float_general,
    ksat     : Float_general,
    alpha    : Float_general,
    n        : Float_general
    )        : 
    m = 1. - 1./n   # Eq.7 in van Genuchten (1980)
    
    # Calculate the effective saturation
    theta_effect = (theta - theta_r) / (theta_sat - theta_r)
    
    # Calculate the soil water potential
    psi = 1./alpha * (theta_effect**(-1./m) - 1.) ** (1./n)
    
    # Calculate the unsaturated hydraulic conductivity
    k = ksat * theta_effect**0.5 * (1-(1-theta_effect)**(1./m)) ** 2.
    # jax.debug.print("van genuchten terms: {}", jnp.array([alpha, theta_effect, m, theta, theta_r, theta_sat]))
    return psi, k

def clapp_hornberger_model(
    theta    : Float_general,
    theta_sat: Float_general,
    ksat     : Float_general,
    psisat   : Float_general,
    b        : Float_general
    )        : 
    # Calculate the soil water potential
    psi = psisat * (theta / theta_sat) ** (-b) # [m], Clapp and Hornberger (1978)
    
    # Calculate the hydraulic conductivity
    k = ksat * (theta / theta_sat) ** (2*b+3)  # [-], Clapp and Hornberger (1978)

    return psi, k

# class SoilWaterRententionBase(eqx.Module):
#     theta_sat: Array
#     ksat: Array

#     def __init__(
#         self, 
#         theta_sat: jnp.array, 
#         ksat: jnp.array,
#         **kwargs
#     ) -> None:
#         super().__init__(**kwargs)
#         self.theta_sat = theta_sat
#         self.ksat = ksat

# class VanGenuchten(SoilWaterRententionBase):
#     theta_r: Array
#     alpha: Array
#     n: Array

#     def __init__(
#         self, 
#         theta_sat: jnp.array, 
#         theta_r: jnp.array,
#         ksat: jnp.array, 
#         alpha: jnp.array,
#         n: jnp.array,
#         **kwargs
#     ) -> None:
#         super().__init__(theta_sat, ksat, **kwargs)
#         self.theta_r = theta_r
#         self.alpha   = alpha
#         self.n       = n
    
#     def __call__(
#         self,
#         theta
#     ) -> list:
#         return van_genuchten_model(
#             theta, self.theta_sat, self.theta_r, self.ksat, self.alpha, self.n
#         )

# class ClappHornberger(SoilWaterRententionBase):
#     psisat: Array
#     b: Array

#     def __init__(
#         self, 
#         theta_sat: jnp.array, 
#         psisat: jnp.array,
#         ksat: jnp.array, 
#         b: jnp.array,
#         **kwargs
#     ) -> None:
#         super().__init__(theta_sat, ksat, **kwargs)
#         self.psisat = psisat
#         self.b      = b
    
#     def __call__(
#         self,
#         theta
#     ) -> list:
#         return clapp_hornberger_model(
#             theta, self.theta_sat, self.ksat, self.psisat, self.b
#         )
 
