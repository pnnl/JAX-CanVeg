"""Class for computing evapotranspiration."""
# TODO: Add units and documentation

import jax
import jax.numpy as jnp
# from jaxtyping import Array

from ..types import Float_general
from .constants import L as le
from .constants import P_AIR as P
from .constants import RHO_AIR as rho
from .constants import C_AIR as c
from .constants import VONKARMAN_CONSTANT as k
from .constants import SECONDS_TO_DAY as s2d

import equinox as eqx
from equinox.module import static_field


# Penman-Monteith equation
def calculate_evaporation_pm(
    R: Float_general, 
    G: Float_general, 
    t: Float_general, 
    uz: Float_general, 
    vpd: Float_general, 
    rc: Float_general, 
    dh: Float_general, 
    zh: Float_general, 
    zm: Float_general, 
    zoh: Float_general, 
    zom: Float_general
    ) -> Float_general: 
    # Some constants (McMhon et al. 2013)
    # le  = 2.45     # The latent heat of vaporization [MJ kg-1]
    # P   = 101.3    # atmospheric pressure [kPa]
    # rho = 1.2      # mean density of air at 20 degC [kg m-3]
    # c   = 0.001013 # specific heat of air [MJ kg-1 K-1]
    # k   = 0.41     # von Karman constant

    # Compute saturated vapor pressure
    es = 0.6108 * jnp.exp(17.27*t/(t+237.3))
    
    # Calculate d and r
    # d = 4098. * es / (ta+237.3)**2  # Dingman (2002)
    d = 4098. * es / (t+237.3)**2.   # [kPa degC-1] Allen et al. (1998)
    r = 0.00163 * P / le              # [kPa degC-1] Allen et al. (1998)

    # Calculate ra
    ra = jnp.log((zm-dh)/zom) * jnp.log((zh-dh)/zoh) / (k**2*uz) # [s m-1] assuming the unit of uz is m/s

    # TODO: Double check
    # Converting the resistance unit from s m-1 to d m-1
    ra = ra * s2d  # [d m-1]
    rc = rc * s2d  # [d m-1]
    
    # Calculate the evapotranspiration
    # jax.debug.print("resistance term: {}", jnp.array([rc,ra]))
    e1 = 1/le * (d*(R-G)) / (d + r*(1+rc/ra))
    e2 = 1/le * (rho*c*vpd/ra) / (d + r*(1+rc/ra))
    # e1 = 1/le * (d*(R-G)) / (d + r)
    # e2 = 1/le * (rho*c*vpd/ra) / (d + r)
    e  = e2+e1
    # jax.debug.print('PM ET aerodynamic terms: {}', jnp.array([rho, c, vpd, ra, rc]))
    # jax.debug.print('PM ET terms: {}', jnp.array([e, d, R-G, e1,e2]))
    # jax.debug.print('ET terms: {}', jnp.array([R, G, e1, e2]))
    return e*1e-3  # [m d-1]

# Penman equation
def calculate_evaporation_p(
    R: Float_general, 
    t: Float_general, 
    uz: Float_general, 
    vpd: Float_general
    ) -> Float_general: 
    # Some constants (McMhon et al. 2013)
    # le  = 2.45     # The latent heat of vaporization [MJ kg-1]
    # P   = 101.3    # atmospheric pressure [kPa]

    # Compute saturated vapor pressure
    es = 0.6108 * jnp.exp(17.27*t/(t+237.3))
    
    # Calculate d and r
    # d = 4098. * es / (ta+237.3)**2  # Dingman (2002)
    d = 4098. * es / (t+237.3)**2.   # [kPa degC-1] Allen et al. (1998)
    r = 0.00163 * P / le              # [kPa degC-1] Allen et al. (1998)

    # Calculate ea
    # u2 = uz * jnp.log(2./)
    # fu = 0.26*(1+0.54*u2)  # Eq(10.16) in Brutsaert (1982)
    fu = 0.26*(1+0.54*uz)  # Eq(10.16) in Brutsaert (1982)
    ea = fu*vpd            # Eq(10.17) in Brutsaert (1982)
    
    # Calculate the evapotranspiration
    e1 = d/(d+r)*R/le 
    e2 = r/(d+r)*ea
    e = e1 + e2 # [mm d-1], Eq(10.15) in Brutsaert (1982)
    # jax.debug.print("Penman evaporation terms: {}", jnp.array([e, e1, e2]))
    return e*1e-3  # [m d-1]

# class PenmanMonteith(eqx.Module):
#     dh: Array
#     zh: Array
#     zm: Array
#     zoh: Array
#     zom: Array

#     def __init__(self, dh: jnp.array, zh: jnp.array, zm: jnp.array, zoh: jnp.array, zom: jnp.array, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.dh = dh
#         self.zh = zh
#         self.zm = zm
#         self.zoh = zoh
#         self.zom = zom
    
#     def __call__(
#         self, 
#         R: Array, 
#         G: Array,
#         t: Array,
#         uz: Array,
#         vpd: Array,
#         rc: Array
#     ):
#         return calculate_evaporation_pm(
#             R, G, t, uz, vpd, rc, 
#             self.dh, self.zh, self.zm, self.zoh, self.zom
#         )
