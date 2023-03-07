"""This calculates sensible/latent/ground heat fluxes based on the energy balance equation."""
# TODO: Look into how to calculate the heat exchange coefficient in the computation of sensible heat flux

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from .constants import PI as pi
from .constants import C_AIR as c
from .constants import RHO_AIR as rho
from .constants import BOLTZMANN_CONSTANT as sigma
from .constants import L as l
from .constants import SECONDS_TO_DAY as s2d
from .constants import VONKARMAN_CONSTANT as k
from ..types import Float_general, Float_0D
from typing import Tuple

# Calculation of the net radiation
# Ref: Brutsaert (1982)
def calculate_net_radiation(
    sw_in: Float_general, 
    lw_in: Float_general,
    ts: Float_general,
    alpha: Float_0D,
    em: Float_0D,
) -> Float_general:
    """Calculation of net radiation.

    Args:
        sw_in (Float_general): In-coming short-wave radiation [MJ m-2 day-1]
        lw_in (Float_general): In-coming long-wave radiation [MJ m-2 day-1]
        ts (Float_general): Surface temperature [K]
        alpha (Float_0D): Albedo [-]
        em (Float_0D): Emissivity [-]

    Returns:
        Float_general: Net radiation [MJ m-2 day-1]
    """
    # Energy unit: MJ m-2 day-1
    # Temperature unit: K
    return sw_in * (1-alpha) + em * (lw_in - sigma*ts**4)

# Calculation of latent heat flux
def calculate_latent_heat_flux(
    e: Float_general
) -> Float_general:
    """Calculation of latent heat flux.

    Args:
        e (Float_general): Evapotranspiration [mm day-1 or kg m-2 day-1]

    Returns:
        Float_general: latent heat flux [MJ m-2 day-1]
    """
    return l * e

# Calculation of sensible heat flux
def calculate_sensible_heat_flux(
    ts: Float_general,
    ta: Float_general,
    u: Float_general,
    z: Float_0D,
    z0: Float_0D
) -> Float_general:
    """Calculation of sensible heat flux. Ref: Hinzman et al., JGR, 1998.

    Args:
        ts (Float_general): Surface temperature [degK] or [degC]
        ta (Float_general): Air temperature [degK] or [degC]
        u (Float_general): Wind speed [m/s]
        z (Float_0D): The height of the wind speed [m]
        z0 (Float_0D): The average surface roughness [m]

    Returns:
        Float_general: sensible heat flux [MJ m-2 day-1]
    """
    D = calculate_heat_exchange_coefficient(u, z, z0)
    sh = c * rho * D * (ts-ta) / s2d
    # jax.debug.print('Sensible heat components: {}', jnp.array([sh, D, c, rho, ts, ta, s2d]))
    return sh

# Calculation of ground heat flux
def calculate_ground_heat_flux(
    ts: Float_general,
    ta: Float_general,
    k: Float_0D,
    dz: Float_0D
) -> Float_general:
    """Calculation of the ground heat flux. Ref: Drewry et al., JGR, 2010 (supporting information).

    Args:
        ts (Float_general): Surface temperature [degK] or [degC]
        ta (Float_general): Air temperature [degK] or [degC]
        k (Float_0D): Soil heat conductivity [MJ m-1 day-1 degC-1]
        dz (Float_0D): The thickness of the near-surface air layer [m]

    Returns:
        Float_general: Ground heat flux [MJ m-2 day-1]
    """
    return k * (ts-ta) / dz

# Calculation of the surface soil temperature -- Option 1
def calculate_surface_temperature_based_on_energy_balance(
    rn: Float_general,
    le: Float_general,
    ta: Float_general,
    u: Float_general,
    z: Float_0D,
    z0: Float_0D,
    k: Float_0D,
    dz: Float_0D
) -> Float_general:
    """Calculation of soil surface temperature based on the energy balance equation.
       Given Rn = LE + H + G. We can have the following expression --
          ts = ta + (Rn-LE) / (cp*rho_a*D+k/dz)

    Args:
        rn (Float_general): Net radiation [MJ m-2 day-1]
        le (Float_general): Latent heat flux [MJ m-2 day-1]
        ta (Float_general): Air temperature [degK] or [degC]
        u (Float_general): Wind speed [m/s]
        z (Float_0D): The height of the wind speed [m]
        z0 (Float_0D): The average surface roughness [m]
        k (Float_0D): Soil heat conductivity [MJ m-1 day-1 degC-1]
        dz (Float_0D): The thickness of the near-surface air layer [m]

    Returns:
        Float_general: Soil surface temperature [degC or degK]
    """
    D = calculate_heat_exchange_coefficient(u, z, z0)
    return ta + (rn+le) / (c*rho*D/s2d+k/dz)

# Calculation of the soil surface temperature -- Option 2
def calculate_surface_temperature_based_on_energy_balance_newton(
    sw_in: Float_general, 
    lw_in: Float_general,
    e: Float_general,
    ta: Float_general,
    u: Float_general,
    alpha: Float_0D,
    em: Float_0D,
    z: Float_0D,
    z0: Float_0D,
    k: Float_0D,
    dz: Float_0D
) -> Float_general:
    """Calculation of the surface temperature based on the energy balance equation using
       Newton Raphson method.

    Args:
        sw_in (Float_general): _description_
        lw_in (Float_general): _description_
        e (Float_general): _description_
        ta (Float_general): _description_
        u (Float_general): _description_
        alpha (Float_0D): _description_
        em (Float_0D): _description_
        z (Float_0D): _description_
        z0 (Float_0D): _description_
        k (Float_0D): _description_
        dz (Float_0D): _description_

    Returns:
        Float_general: Surface temperature [degC or degK]
    """
    def energy_balance(ts):
        rn = calculate_net_radiation(sw_in, lw_in, ts, alpha, em)
        le = calculate_latent_heat_flux(e)
        s  = calculate_sensible_heat_flux(ts, ta, u, z, z0)
        g  = calculate_ground_heat_flux(ts, ta, k, dz)
        return rn - le - s - g
    
    # Use ta as the initial condition
    result = minimize(fun=energy_balance, x0=ta)
    ts     = result.x
    return ts

# Calculation of the RHS of the surface temperature based on Bhumralkar, 1976
def calculate_rhs_surface_temperature(
    ts: Float_0D,
    ts_avg: Float_0D,
    g: Float_0D,
    k: Float_0D,
    cs: Float_0D,
    rhos: Float_0D,
    omega: Float_0D
) -> Tuple[Float_0D, Float_0D]:
# ) -> Float_0D, Float_0D:
    """Calculate the ODE RHS of the surface and averaged temperature based on Bhumralkar (1976) and Noilhan and Planton (1989)

    Args:
        ts (Float_0D): Soil surface temperature [degC or degK]
        ts_avg (Float_0D): Averaged surface temperature [degC or degK]
        g (Float_0D): Ground heat flux [MJ m-2 day-1]
        k (Float_0D): Soil thermal conductivity [MJ m-1 day-1 degK-1 or MJ m-1 day-1 degC-1]
        cs (Float_0D): Soil specific heat [MJ kg-1 degK-1 or MJ kg-1 degC-1]
        rhos (Float_0D): Soil density [kg m-3]
        omega (Float_0D): The frequency of the osillation [rad day-1], equal to 2*pi / period

    Returns:
        tuple(Float_0D, Float_0D): the ODE RHS of the surface and averaged temperature
    """
    c1      = cs*rhos + (k*cs*rhos/(2*omega)) ** 0.5
    dts     = g / c1 - (ts-ts_avg) / c1 * (k*cs*rhos*omega/2) ** 0.5
    dts_avg = 2*pi/omega * (ts-ts_avg)
    # jax.debug.print('c1 components: {}', jnp.array([c1, cs*rhos, (k*cs*rhos/(2*omega)) ** 0.5, cs, rhos, k, omega]))
    # jax.debug.print('dts: {}', jnp.array([dts, g/c1, g, c1]))
    return dts, dts_avg

# Calculation of the heat exchange coefficients D
def calculate_heat_exchange_coefficient(
    u: Float_general,
    z: Float_0D,
    z0: Float_0D
) -> Float_general:
    """Calculation of heat/vapor exchange coefficient for neutral atmospheric conditions. Ref: Hinzman et al., JGR, 1998.

    Args:
        u (Float_general): Wind speed [m/s]
        z (Float_0D): The height of the wind speed [m]
        z0 (Float_0D): The average surface roughness [m]

    Returns:
        Float_general: Heat exchange coefficient for neutral atmospheric conditions [m/s]
    """
    # jax.debug.print("Heat exchange coefficient components: {}", z/z0)
    return u * k**2 / (jnp.log(z/z0))**2