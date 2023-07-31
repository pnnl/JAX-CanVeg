"""
Soil energy balance functions and subroutines, including:
- soil_energy_balance()

Author: Peishi Jiang
Date: 2023.07.25.
"""

# import jax
# import jax.numpy as jnp

# from typing import Tuple

# from ...shared_utilities.types import Float_1D
from ...subjects import ParNir, Ir, Met, Prof, Para, Soil


def soil_energy_balance(
    quantum: ParNir, nir: ParNir, ir: Ir, met: Met, prof: Prof, prm: Para, soil: Soil
) -> Soil:
    """The soil energy balance model of Campbell has been adapted to
       compute soil energy fluxes and temperature profiles at the soil
       surface.  The model has been converted from BASIC to C.  We
       also use an analytical version of the soil surface energy
       balance to solve for LE, H and G.

       Combine surface energy balance calculations with soil heat
       transfer model to calculate soil conductive and convective heat
       transfer and evaporation rates.  Here, only the deep temperature
       is needed and G, Hs and LEs can be derived from air temperature
       and energy inputs.

       Soil evaporation models by Kondo, Mafouf et al. and
       Dammond and Simmonds are used. Dammond and Simmonds for example
       have a convective adjustment to the resistance to heat transfer.
       Our research in Oregon and Canada have shown that this consideration
       is extremely important to compute G and Rn_soil correctly.

    Args:
        quantum (ParNir): _description_
        nir (ParNir): _description_
        ir (Ir): _description_
        met (Met): _description_
        prof (Prof): _description_
        prm (Para): _description_
        soil (Soil): _description_

    Returns:
        Soil: _description_
    """
    # TODO!
    return soil
