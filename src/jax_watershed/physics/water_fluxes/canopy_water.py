"""
Canopy water balance.

Author: Peishi Jiang
Date: 2023.06.13

"""  # noqa: E501

# import jax
import jax.numpy as jnp

from typing import Tuple
from ...shared_utilities.types import Float_0D


def canopy_water_balance(
    Δt: Float_0D,
    W_can: Float_0D,
    L: Float_0D,
    S: Float_0D,
    P: Float_0D,
    E_vw: Float_0D
    # ) -> List[Float_0D]:
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D]:
    """Calculate the components of the canopy water balance.

    Args:
        Δt (Float_0D): The time interval [s]
        W_can (Float_0D): The canopy water at the current step [kg m-2 s-1]
        L (Float_0D): The leaf area index [-]
        S (Float_0D): The stem area index [-]
        P (Float_0D): The rain precipitation [kg m-2 s-1]
        E_vw (Float_0D): The canopy evaporation [kg m-2 s-1]

    Returns:
        Float_0D: The change of the canopy water
    """
    # The interception of water by canopy
    # from precipitation [kg m-2 s-1]
    α = 1.0
    f_pi = α * jnp.tanh((L + S))  # Eq.(7.4) in CLM5
    Q_intr = f_pi * P  # Eq.(7.2) in CLM5

    # The canopy drip flux [kg m-2 s-1]
    # Based on Eqs.(7.8) - (7.13) in CLM5
    p = 0.1  # [kg m-2] from Dickinson et al. (1993)
    W_canmax = p * (L + S)
    Q_drip = (W_can - W_canmax) / Δt + Q_intr
    Q_drip = jnp.max(jnp.array([0.0, Q_drip]))

    # print(Q_intr, Q_drip, E_vw, W_canmax)
    # Change of the canopy water balance
    ΔW_can_Δt = Q_intr - Q_drip - E_vw

    return ΔW_can_Δt, Q_intr, Q_drip, f_pi
