"""
Canopy structure functions, including:
- angle()
- gammaf()
- lai_time()
- freq()

Author: Peishi Jiang
Date: 2023.06.28.
"""

import jax
import jax.numpy as jnp

# from typing import Tuple

from ...shared_utilities.types import Float_0D


def angle():
    pass


def gammaf(x: Float_0D) -> Float_0D:
    """Gamma function.

    Args:
        x (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    gam = (
        (1.0 / (12.0 * x))
        + (1.0 / (288.0 * x * x))
        - (139.0 / (51840.0 * jnp.power(x, 3.0)))
    )
    gam = gam + 1.0

    # x has to be positive !
    y = jax.lax.cond(
        x > 0,
        lambda x: jnp.sqrt(2.0 * jnp.pi / x) * jnp.power(x, x) * jnp.exp(-x) * gam,
        lambda x: 0.0,
        x,
    )

    return y


def lai_time():
    pass


def freq(lflai: Float_0D) -> Float_0D:
    """Use the beta distribution to compute the probability frequency distribution
       for a known mean leaf inclication angle starting from the top of the canopy,
       where llai=0 (after Goel and Strebel (1984)).

    Args:
        lflai (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # spherical leaf angle
    MEAN = 57.4
    STD = 26
    VAR = STD * STD + MEAN * MEAN
    nuu = (1.0 - VAR / (90.0 * MEAN)) / (VAR / (MEAN * MEAN) - 1.0)
    MU = nuu * ((90.0 / MEAN) - 1.0)
    SUM = nuu + MU

    FL1 = gammaf(SUM) / (gammaf(nuu) * gammaf(MU))
    MU1 = MU - 1.0
    nu1 = nuu - 1.0

    CONS = 1.0 / 9.0

    # COMPUTE PROBABILITY DISTRIBUTION FOR 9 ANGLE CLASSES
    # BETWEEN 5 AND 85 DEGREES, WITH INCREMENTS OF 10 DEGREES
    def calculate_bden(carry, i):
        ANG = 10.0 * (i + 1) - 5.0
        FL2 = jnp.power((1.0 - ANG / 90.0), MU1)
        FL3 = jnp.power((ANG / 90.0), nu1)
        y = CONS * FL1 * FL2 * FL3
        return None, y

    _, bdens = jax.lax.scan(calculate_bden, None, jnp.arange(9))

    return bdens
