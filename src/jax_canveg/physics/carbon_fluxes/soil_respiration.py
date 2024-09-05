"""
Soil respiration modules, including:
- soil_respiration()
- soil_respiration_q10_power()
- soil_respiration_alfalfa()
- soil_respiration_dnn()

Author: Peishi Jiang
Date: 2024.09.04.
"""

import jax
import jax.numpy as jnp

from ...shared_utilities.types import Float_1D
from ...subjects import Para

from ...shared_utilities.utils import dot, add


def soil_respiration(
    Ac: Float_1D,
    Tsoil: Float_1D,
    soilmoisture: Float_1D,
    veght: Float_1D,
    Rd: Float_1D,
    prm: Para,
    soilresp: int = 0,
) -> Float_1D:
    """General function for calculating soil respiration"""
    arg = [Ac, Tsoil, soilmoisture, veght, Rd, prm]
    return jax.lax.switch(
        soilresp,
        [soil_respiration_alfalfa, soil_respiration_q10_power, soil_respiration_dnn],
        *arg
    )


def soil_respiration_q10_power(
    Ac: Float_1D,
    Tsoil: Float_1D,
    soilmoisture: Float_1D,
    veght: Float_1D,
    Rd: Float_1D,
    prm: Para,
) -> Float_1D:
    # Use the Q10 power equation: SR = a x b^{(Ts-10)/10} x SWC^c
    a, b, c = prm.q10a, prm.q10b, prm.q10c

    Tsoil_C = Tsoil - 273.15
    temp1 = a * jnp.power(b, (Tsoil_C - 10.0) / 10.0)
    temp2 = jnp.power(soilmoisture, c)

    return temp1 * temp2


def soil_respiration_alfalfa(
    Ac: Float_1D,
    Tsoil: Float_1D,
    soilmoisture: Float_1D,
    veght: Float_1D,
    Rd: Float_1D,
    prm: Para,
) -> Float_1D:
    # Use statistical model derived from ACE analysis on alfalfa
    x = jnp.array([Ac + Rd, Tsoil, soilmoisture, veght])  # (4, ntime)

    b1 = jnp.array(
        [-1.51316616076344, 0.673139978230031, -59.2947930385706, -3.33294857624960]
    )
    b2 = jnp.array(
        [1.38034307225825, 3.73823712105636, 59.9066980644239, 1.36701108005293]
    )
    b3 = jnp.array(
        [2.14475255616910, 19.9136298988773, 0.000987230986585085, 0.0569682453563841]
    )

    temp = x / add(b3, x)
    temp = dot(b2, temp)
    phi = add(b1, temp)  # (4, ntime)
    phisum = phi.sum(axis=0)  # (ntime,)

    b0_0 = 4.846927939437475
    b0_1 = 2.22166756601498
    b0_2 = -0.0281417120818586

    # Reco-Rd = Rsoil
    resp = b0_0 + b0_1 * phisum + b0_2 * phisum * phisum - Rd

    return resp


def soil_respiration_dnn(
    # Tsoil_K: Float_2D,
    # swc: Float_1D,
    # prm: Para,
    # RsoilDL: eqx.Module,
    Ac: Float_1D,
    Tsoil: Float_1D,
    soilmoisture: Float_1D,
    veght: Float_1D,
    Rd: Float_1D,
    prm: Para,
) -> Float_1D:
    RsoilDL = prm.RsoilDL

    # Normalize the inputs
    Tsoil_C = Tsoil - 273.15
    # Tsfc_norm = (Tsfc - prm.var_mean.Tsoil) / prm.var_std.Tsoil
    # swc_norm = (swc - prm.var_mean.soilmoisture) / prm.var_std.soilmoisture
    Tsoil_norm = (Tsoil_C - prm.var_min.T_air) / (  # pyright: ignore
        prm.var_max.T_air - prm.var_min.T_air  # pyright: ignore
    )  # pyright: ignore
    swc_norm = (soilmoisture - prm.var_min.soilmoisture) / (  # pyright: ignore
        prm.var_max.soilmoisture - prm.var_min.soilmoisture  # pyright: ignore
    )  # pyright: ignore

    # Get the inputs
    # x = Tsoil_norm
    # jax.debug.print("Tsoil shape: {a}", a=x.shape)
    # x = jnp.array([Tsoil_norm[:, 0], swc_norm]).T
    x = jnp.array([Tsoil_norm, swc_norm]).T
    # x = jnp.expand_dims(Tsfc_norm, axis=-1)

    # Perform the Rsoil calculation
    rsoil_norm = jax.vmap(RsoilDL)(x)  # pyright: ignore
    rsoil_norm = rsoil_norm.flatten()

    # jax.debug.print("rsoil_norm: {x}", x=rsoil_norm)
    # jax.debug.print("rsoil_norm: {x}", x=RsoilDL.layers[0].weight)

    # Transform it back
    # rsoil = rsoil_norm * prm.var_std.rsoil + prm.var_mean.rsoil
    rsoil = (
        rsoil_norm * (prm.var_max.rsoil - prm.var_min.rsoil)  # pyright: ignore
        + prm.var_min.rsoil  # pyright: ignore
    )

    return rsoil


# def soil_respiration(
#     Ts: Float_1D, base_respiration: Float_0D = 8.0
# ) -> Tuple[Float_1D, Float_1D]:
#     """Compute soil respiration

#     Args:
#         Ts (Float_0D): _description_
#         base_respiration (Float_0D, optional): _description_. Defaults to 8..
#     """
#     # After Hanson et al. 1993. Tree Physiol. 13, 1-15
#     # reference soil respiration at 20 C, with value of about 5 umol m-2 s-1
#     # from field studies

#     # assume Q10 of 1.4 based on Mahecha et al Science 2010, Ea = 25169
#     respiration_mole = base_respiration * jnp.exp(
#         (25169.0 / 8.314) * ((1.0 / 295.0) - 1.0 / (Ts + 273.16))
#     )

#     # soil wetness factor from the Hanson model, assuming constant and wet soils
#     respiration_mole *= 0.86

#     # convert soilresp to mg m-2 s-1 from umol m-2 s-1
#     respiration_mg = respiration_mole * 0.044

#     return respiration_mole, respiration_mg
