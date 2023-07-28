import jax
import jax.numpy as jnp

from ..shared_utilities.types import Float_0D, Float_1D, Float_ND
from ..shared_utilities.constants import rgc1000


def llambda(tak: Float_0D) -> Float_0D:
    """Latent heat vaporization, J kg-1.

    Args:
        tak (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    y = 3149000.0 - 2370.0 * tak
    # add heat of fusion for melting ice
    y = jax.lax.cond(tak < 273.0, lambda x: x + 333.0, lambda x: x, y)
    return y


def es(tk: Float_ND) -> Float_ND:
    """Calculate saturated vapor pressure given temperature in Kelvin.

    Args:
        tk (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    tc = tk - 273.15
    return 613.0 * jnp.exp(17.502 * tc / (240.97 + tc))


def desdt(t: Float_ND) -> Float_ND:
    """Calculate the first derivative of es with respect to t.

    Args:
        t (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # llambda_all = llambda(t)
    llambda_all = jnp.vectorize(llambda)(t)
    return es(t) * llambda_all * 18.0 / (rgc1000 * t * t)


def des2dt(t: Float_ND) -> Float_ND:
    """Calculate the second derivative of the saturation vapor pressure
       temperature curve.

    Args:
        t (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # llambda_all = llambda(t)
    llambda_all = jnp.vectorize(llambda)(t)
    return -2.0 * es(t) * llambda_all * 18.0 / (rgc1000 * t * t * t) + desdt(
        t
    ) * llambda_all * 18.0 / (rgc1000 * t * t)


def soil_sfc_res(wg: Float_1D) -> Float_1D:
    """Calculate the soil surface resistance.

    Args:
        wg (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # Camillo and Gurney model for soil resistance
    # Rsoil= 4104 (ws-wg)-805, ws=.395, wg=0
    # ws= 0.395
    # wg is at 10 cm, use a linear interpolation to the surface, top cm, mean
    # between 0 and 2 cm
    wg0 = 1.0 * wg / 10.0
    # y=4104.* (0.395-wg0)-805.;

    # model of Kondo et al 1990, JAM for a 2 cm thick soil
    y = 3.0e10 * jnp.power((0.395 - wg0), 16.6)

    return y
