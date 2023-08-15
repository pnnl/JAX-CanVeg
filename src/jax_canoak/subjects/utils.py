import jax
import jax.numpy as jnp

from ..shared_utilities.types import Float_0D, Float_1D, Float_2D, Float_ND
from ..shared_utilities.types import Int_0D
from ..shared_utilities.constants import rgc1000
from ..shared_utilities.utils import dot


@jnp.vectorize
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
    # tc = tk - 273.15
    # return 613.0 * jnp.exp(17.502 * tc / (240.97 + tc))
    # Use the matlab version here to keep consistent
    return 100.0 * jnp.exp(54.8781919 - 6790.4985 / tk - 5.02808 * jnp.log(tk))


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


def conc_mx(
    source: Float_2D,
    soilflux: Float_1D,
    delz: Float_1D,
    dij: Float_2D,
    # met: Met,
    ustar: Float_1D,
    zL: Float_1D,
    cref: Float_1D,
    jtot: Int_0D,
    nlayers_atmos: Int_0D,
    # prm: Para,
    factor: Float_1D,
) -> Float_2D:
    """Subroutine to compute scalar concentrations from source
       estimates and the Lagrangian dispersion matrix.

    Args:
        source (Float_2D): _description_
        soilflux (Float_2D): _description_
        delz (Float_1D): _description_
        Dij (Float_2D): _description_
        met (Met): _description_
        cref (Float_1D): _description_
        prm (Para): _description_
        factor (Float_1D): _description_

    Returns:
        Float_2D: _description_
    """
    # ustar = met.ustar

    # Compute concentration profiles from Dispersion matrix
    ustar_ref = 1
    ustfact = ustar_ref / ustar  #   ustar_ref /ustar
    # ntime = ustfact.size

    # @jax.vmap
    # @partial(jax.vmap, )
    def calculate_ccnc(ustfact_e, zL, source_e, soilflux_e, factor_e, cref_e):
        # CC is the differential concentration (Ci-Cref)
        # Ci-Cref = SUM (Dij S DELZ), units mg m-3 or mole m-3
        # S = dfluxdz/DELZ
        # note delz values cancel
        # scale dispersion matrix according to friction velocity
        # disper = ustfact_e * dij[: prm.nlayers_atmos, :]
        disper = ustfact_e * dij[:nlayers_atmos, :]
        # disperzl=disper
        # Updated dispersion matrix
        disperzl = jax.lax.cond(
            zL < 0,
            lambda: disper * (0.973 * -0.7182) / (zL - 0.7182),
            lambda: disper * (-0.31 * zL + 1.00),
        )  # (nlayers_atmos, jktot)

        # Compute cncc for all layers
        # delz[:prm.jtot] * source_e[:prm.jtot] * disperzl[:prm.jtot,:]
        sumcc = dot(
            # delz[: prm.jtot] * source_e[: prm.jtot],
            # disperzl.T[: prm.jtot, : prm.nlayers_atmos],
            delz[:jtot] * source_e[:jtot],
            disperzl.T[:jtot, :nlayers_atmos],
        )
        sumcc = sumcc.sum(axis=0)  # (nlayers_atmos,)

        # scale dispersion matrix according to Z/L
        # dispersoil = ustfact_e * dij[: prm.nlayers_atmos, 0]  # (nlayers_atmos,)
        dispersoil = ustfact_e * dij[:nlayers_atmos, 0]  # (nlayers_atmos,)

        # add soil flux to the lowest boundary condition
        soilbnd = soilflux_e * dispersoil / factor_e  # (nlayers_atmos,)
        cc = sumcc / factor_e + soilbnd  # (nlayers_atmos,)

        # factor to adjust dij with alternative u* values
        # compute scalar profile below reference
        cncc_e = cc + cref_e - cc[-1]

        return cncc_e

    cncc = jax.vmap(calculate_ccnc, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)(
        # ustfact, met.zL, source, soilflux, factor, cref
        ustfact,
        zL,
        source,
        soilflux,
        factor,
        cref,
    )  # (ntime, nlayers_atmos)

    return cncc
