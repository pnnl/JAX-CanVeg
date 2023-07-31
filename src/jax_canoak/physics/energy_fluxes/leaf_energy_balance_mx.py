"""
Leaf energy balance subroutines and runctions, including:
- compute_qin()
- energy_balance_amphi()
- sfc_vpd()

Author: Peishi Jiang
Date: 2023.07.07.
"""

# import jax
# import jax.numpy as jnp

from ...subjects import ParNir, Ir, Para, Qin

# from typing import Tuple

# from ...shared_utilities.types import Float_0D, Float_ND


def compute_qin(quantum: ParNir, nir: ParNir, ir: Ir, prm: Para, qin: Qin) -> Qin:
    """Available energy on leaves for evaporation.
       Values are average of top and bottom levels of a layer.
       The index refers to the layer.  So layer 3 is the average
       of fluxes at level 3 and 4.  Level 1 is soil and level
       j+1 is the top of the canopy. layer jtot is the top layer

       Sunlit and shaded Absorbed Energy on the Top and Bottom of Leaves

    Args:
        quantum (ParNir): _description_
        nir (ParNir): _description_
        ir (Ir): _description_
        prm (Para): _description_

    Returns:
        Qin: _description_
    """
    # convert umol m-2 s-1 PPFD to W m-2 in visible
    vis_sun_abs = quantum.sun_abs / 4.6
    vis_sh_abs = quantum.sh_abs / 4.6

    # noticed in code ir.shade was multiplied by prm.ep twice, eg in
    # fIR_RadTranCanopy_MatrixV2 and here. removed prm.ep in the IR
    # subroutine
    qin.sun_abs = nir.sun_abs + vis_sun_abs + ir.shade * prm.ep
    qin.shade_abs = nir.sh_abs + vis_sh_abs + ir.shade * prm.ep

    return qin


# def sfc_vpd(
#     tlk: Float_0D,
#     leleafpt: Float_0D,
#     latent: Float_0D,
#     vapor: Float_0D,
#     rhov_air_z: Float_0D,
# ) -> Float_0D:
#     """This function computes the relative humidity at the leaf surface
#        for application int he Ball Berry Equation.
#        Latent heat flux, LE, is passed through the function, mol m-2 s-1
#        and it solves for the humidity at the leaf surface.

#     Args:
#         tlk (Float_0D): _description_
#         leleafpt (Float_0D): _description_
#         latent (Float_0D): _description_
#         vapor (Float_0D): _description_
#         rhov_air_z (Float_0D): _description_

#     Returns:
#         Float_0D: _description_
#     """
#     # Saturation vapor pressure at leaf temperature
#     es_leaf = es(tlk)

#     # Water vapor density at leaf [kg m-3]
#     rhov_sfc = (leleafpt / latent) * vapor + rhov_air_z
#     # jax.debug.print("rhov_air_z: {x}", x=rhov_air_z)

#     e_sfc = 1000 * rhov_sfc * tlk / 2.165  # Pa
#     vpd_sfc = es_leaf - e_sfc  # Pa
#     rhum_leaf = 1.0 - vpd_sfc / es_leaf  # 0 to 1.0

#     return rhum_leaf
