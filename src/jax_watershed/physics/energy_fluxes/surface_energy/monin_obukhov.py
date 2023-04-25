"""
Calculate the Momin Obukhov length using the temperature/specific humidity in the 
dual source model.

Author: Peishi Jiang
Date: 2023.03.28.
"""

# import jax
import jax.numpy as jnp

from typing import Tuple
from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import G as g

from ..turbulent_fluxes import calculate_ψc_most, calculate_ψm_most, calculate_L_most
from ..turbulent_fluxes import (
    calculate_ustar_most,
    calculate_qstar_most,
    calculate_Tstar_most,
)
from ..turbulent_fluxes import (
    calculate_scalar_conduct_surf_atmos,
    calculate_total_conductance_leaf_boundary,
    calculate_total_conductance_leaf_water_vapor,
)
from ..turbulent_fluxes import (
    calculate_conductance_ground_canopy,
    calculate_conductance_ground_canopy_water_vapo,
)
from .surface_state import calculate_qs_from_qvqgqa, calculate_Ts_from_TvTgTa
from ...water_fluxes import q_from_e_pres, esat_from_temp


def func_most_dual_source(
    L_guess: Float_0D,
    pres: Float_0D,
    T_v: Float_0D,
    T_g: Float_0D,
    T_a: Float_0D,
    u_a: Float_0D,
    q_a: Float_0D,
    q_g: Float_0D,
    L: Float_0D,
    S: Float_0D,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
) -> Float_0D:
    """Residual function of the Monin Obukhov self-similarity calculation based on
       the environmental states of a dual source model.

    Args:
        L_guess (Float_0D): The initial guess of the Obukhov length [m]
        pres (Float_0D): The atmospheric pressure [Pa]
        T_v (Float_0D): The vegetation temperature [degK]
        T_g (Float_0D): The ground temperature [degK]
        T_a (Float_0D): The atmospheric temperature [degK]
        u_a (Float_0D): The wind speed at the reference height of the atmosphere [m/s]
        q_a (Float_0D): The specific humidity at the reference height [kg kg-1]
        q_g (Float_0D): The specific humidity at the ground [kg kg-1]
        L (Float_0D):  The leaf area index [m2 m-2]
        S (Float_0D): The stem area index [m2 m-2]
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]

    Returns:
        Float_0D: The difference between the initial and updated Obukhov length.
    """
    (
        L_update,
        gam,
        gaw,
        gvm,
        gvw,
        ggm,
        ggw,
        q_v_sat,
        T_s,
        q_s,
    ) = perform_most_dual_source(
        L_guess=L_guess,
        pres=pres,
        T_v=T_v,
        T_g=T_g,
        T_a=T_a,
        u_a=u_a,
        q_a=q_a,
        q_g=q_g,
        L=L,
        S=S,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    dL = L_update - L_guess

    return dL


# TODO: probably incorrect
def calculate_initial_guess_L(
    T_a: Float_0D,
    T_s: Float_0D,
    q_a: Float_0D,
    q_s: Float_0D,
    u_a: Float_0D,
    z_a: Float_0D,
    d: Float_0D,
    z0m: Float_0D,
):
    """Calculate the initial guess of the Obukov length.

    Args:
        T_a (Float_0D): The air temperature [degK]
        T_s (Float_0D): The surface temperature [degK]
        q_a (Float_0D): The air specific humidity [kg kg-1]
        q_s (Float_0D): The surface specific humidity [kg kg-1]
        u_a (Float_0D): The air wind speed [m/s]
        z_a (Float_0D): The
        d (Float_0D): _description_
        z0m (Float_0D): _description_

    Returns:
        _type_: _description_
    """
    Tv_a = T_a * (1 + 0.608 * q_a)
    Tv_s = T_s * (1 + 0.608 * q_s)

    # Calculate the bulk Richardson number
    Rb = g * (z_a - d) / (u_a**2) * (Tv_a - Tv_s) / Tv_a

    # Calculate the dimensionless parameter accounting for the
    # effect of buoyancy in the Monin-Obukhov similarity theory
    if Rb >= 0:
        ζ = Rb * jnp.log((z_a - d) / z0m) / (1 - 5 * jnp.min(jnp.array([Rb, 0.19])))
    else:
        ζ = Rb * jnp.log((z_a - d) / z0m)

    # Calculate the initial guess on the length
    L_guess = (z_a - d) / ζ

    return L_guess


def perform_most_dual_source(
    L_guess: Float_0D,
    pres: Float_0D,
    T_v: Float_0D,
    T_g: Float_0D,
    T_a: Float_0D,
    u_a: Float_0D,
    q_a: Float_0D,
    q_g: Float_0D,
    L: Float_0D,
    S: Float_0D,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
) -> Tuple:
    """Perform the Monin Obukhov self-similarity calculation based on the environmental
       states of a dual source model.

    Args:
        L_guess (Float_0D): The initial guess of the Obukhov length [m]
        pres (Float_0D): The atmospheric pressure [Pa]
        T_v (Float_0D): The vegetation temperature [degK]
        T_g (Float_0D): The ground temperature [degK]
        T_a (Float_0D): The atmospheric temperature [degK]
        u_a (Float_0D): The wind speed at the reference height of the atmosphere [m/s]
        q_a (Float_0D): The specific humidity at the reference height [kg kg-1]
        q_g (Float_0D): The specific humidity at the ground [kg kg-1]
        L (Float_0D):  The leaf area index [m2 m-2]
        S (Float_0D): The stem area index [m2 m-2]
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]

    Returns:
        Tuple: The updated Obukhov length and other estimates
    """

    # Monin-Ob similarity theory (MOST)
    ψm_a = calculate_ψm_most(ζ=(z_a - d) / L_guess)
    # jax.debug.print("ψm_a: {}", jnp.array([ψm_a, L_guess, z_a, d]))
    ψm_s = calculate_ψm_most(ζ=z0m / L_guess)
    # jax.debug.print("ψm_s: {}", jnp.array([ψm_s, L_guess, z0m]))
    ψc_a = calculate_ψc_most(ζ=(z_a - d) / L_guess)
    ψc_s = calculate_ψc_most(ζ=z0c / L_guess)
    ustar = calculate_ustar_most(
        u1=0, u2=u_a, z1=z0m + d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a
    )

    # Calculate the conductances of heat and water vapor
    gam = calculate_scalar_conduct_surf_atmos(
        uref=u_a,
        zref=z_a,
        d=d,
        z0m=z0m,
        z0c=z0c,
        ψmref=ψm_a,
        ψms=ψm_s,
        ψcref=ψc_a,
        ψcs=ψc_s,
    )
    gaw = gam
    gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)
    gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
    ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
    ggw = calculate_conductance_ground_canopy_water_vapo(
        L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil
    )
    # jax.debug.print("Conductances: {}", jnp.array([ustar, u_a, gam, gvm, ggm, gaw, gvw, ggw, L, S]))  # noqa: E501
    # jax.debug.print("gvm: {}", jnp.array([ustar, L, S]))  # noqa: E501
    # print(gvw, gvm)

    # Calculate the saturated specific humidity from temperature and pressure
    # a, b = 17.2693882, 35.86
    # e_v_sat = 610.78 * jnp.exp(a * (T_v - c2k) / (T_v - b)) # [Pa]
    # q_v_sat = (0.622 * e_v_sat) / (pres*1e3 - 0.378 *e_v_sat) # [kg kg-1]
    e_v_sat = esat_from_temp(T=T_v)
    q_v_sat = q_from_e_pres(pres=pres, e=e_v_sat)

    # Calculate the temperature and specific humidity of the canopy air/surface
    T_s = calculate_Ts_from_TvTgTa(Tv=T_v, Tg=T_g, Ta=T_a, gam=gam, gvm=gvm, ggm=ggm)
    q_s = calculate_qs_from_qvqgqa(
        qv_sat=q_v_sat, qg=q_g, qa=q_a, gaw=gaw, gvw=gvw, ggw=ggw
    )

    # Calculate the updated Obukhov length
    tstar = calculate_Tstar_most(
        T1=T_s, T2=T_a, z1=d + z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a
    )  # [degK]
    qstar = calculate_qstar_most(
        q1=q_s, q2=q_a, z1=d + z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a
    )  # [kg kg-1]

    Tzv = T_a * (1 + 0.608 * q_a)
    Tvstar = tstar * (1 + 0.608 * q_a) + 0.608 * T_a * qstar  # Eq(5.17) in CLM5
    L_est = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)
    # jax.debug.print("L_est: {}", jnp.array([ustar, tstar, qstar, Tzv, Tvstar, L_est]))  # noqa: E501

    return L_est, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat, T_s, q_s
