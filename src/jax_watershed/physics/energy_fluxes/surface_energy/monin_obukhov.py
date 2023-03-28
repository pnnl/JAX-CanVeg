"""
Calculate the Momin Obukhov length using the temperature/specific humidity in the dual source model.

Author: Peishi Jiang
Date: 2023.03.28.
"""

import jax
import jax.numpy as jnp

from ..turbulent_fluxes import calculate_ψc_most, calculate_ψm_most, calculate_L_most
from ..turbulent_fluxes import calculate_ustar_most, calculate_qstar_most, calculate_Tstar_most
from ..turbulent_fluxes import calculate_scalar_conduct_surf_atmos, calculate_total_conductance_leaf_boundary, calculate_total_conductance_leaf_water_vapor
from ..turbulent_fluxes import calculate_conductance_ground_canopy, calculate_conductance_ground_canopy_water_vapo
from .surface_state import calculate_qs_from_qvqgqa, calculate_Ts_from_TvTgTa
from ...water_fluxes import q_from_e_pres, esat_from_temp, qsat_from_temp_pres


def func_most_dual_source(
    L_guess:float, pres:float, 
    T_v:float, T_g:float, T_a:float, u_a:float, q_a:float, q_g:float, L:float, S:float,
    z_a:float, z0m:float, z0c:float, d:float, gstomatal:float, gsoil:float
) -> float:
    """Residual function of the Monin Obukhov self-similarity calculation based on the environmental states of a dual source model.

    Args:
        L_guess (float): The initial guess of the Obukhov length [m]
        pres (float): The atmospheric pressure [Pa]
        T_v (float): The vegetation temperature [degK]
        T_g (float): The ground temperature [degK]
        T_a (float): The atmospheric temperature [degK]
        u_a (float): The wind speed at the reference height of the atmosphere [m/s]
        q_a (float): The specific humidity at the reference height [kg kg-1]
        q_g (float): The specific humidity at the ground [kg kg-1]
        L (float):  The leaf area index [m2 m-2]
        S (float): The stem area index [m2 m-2]
        z_a (float): The reference height of the atmosphere [m]
        z0m (float): The roughness length for momentum [m]
        z0c (float): The roughness length for scalars [m]
        d (float): The displacement height [m]
        gstomatal (float): The stomatal conductance [m s-1]
        gsoil (float): The soil conductance [m s-1]

    Returns:
        float: The difference between the initial and updated Obukhov length.
    """
    L_update, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat, T_s, q_s  = perform_most_dual_source(
        L_guess=L_guess, pres=pres, 
        T_v=T_v, T_g=T_g, T_a=T_a, u_a=u_a, q_a=q_a, q_g=q_g, L=L, S=S,
        z_a=z_a, z0m=z0m, z0c=z0c, d=d, gstomatal=gstomatal, gsoil=gsoil
    )

    dL = L_update - L_guess

    return dL


def perform_most_dual_source(
    L_guess:float, pres:float, 
    T_v:float, T_g:float, T_a:float, u_a:float, q_a:float, q_g:float, L:float, S:float,
    z_a:float, z0m:float, z0c:float, d:float, gstomatal:float, gsoil:float) -> float:
    """Perform the Monin Obukhov self-similarity calculation based on the environmental states of a dual source model.

    Args:
        L_guess (float): The initial guess of the Obukhov length [m]
        pres (float): The atmospheric pressure [Pa]
        T_v (float): The vegetation temperature [degK]
        T_g (float): The ground temperature [degK]
        T_a (float): The atmospheric temperature [degK]
        u_a (float): The wind speed at the reference height of the atmosphere [m/s]
        q_a (float): The specific humidity at the reference height [kg kg-1]
        q_g (float): The specific humidity at the ground [kg kg-1]
        L (float):  The leaf area index [m2 m-2]
        S (float): The stem area index [m2 m-2]
        z_a (float): The reference height of the atmosphere [m]
        z0m (float): The roughness length for momentum [m]
        z0c (float): The roughness length for scalars [m]
        d (float): The displacement height [m]
        gstomatal (float): The stomatal conductance [m s-1]
        gsoil (float): The soil conductance [m s-1]

    Returns:
        float: The updated Obukhov length
    """

    # Monin-Ob similarity theory (MOST)
    ψm_a  = calculate_ψm_most(ζ=z_a-d / L_guess)
    ψm_s  = calculate_ψm_most(ζ=z0m / L_guess)
    ψc_a  = calculate_ψc_most(ζ=z_a-d / L_guess)
    ψc_s  = calculate_ψc_most(ζ=z0c / L_guess)
    ustar = calculate_ustar_most(u1=0, u2=u_a, z1=z0m+d, z2=z_a, d=d, ψm1=ψm_s, ψm2=ψm_a)

    # Calculate the conductances of heat and water vapor
    # gamom = calculate_momentum_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, ψmref=ψm_a, ψms=ψm_s)
    gam = calculate_scalar_conduct_surf_atmos(uref=u_a, zref=z_a, d=d, z0m=z0m, z0c=z0c, ψmref=ψm_a, ψms=ψm_s, ψcref=ψc_a, ψcs=ψc_s)
    gaw = gam
    gvm = calculate_total_conductance_leaf_boundary(ustar=ustar, L=L, S=S)    
    gvw = calculate_total_conductance_leaf_water_vapor(gh=gvm, gs=gstomatal)
    ggm = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)
    ggw = calculate_conductance_ground_canopy_water_vapo(L=L, S=S, ustar=ustar, z0m=z0m, gsoil=gsoil)
    # print(gvw, gvm)

    # Calculate the saturated specific humidity from temperature and pressure
    # a, b = 17.2693882, 35.86
    # e_v_sat = 610.78 * jnp.exp(a * (T_v - c2k) / (T_v - b)) # [Pa]
    # q_v_sat = (0.622 * e_v_sat) / (pres*1e3 - 0.378 *e_v_sat) # [kg kg-1]
    e_v_sat = esat_from_temp(T=T_v)
    q_v_sat = q_from_e_pres(pres=pres, e=e_v_sat)

    # Calculate the temperature and specific humidity of the canopy air/surface
    T_s = calculate_Ts_from_TvTgTa(Tv=T_v, Tg=T_g, Ta=T_a, gam=gam, gvm=gvm, ggm=ggm)
    q_s = calculate_qs_from_qvqgqa(qv_sat=q_v_sat, qg=q_g, qa=q_a, gaw=gaw, gvw=gvw, ggw=ggw)

    # Calculate the updated Obukhov length
    tstar = calculate_Tstar_most(T1=T_s, T2=T_a, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [degK]
    qstar = calculate_qstar_most(q1=q_s, q2=q_a, z1=d+z0c, z2=z_a, d=d, ψc1=ψc_s, ψc2=ψc_a) # [kg kg-1]
    
    Tzv = T_a * (1 + 0.608 * q_a)
    Tvstar = tstar * (1 + 0.608 * q_a) + 0.608 * T_a * qstar # Eq(5.17) in CLM5
    L_est = calculate_L_most(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)

    return L_est, gam, gaw, gvm, gvw, ggm, ggw, q_v_sat, T_s, q_s

