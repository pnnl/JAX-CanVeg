"""
Calculating the vector fields of the heat equation for solving the soil temperatures.

Author: Peishi Jiang
Date: 2023.04.04.
"""

# TODO: Add unit tests!

# import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D, Float_1D
from ....shared_utilities.constants import λ_VAP as λ

from ..surface_energy.monin_obukhov import perform_most_dual_source
from ..turbulent_fluxes import calculate_E, calculate_H
from ..radiative_transfer import calculate_ground_longwave_fluxes

# from ...water_fluxes import qsat_from_temp_pres
from ...water_fluxes import calculate_ground_specific_humidity


def Tsoil_vector_field_varyingG(
    Tsoil: Float_1D,
    κ: Float_1D,
    cv: Float_1D,
    Δz: Float_1D,
    l: Float_0D,  # noqa: E741
    S_g: Float_0D,
    L_down: Float_0D,
    pres: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2: Float_0D,
    T_g_t1: Float_0D,
    T_a_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    L: Float_0D,
    S: Float_0D,
    ε_g: Float_0D,
    ε_v: Float_0D,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
) -> Float_1D:
    """Calculate the vector field of the heat equation of the soil temperature profile.

    Args:
        Tsoil (Float_1D): The soil temperature with nsoil layers [degK]
        cv (Float_1D): The volumetric heat capacity with nsoil layers [J m-3 K-1]
        κ (Float_1D): The thermal conductivity with nsoil layers [W m-1 K-1]
        Δz (Float_1D): The thickness of the nsoil layers [m]
        l (Float_0D): The Obuhkov length [m]
        S_g (Float_0D): The incoming solar radiation on the ground [W m-2]
        L_down (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        pres (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        ρ_atm_t2 (Float_0D): The air density at the reference height at the current time step t2 [kg m-3]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2 (Float_0D): The vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        L (Float_0D): The leaf area index [m2 m-2]
        S (Float_0D): The steam area index [m2 m-2]
        ε_g (Float_0D): The ground emissivities [-]
        ε_v (Float_0D): The vegetation emissivities [-]
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]

    Returns:
        Float_1D: The vector field.
    """  # noqa: E501
    T_g_t2 = Tsoil[0]

    # Calculate the conductances and T_s/q_s
    # TODO: Update the ground specific humidity
    # q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
    # q_g_t2 = q_g_t2_sat
    q_g_t2 = calculate_ground_specific_humidity(T_g_t2, pres)
    (_, _, _, _, _, ggm, ggw, q_v_sat_t2, T_s_t2, q_s_t2,) = perform_most_dual_source(
        L_guess=l,
        pres=pres,
        T_v=T_v_t2,
        T_g=T_g_t2,
        T_a=T_a_t2,
        u_a=u_a_t2,
        q_a=q_a_t2,
        q_g=q_g_t2,
        L=L,
        S=S,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

    # Calculate the ground heat flux based on energy balance
    L_g = calculate_ground_longwave_fluxes(
        L_down=L_down,
        ε_v=ε_v,
        ε_g=ε_g,
        L=L,
        S=S,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_g_t2=T_g_t2,
    )
    H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
    E_g = calculate_E(q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw)  # [kg m-2 s-1]
    λE_g = λ * E_g
    G = S_g - L_g - H_g - λE_g
    G = -G

    # Calculate the depths of the soil interfaces
    zn = jnp.cumsum(Δz)  # shape: (nsoil)
    # Add the ground surface level, which is zero
    zn = jnp.concatenate([jnp.array([0]), zn])  # shape: (nsoil+1)
    # jax.debug.print("The depths of soil layer interfaces: ", zn)

    # Calculate the depths of the soil layer mid points
    z = zn[:-1] + Δz / 2  # shape: (nsoil)
    # jax.debug.print("The depths of the soil layer mid points: ", z)

    # Calculate the distances between the mid points of adjacent layers
    Δzn = z[1:] - z[:-1]  # shape: (nsoil-1)
    Δzn = jnp.concatenate([z[:1], Δzn])  # shape: (nsoil)
    # jax.debug.print("The distances between the mid points of adjacent layers: {}", Δzn)  # noqa: E501

    # Calculate the harmonic mean of κ between layers
    # Eq(6.8)
    κn = (
        κ[:-1]
        * κ[1:]
        * Δzn[1:]
        / (κ[:-1] * (z[1:] - zn[1:-1]) + κ[1:] * (zn[1:-1] - z[:-1]))
    )  # shape: (nsoil-1)
    κn = jnp.concatenate([jnp.array([0]), κn])  # shape: (nsoil)
    # jax.debug.print("The harmonic mean of κ between layers: ", κn)

    # Calculate the heat flux between two soil layers
    F = -κn[1:] / Δzn[1:] * (Tsoil[:-1] - Tsoil[1:])  # shape: (nsoil-1)
    # jax.debug.print("heat Fluxes: ", F)

    # Calculate the vector field for the top soil layer
    f1 = (-G + F[0]) / (cv[0] * Δz[0])
    # jax.debug.print("The first layer fluxes: {}", jnp.array([G, F[0]]))

    # Calculate the vector field for the bottom soil layer
    fN = (-F[-1] + 0) / (cv[-1] * Δz[-1])

    # Calculate the vector fields for the remaining soil layers
    fi = (-F[:-1] + F[1:]) / (cv[1:-1] * Δz[1:-1])

    return jnp.concatenate([jnp.array([f1]), fi, jnp.array([fN])])


def Tsoil_vector_field(
    # t: Float_0D, Tsoil: Float_1D, κ: Float_1D, cv: Float_1D, Δz: Float_1D, G: Float_0D
    Tsoil: Float_1D,
    κ: Float_1D,
    cv: Float_1D,
    Δz: Float_1D,
    G: Float_0D,
) -> Float_1D:
    """Calculate the vector field of the heat equation of the soil temperature profile.

    Args:
        t (Float_0D): The time step.
        Tsoil (Float_1D): The soil temperature with nsoil layers [degK]
        cv (Float_1D): The volumetric heat capacity with nsoil layers [J m-3 K-1]
        κ (Float_1D): The thermal conductivity with nsoil layers [W m-1 K-1]
        Δz (Float_1D): The thickness of the nsoil layers [m]
        G (Float_0D): The ground heat flux with positive values indicating the upward direction [W m-2]

    Returns:
        Float_1D: The vector field.
    """  # noqa: E501
    # Calculate the depths of the soil interfaces
    zn = jnp.cumsum(Δz)  # shape: (nsoil)
    # Add the ground surface level, which is zero
    zn = jnp.concatenate([jnp.array([0]), zn])  # shape: (nsoil+1)
    # jax.debug.print("The depths of soil layer interfaces: ", zn)

    # Calculate the depths of the soil layer mid points
    z = zn[:-1] + Δz / 2  # shape: (nsoil)
    # jax.debug.print("The depths of the soil layer mid points: ", z)

    # Calculate the distances between the mid points of adjacent layers
    Δzn = z[1:] - z[:-1]  # shape: (nsoil-1)
    Δzn = jnp.concatenate([z[:1], Δzn])  # shape: (nsoil)
    # jax.debug.print("The distances between the mid points of adjacent layers: {}", Δzn)  # noqa: E501

    # Calculate the harmonic mean of κ between layers
    # Eq(6.8)
    κn = (
        κ[:-1]
        * κ[1:]
        * Δzn[1:]
        / (κ[:-1] * (z[1:] - zn[1:-1]) + κ[1:] * (zn[1:-1] - z[:-1]))
    )  # shape: (nsoil-1)
    κn = jnp.concatenate([jnp.array([0]), κn])  # shape: (nsoil)
    # jax.debug.print("The harmonic mean of κ between layers: ", κn)

    # Calculate the heat flux between two soil layers
    F = -κn[1:] / Δzn[1:] * (Tsoil[:-1] - Tsoil[1:])  # shape: (nsoil-1)
    # jax.debug.print("heat Fluxes: ", F)

    # Calculate the vector field for the top soil layer
    f1 = (-G + F[0]) / (cv[0] * Δz[0])
    # jax.debug.print("The first layer fluxes: {}", jnp.array([G, F[0]]))

    # Calculate the vector field for the bottom soil layer
    fN = (-F[-1] + 0) / (cv[-1] * Δz[-1])

    # Calculate the vector fields for the remaining soil layers
    fi = (-F[:-1] + F[1:]) / (cv[1:-1] * Δz[1:-1])

    return jnp.concatenate([jnp.array([f1]), fi, jnp.array([fN])])


def calculate_Tg_from_Tsoil1(
    Tsoil1: Float_0D, G: Float_0D, Δz: Float_0D, κ: Float_0D
) -> Float_0D:
    """Calculate the ground surface temperature from the first layer of soil temperature.

    Args:
        Tsoil1 (Float_0D): The temperature of the first soil layer [degK]
        κ (Float_1D): The thermal conductivity of the first soil layer [W m-1 K-1]
        Δz (Float_1D): The distance between the ground and the center of the first layer [m]
        G (Float_0D): The ground heat flux with positive values indicating the upward direction [W m-2]

    Returns:
        Float_0D: The ground surface temperature [degK]
    """  # noqa: E501
    Tg = -G * Δz / κ + Tsoil1

    return Tg
