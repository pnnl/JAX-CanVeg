"""
Calculating the vector fields of the heat equation for solving the soil temperatures.

Author: Peishi Jiang
Date: 2023.04.04.
"""

# TODO: Add unit tests!

# import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D, Float_1D


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

    # Calculate the vector field for the bottom soil layer
    fN = (-F[-1] + 0) / (cv[-1] * Δz[-1])

    # Calculate the vector fields for the remaining soil layers
    fi = (-F[:-1] + F[1:]) / (cv[1:-1] * Δz[1:-1])

    return jnp.concatenate([jnp.array([f1]), fi, jnp.array([fN])])
