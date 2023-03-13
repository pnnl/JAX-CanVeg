"""
Calculation of canopy radiative transfer based on:
(1) the two-stream appximation of Dickinson (1983) and Sellers (1985) and Bonan (1996);
(2) the implementation in CLM5 documentation

Author: Peishi Jiang
Date: 2023.3.12
"""

import jax
import jax.numpy as jnp

from ...shared_utilities.types import Float_1D

from ...shared_utilities.constants import RADIANS as RADD
from ...shared_utilities.constants import χl_clm5, α_leaf_clm5, α_stem_clm5
from ...shared_utilities.constants import τ_leaf_clm5, τ_stem_clm5
from ...shared_utilities.constants import ω_snow_clm5, β_snow_clm5, β0_snow_clm5

def calculate_canopy_fluxes_per_unit_incident(
    solar_elev_angle: float, α_g_db: float, α_g_dif: float,
    L: float, S: float, f_cansno: float,
    rad_type: int, pft_ind: str
) -> Float_1D:
    """Calculate the canopy radiative fluxes per unit incident radiation based on Section 3.1 in CLM5.

    Args:
        solar_elev_angle (float): Solar elevation angle [degree]
        α_g_db (float): The overall direct beam ground albedo [-]
        α_g_dif (float): The overall diffuse ground albedo [-]
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]
        f_cansno (float): The canopy snow-covered fraction
        rad_type (str): The radiation type or band, either PAR or NIR
        pft_ind (int): The index of plant functional type based on the pft_clm5

    Returns:
        Float_1D: Different components of canopy radiative fluxes per unit incident radiation.
    """
    # Get the parameters from CLM5 database
    χl      = χl_clm5[pft_ind]
    # χl      = -1
    α_leaf  = α_leaf_clm5[rad_type][pft_ind]
    α_stem  = α_stem_clm5[rad_type][pft_ind]
    τ_leaf  = τ_leaf_clm5[rad_type][pft_ind]
    τ_stem  = τ_stem_clm5[rad_type][pft_ind]
    ω_snow  = ω_snow_clm5[rad_type]
    β_snow  = β_snow_clm5[rad_type]
    β0_snow = β0_snow_clm5[rad_type]

    # Calculate the cosine of the solar zenith angle of the incident beam
    # , or the sine of the solar elevation angle
    solar_elev_rad = solar_elev_angle * RADD
    μ              = jnp.sin(solar_elev_rad)

    # Calculate G, K, and μ_bar
    # TODO: the calculation of φ1 and φ2 might be incorrect
    #       the following is only applicable to -0.4 <= χl <= 0.6
    φ1    = 0.5 - 0.633*χl - 0.33*χl**2
    φ2    = 0.877 * (1-2*φ1)
    K     = G(φ1, φ2, μ) / μ
    μ_bar = 1./φ2 * (1 - φ1/φ2*jnp.log((φ1+φ2)/φ1))  # Eq(3.4) in CLM5

    # Calculate ω, β, β0 
    # Note that τ_leaf, τ_stem, α_leaf, α_stem, and χl are obtained from Table(3.1) in CLM5, and
    #           ω_snow, β_snow, and β0_snow are obtained from Table(3.2) in CLM5
    #           based on the plant functional type and PAR(VIS) or NIR radiation type.
    w_leaf    = L / (L+S)
    w_stem    = S / (L+S)
    τ         = τ_leaf * w_leaf + τ_stem * w_stem  # Eq(3.12) in CLM5
    α         = α_leaf * w_leaf + α_stem * w_stem  # Eq(3.11) in CLM5
    ω_veg     = α + τ
    # min_value = jnp.min(jnp.array([μ*φ2+G(φ1,φ2,μ),1e-6]))
    # Note: there is a typo in Eq(3.16), 
    #       where it should be max(μ*φ2+G(φ1,φ2,μ),1e-6) to avoid the impact of small values
    min_value = jnp.max(jnp.array([μ*φ2+G(φ1,φ2,μ),1e-6]))
    as_μ      = ω_veg/2. * G(φ1,φ2,μ)/min_value * ( 1. - μ*φ1/min_value * jnp.log((μ*φ1+min_value)/(μ*φ1)) )   # Eq(3.16) in CLM5
    β_veg    = 0.5 * ( α + τ + (α-τ) * ((1 + χl)/2)**2 ) / ω_veg  # Eq(3.13) in CLM5
    β0_veg   = (1 + μ_bar * K) / (ω_veg*μ_bar*K) * as_μ   # Eq(3.15) in CLM5
    ω         = ω_veg * (1 - f_cansno) + ω_snow * f_cansno  # Eq(3.5) in CLM5
    ωβ        = ω_veg * β_veg * (1 - f_cansno) + ω_snow * β_snow * f_cansno  # Eq(3.6) in CLM5
    ωβ0       = ω_veg * β0_veg * (1 - f_cansno) + ω_snow * β0_snow * f_cansno  # Eq(3.7) in CLM5
    β, β0     = ωβ / ω, ωβ0 / ω

    # Calculate the remaining auxiliary variables
    # TODO: which albedo should be used? α_g_db or α_g_dif?
    h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, \
        σ, h, s1, s2 = calculate_auxiliary_variables(ω, β, β0, μ_bar, α_g_db, K, L, S)
    
    # Calculate the upward radiations per unit incident flux
    I_up_db  = h1 / σ + h2 + h3   # direct beam Eq(3.17)
    I_up_dif = h7 + h8          # diffuse Eq(3.18)

    # Calculate the downward radiations per unit incident flux
    I_down_db  = h4/σ * jnp.exp(-K*(L+S)) + h5*s1 + h6/s1  # direct beam Eq(3.19)
    I_down_dif = h9 * s1 + h10 / s1   # diffuse Eq(3.20)

    # Calculate the direct beam flux transmitted through the canopy
    I_down_trans_can = jnp.exp(-K*(L+S))

    # Calculate the radiations absorbed by the vegetation
    I_can_db  = 1 - I_up_db - (1-α_g_dif)*I_down_db - (1-α_g_db)*I_down_trans_can  # direct beam Eq(3.21)
    I_can_dif = 1 - I_up_dif - (1-α_g_dif)*I_down_dif    # diffuse Eq(3.22)

    # Calculate the absorption of direct beam radiation by sunlit and shaded leaves
    # Eqs(3.23)-(3.26)
    a1           = h1/σ*(1-s2**2)/(2*K) + h2*(1-s2*s1)/(K+h) + h3*(1-s2/s1)/(K-h)
    a2           = h4/σ*(1-s2**2)/(2*K) + h5*(1-s2*s1)/(K+h) + h6*(1-s2/s1)/(K-h)
    I_can_sun_db = (1-ω) * (1 - s2 + 1/μ_bar*(a1+a2))
    I_can_sha_db = I_can_db - I_can_sun_db

    # Calculate the absorption of diffuse radiation by sunlit and shaded leaves
    # Eqs(3.27)-(3.30)
    a1            = h7*(1-s2*s1)/(K+h) + h8*(1-s2/s1)/(K-h)
    a2            = h9*(1-s2*s1)/(K+h) + h10*(1-s2/s1)/(K-h)
    I_can_sun_dif = (1-ω)/μ_bar * (a1+a2)
    I_can_sha_dif = I_can_dif - I_can_sun_dif

    # Note that the sum of the them equal to 2.
    return jnp.array([
        I_up_db, I_up_dif, I_down_db, I_down_dif, I_down_trans_can,
        I_can_sun_db, I_can_sha_db, I_can_sun_dif, I_can_sha_dif
    ])


def G(φ1, φ2, μ):
    return φ1 + φ2 * μ  # Eq(3.3) in CLM5


def calculate_auxiliary_variables(
    ω: float, β: float, β0: float,
    μ_bar: float, α_g: float, K: float,
    L: float, S: float
):
    # Based on Eqs(3.31)-(3.57) in CLM5
    b  = 1 - ω + ω * β
    c  = ω * β
    d  = ω * μ_bar * K * β0
    f  = ω * μ_bar * K * (1-β0)
    # print(d + f)
    # print(ω, μ_bar, K, β0)
    h  = jnp.sqrt(b**2 - c**2) / μ_bar
    σ  = (μ_bar * K) ** 2 + c ** 2 - b ** 2
    # print(c, b, ω, μ_bar * K)
    u1 = b - c  / α_g
    u2 = b - c * α_g
    u3 = f + c * α_g
    s1 = jnp.exp( - jnp.min(jnp.array([h*(L+S), 40])) )
    s2 = jnp.exp( - jnp.min(jnp.array([K*(L+S), 40])) )
    p1 = b + μ_bar * h
    p2 = b - μ_bar * h
    p3 = b + μ_bar * K
    p4 = b - μ_bar * K
    d1 = p1 * (u1 - μ_bar*h) / s1 - p2 * (u1 + μ_bar*h) * s1
    d2 = (u2 + μ_bar*h) / s1 - (u2 - μ_bar*h) * s1
    h1 = - d * p4 - c*f
    h2 = 1. / d1 * (
        (d - h1/σ * p3) * (u1 - μ_bar*h) / s1 - \
        p2 * (d - c - h1/σ * (u1+μ_bar*K)) * s2
    )
    h3 = - 1. / d1 * (
        (d - h1/σ * p3) * (u1 + μ_bar*h) * s1 - \
        p1 * (d - c - h1/σ * (u1+μ_bar*K)) * s2
    )
    h4 = - f * p3 - c * d
    h5 = - 1. / d2 * (
        h4 * (u2 + μ_bar*h) / (s1*σ) + \
        (u3 - h4/σ * (u2-μ_bar*K)) * s2
    )
    h6 = 1. / d2 * (
        h4 / σ * (u2 - μ_bar*h) * s1 + \
        (u3 - h4/σ * (u2-μ_bar*K)) * s2
    )
    # print(h5, h6)
    h7  = c*(u1 - μ_bar*h) / (d1*s1)
    h8  = -c*(u1 + μ_bar*h) * s1 / d1
    h9  = (u2 + μ_bar*h) / (d2*s1)
    h10 = - s1 * (u2 - μ_bar*h) / d2

    # print(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10)
    
    return h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, σ, h, s1, s2