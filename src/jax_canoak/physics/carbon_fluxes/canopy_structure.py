"""
Canopy structure functions, including:
- angle()
- lai_time()

Author: Peishi Jiang
Date: 2023.06.28.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from ..energy_fluxes import g_func_diffuse
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import PI, markov


def angle(
    latitude: Float_0D,
    longitude: Float_0D,
    zone: int,
    year: int,
    day: int,
    hour: Float_0D,
) -> Tuple[Float_0D, Float_0D, Float_0D]:
    """ANGLE computes solar elevation angles,
       This subroutine is based on algorithms in
       Walraven. 1978. Solar Energy. 20: 393-397.

    Args:
        latitude (Float_0D): _description_
        longitude (Float_0D): _description_
        zone (int): _description_
        year (int): _description_
        day (int): _description_
        hour (Float_0D): _description_

    Returns:
        Tuple[Float_0D, Float_0D, Float_0D]: _description_
    """
    RADD = PI / 180.0
    lat_rad = latitude * RADD  # latitude, radians
    # long_rad = longitude*RADD # longitude, radians

    # delta_hours = delta_long * 12. / PI
    #
    # Calculate declination angle
    declin = -23.45 * PI / 180 * jnp.cos(2 * PI * (day + 10) / 365)
    # print(declin)

    # Calculate hours of day length
    # -jnp.tan(lat_rad) * jnp.tan(declin)
    # sunrise = 12 - 12*jnp.arccos(cos_hour) / PI
    # sunset  = 12 + 12*jnp.arccos(cos_hour) / PI
    # daylength = sunset - sunrise
    f = PI * (279.5 + 0.9856 * day) / 180
    # print(sunrise, sunset)

    # Calcuate equation of time in hours
    Et = (
        -104.7 * jnp.sin(f)
        + 596.2 * jnp.sin(2 * f)
        + 4.3 * jnp.sin(3 * f)
        - 12.7 * jnp.sin(4 * f)
        - 429.3 * jnp.cos(f)
        - 2.0 * jnp.cos(2 * f)
        + 19.3 * jnp.cos(3 * f)
    ) / 3600
    # print(Et*60)

    # Perform longitudinal correction
    # Lc_deg = longitude + zone * 15  # degrees from local meridian
    Lc_deg = longitude - zone * 15  # degrees from local meridian
    Lc_hr = Lc_deg * 4.0 / 60.0  # hours, 4 minutes/per degree
    T0 = 12 - Lc_hr - Et
    hour_rad = PI * (hour - T0) / 12.0  # hour angle, radians

    # Calculate sine of solar elevation ,beta
    # print(lat_rad, declin, hour_rad, hour, T0)
    sin_beta = jnp.sin(lat_rad) * jnp.sin(declin) + jnp.cos(lat_rad) * jnp.cos(
        declin
    ) * jnp.cos(hour_rad)

    # Calculate solar elevation, radians
    beta_rad = jnp.arcsin(sin_beta)

    # Calculate solar elevation, degrees
    beta_deg = beta_rad * 180 / PI

    return beta_rad, sin_beta, beta_deg


def lai_time(
    sze: int,
    lai: Float_0D,
    ht: Float_0D,
    # tsoil: Float_0D,
    # par_reflect: Float_0D, par_trans: Float_0D, par_soil_refl: Float_0D,
    # par_absorbed: Float_0D, nir_reflect: Float_0D, nir_trans: Float_0D,
    # nir_soil_refl: Float_0D, nir_absorbed: Float_0D,
    ht_midpt: Float_1D,
    lai_freq: Float_1D,
):
    jtot = sze - 2
    # Beta distribution
    # f(x) = x^(p-1) (1-x)^(q-1) / B(v,w)
    # B(v,w) = int from 0 to 1 x^(p-1) (1-x)^(q-1) dx
    # p = mean{[mean(1-mean)/var]-1}
    # q =(1-mean){[mean(1-mean)/var]-1}

    # Height at the midpoint
    # ht_midpt = ht_midpt_scaled / ht
    ht_midpt = ht_midpt / ht
    TF = jnp.sum(lai_freq)
    MU1 = jnp.sum(ht_midpt * lai_freq)
    MU2 = jnp.sum(ht_midpt * ht_midpt * lai_freq)

    # Normalize MU by LAI
    MU1, MU2 = MU1 / TF, MU2 / TF

    # Compute Beta parameters
    P_beta = MU1 * (MU1 - MU2) / (MU2 - MU1 * MU1)
    Q_beta = (1.0 - MU1) * (MU1 - MU2) / (MU2 - MU1 * MU1)
    P_beta -= 1.0
    Q_beta -= 1.0

    # integrate Beta function, with Simpson's Approx.
    # The boundary conditions are level 1 is height of ground
    # and level jtot+1 is height of canopy.  Layer 1 is between
    # height levels 1 and 2.  Layer jtot is between levels
    # jtot and jtot+1
    # Thickness of layer
    dx = 1.0 / jtot
    DX2, DX4 = dx / 2.0, dx / 4.0
    X = DX4

    F2 = (jnp.power(X, P_beta)) * jnp.power((1.0 - X), Q_beta)
    X += DX4
    F3 = jnp.power(X, P_beta) * jnp.power((1.0 - X), Q_beta)

    # start integration at lowest boundary
    beta_fnc, integr_beta = jnp.zeros(sze), 0
    beta_fnc = beta_fnc.at[0].set(DX4 * (4.0 * F2 + F3) / 3.0)
    integr_beta += beta_fnc[0]
    JM1 = jtot - 1

    def update_beta(c, i):
        integr_beta, X, F3 = c
        F1 = F3
        X += DX2
        F2 = jnp.power(X, P_beta) * jnp.power((1.0 - X), Q_beta)
        X += DX2
        F3 = jnp.power(X, P_beta) * jnp.power((1.0 - X), Q_beta)
        beta_fnc_each = DX2 * (F1 + 4.0 * F2 + F3) / 3.0
        integr_beta += beta_fnc_each
        c_new = [integr_beta, X, F3]
        return c_new, beta_fnc_each

    carry, beta_fnc_update = jax.lax.scan(
        update_beta, [integr_beta, X, F3], jnp.arange(1, JM1)
    )
    integr_beta, X, F3 = carry
    beta_fnc = beta_fnc.at[1:JM1].set(beta_fnc_update)
    F1 = F3
    X += DX4
    F2 = jnp.power(X, P_beta) * jnp.power((1.0 - X), Q_beta)

    # compute integrand at highest boundary
    beta_fnc = beta_fnc.at[jtot - 1].set(DX4 * (F1 + 4.0 * F2) / 3.0)
    integr_beta += beta_fnc[jtot - 1]

    # lai_z IS THE LEAF AREA AS A FUNCTION OF Z
    # beta_fnc is the pdf for the interval dx
    # lai_z = jnp.zeros(sze)
    # lai_z = lai_z.at[0].set(beta_fnc[0]*lai/integr_beta)
    lai_z = beta_fnc * lai / integr_beta
    # delz = ht / jtot
    # cum_ht = delz * jtot
    # cum_lai = jnp.sum(lai_z)
    # dLAIdz = lai_z

    Gfunc_sky = g_func_diffuse(lai_z)

    # compute the probability of diffuse radiation penetration through the hemisphere.
    # This computation is not affected by penubra
    # since we are dealing only with diffuse radiation from a sky
    # sector.
    # The probability of beam penetration is computed with a
    # Markov distribution.
    exxpdir = jnp.zeros(sze)

    def update_exxpdir(c, i):
        XX, AA, DA = 0.0, 0.087, 0.1745
        dff_Markov = lai_z[i] * markov

        def calculate_xx(c2, j):
            AA, XX = c2
            cos_AA, sin_AA = jnp.cos(AA), jnp.sin(AA)
            exp_diffuse = jnp.exp(-dff_Markov * Gfunc_sky[i, j] / cos_AA)
            XX += cos_AA * sin_AA * exp_diffuse
            AA += DA
            c2_new = [AA, XX]
            return c2_new, c2_new

        c2, _ = jax.lax.scan(calculate_xx, [AA, XX], jnp.arange(9))
        _, XX = c2
        # jax.debug.print("{x}, {y}", x=XX, y=DA)
        exxpdir_each = 2.0 * XX * DA
        return c, exxpdir_each

    _, exxpdir_update = jax.lax.scan(update_exxpdir, None, jnp.arange(jtot))
    exxpdir_update = jnp.clip(exxpdir_update, a_max=0.9999)
    # jax.debug.print("exxpdir_update: {x}", x=exxpdir_update)
    exxpdir = exxpdir.at[:jtot].set(exxpdir_update)

    return exxpdir, lai_z, Gfunc_sky


# def gammaf(x: Float_0D) -> Float_0D:
#     """Gamma function.

#     Args:
#         x (Float_0D): _description_

#     Returns:
#         Float_0D: _description_
#     """
#     gam = (
#         (1.0 / (12.0 * x))
#         + (1.0 / (288.0 * x * x))
#         - (139.0 / (51840.0 * jnp.power(x, 3.0)))
#     )
#     gam = gam + 1.0

#     # x has to be positive !
#     y = jax.lax.cond(
#         x > 0,
#         lambda x: jnp.sqrt(2.0 * jnp.pi / x) * jnp.power(x, x) * jnp.exp(-x) * gam,
#         lambda x: 0.0,
#         x,
#     )

#     return y

# def freq(lflai: Float_0D) -> Float_0D:
#     """Use the beta distribution to compute the probability frequency distribution
#        for a known mean leaf inclication angle starting from the top of the canopy,
#        where llai=0 (after Goel and Strebel (1984)).

#     Args:
#         lflai (Float_0D): _description_

#     Returns:
#         Float_0D: _description_
#     """
#     # spherical leaf angle
#     MEAN = 57.4
#     STD = 26
#     VAR = STD * STD + MEAN * MEAN
#     nuu = (1.0 - VAR / (90.0 * MEAN)) / (VAR / (MEAN * MEAN) - 1.0)
#     MU = nuu * ((90.0 / MEAN) - 1.0)
#     SUM = nuu + MU

#     FL1 = gammaf(SUM) / (gammaf(nuu) * gammaf(MU))
#     MU1 = MU - 1.0
#     nu1 = nuu - 1.0

#     CONS = 1.0 / 9.0

#     # COMPUTE PROBABILITY DISTRIBUTION FOR 9 ANGLE CLASSES
#     # BETWEEN 5 AND 85 DEGREES, WITH INCREMENTS OF 10 DEGREES
#     def calculate_bden(carry, i):
#         ANG = 10.0 * (i + 1) - 5.0
#         FL2 = jnp.power((1.0 - ANG / 90.0), MU1)
#         FL3 = jnp.power((ANG / 90.0), nu1)
#         y = CONS * FL1 * FL2 * FL3
#         return None, y

#     _, bdens = jax.lax.scan(calculate_bden, None, jnp.arange(9))

#     return bdens
