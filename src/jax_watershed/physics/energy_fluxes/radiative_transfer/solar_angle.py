""" 
Calculation of solar elevation angles.

The accuracy of the implementation can be verified against NOAA solar position calculator:
https://gml.noaa.gov/grad/solcalc/azel.html
https://gml.noaa.gov/grad/solcalc/

Author: Peishi Jiang
Date: 2023.03.09
"""  # noqa: E501
import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D

from ....shared_utilities.constants import PI, TWOPI
from ....shared_utilities.constants import RADIANS as RADD


def calculate_solar_elevation(
    latitude: Float_0D,
    longitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int = 8,
    is_day_saving: bool = False,
) -> Float_0D:
    """Calculation of solar elevation and zenith angle, based on:
       (1) the algorithms in Walraven. 1978. Solar Energy. 20: 393-397;
       (2) the ANGLE() subroutine of CANOAK.

    Args:
        latitude (Float_0D): The latitude [degree].
        longitude (Float_0D): The longitude [degree].
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time
                                        saving period. Defaults to 0.0.

    Returns:
        Float_0D: The solar elevation [degree].
    """

    lat_rad = latitude * RADD  # latitude, radians
    # long_rad = longitude*RADD # longitude, radians

    # delta_hours = delta_long * 12. / PI

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
    Lc_deg = longitude + zone * 15  # degrees from local meridian
    Lc_hr = Lc_deg * 4.0 / 60.0  # hours, 4 minutes/per degree
    T0 = 12 - Lc_hr - Et
    hour_rad = PI * (hour - T0) / 12.0  # hour angle, radians

    # Calculate sine of solar elevation ,beta
    sin_beta = jnp.sin(lat_rad) * jnp.sin(declin) + jnp.cos(lat_rad) * jnp.cos(
        declin
    ) * jnp.cos(hour_rad)

    # Calculate solar elevation, radians
    beta_rad = jnp.arcsin(sin_beta)

    # Calculate solar elevation, degrees
    beta_deg = beta_rad * 180 / PI

    return beta_deg


# !!! Note that the following implementation is problematic.
# and does not give the right solution.
def calculate_solar_elevation_Walraven(
    latitude: Float_0D,
    longitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int = 8,
    is_day_saving: bool = False,
) -> Float_0D:
    """Calculation of solar elevation and zenith angle, based on:
       (1) the algorithms in Walraven. 1978. Solar Energy. 20: 393-397.

    Args:
        latitude (Float_0D): The latitude.
        longitude (Float_0D): The longitude.
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time
                                        saving period. Defaults to 0.0.

    Returns:
        Float_0D: The solar elevation.
    """
    day_savings_time = jax.lax.cond(is_day_saving, return_zero, return_one, 1)

    delyr = year - 1980.0
    delyr4 = delyr / 4.0
    leap_yr = jnp.floor(delyr4)
    time_1980 = delyr * 365 + leap_yr + day - 1 + hour / 24.0
    leap_yr_4 = leap_yr * 4

    cond1 = delyr == leap_yr_4
    time_1980 = jax.lax.cond(cond1, minus_one, keep_as_is, time_1980)
    cond2 = (delyr < 0.0) and (delyr != leap_yr_4)
    time_1980 = jax.lax.cond(cond2, minus_one, keep_as_is, time_1980)

    theta = (360.0 * time_1980 / 365.25) * RADD
    g = -0.031272 - 4.53963e-7 * time_1980 + theta
    el = (
        4.900968
        + 3.6747e-7 * time_1980
        + (0.033434 - 2.3e-9 * time_1980) * jnp.sin(g)
        + 0.000349 * jnp.sin(2.0 * g)
        + theta
    )
    eps = 0.40914 - 6.2149e-9 * time_1980
    sel = jnp.sin(el)
    A1 = sel * jnp.cos(eps)
    A2 = jnp.cos(el)
    RA = jnp.arctan2(A1, A2)
    cond = RA < 0
    RA = jax.lax.cond(cond1, plus_pi, keep_as_is, RA)
    decl = jnp.arcsin(sel * jnp.sin(eps))
    # print(decl)
    st = 1.759335 + TWOPI * (time_1980 / 365.25 - delyr) + 3.694e-7 * time_1980
    cond = st >= TWOPI
    st = jax.lax.cond(cond, minus_twopi, keep_as_is, st)
    s = (
        st
        - longitude * RADD
        + 1.0027379 * (zone - day_savings_time + hour) * 15.0 * RADD
    )
    cond = s >= TWOPI
    s = jax.lax.cond(cond, minus_twopi, keep_as_is, s)
    h = RA - s
    phi = latitude * RADD
    # print(RA, st, s, h, phi)

    # Calculation of solar angle (a) and solar elevation (e)
    e = jnp.arcsin(
        jnp.sin(phi) * jnp.sin(decl) + jnp.cos(phi) * jnp.cos(decl) * jnp.cos(h)
    )
    # a = jnp.arcsin(
    #     jnp.cos(decl)*jnp.sin(h) / jnp.cos(e)
    # ) / RADD
    # print(a, e/RADD)

    # cond1 = jnp.sin(e) >= jnp.sin(decl)/jnp.sin(phi)
    # cond2 = a < 0.
    e = e / RADD
    return e


# !!! Note that the following implementation is problematic.
# and does not give the right solution.
def calculate_solar_elevation_Walraven_CANOAK(
    latitude: Float_0D,
    longitude: Float_0D,
    year: int,
    day: int,
    hour: Float_0D,
    zone: int = 8,
    is_day_saving: bool = False,
) -> Float_0D:
    """Calculation of solar elevation and zenith angle, based on:
       (1) the algorithms in Walraven. 1978. Solar Energy. 20: 393-397;
       (2) the ANGLE() subroutine of CANOAK.

    Args:
        latitude (Float_0D): The latitude.
        longitude (Float_0D): The longitude.
        year (int): The year.
        day (int): The day of the year.
        hour (Float_0D): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time
                                        saving period. Defaults to 0.0.

    Returns:
        Float_0D: The solar elevation.
    """

    day_savings_time = jax.lax.cond(is_day_saving, return_zero, return_one, 1)

    # Calculate the time
    delyr = year - 1980.0
    delyr4 = delyr / 4.0
    leap_yr = jnp.floor(delyr4)
    time_1980 = delyr * 365 + leap_yr + day - 1 + hour / 24.0

    leap_yr_4 = leap_yr * 4

    cond1 = delyr == leap_yr_4
    time_1980 = jax.lax.cond(cond1, minus_one, keep_as_is, time_1980)
    cond2 = (delyr < 0.0) and (delyr != leap_yr_4)
    time_1980 = jax.lax.cond(cond2, minus_one, keep_as_is, time_1980)

    # Calculate the longitude of the sun
    theta_angle = 2 * PI * time_1980 / 365.25
    G = -0.031272 - 4.53963e-7 * time_1980 + theta_angle
    EL = (
        4.900968
        + 3.6747e-7 * time_1980
        + (0.033434 - 2.3e-9 * time_1980) * jnp.sin(G)
        + 0.000349 * jnp.sin(2.0 * G)
        + theta_angle
    )
    EPS = 0.40914 - 6.2149e-9 * time_1980
    sin_el = jnp.sin(EL)

    # Calcuate the right ascension and declination
    # A1 = jnp.sin(L)*jnp.cos(eps)
    # A2 = jnp.cos(eps)*jnp.tan(L)
    A1 = sin_el * jnp.cos(EPS)
    A2 = jnp.cos(EL)
    RA = jnp.arctan(A1 / A2)
    cond1 = (A1 > 0) and (A2 <= 0)
    cond2 = (A1 <= 0) and (A2 <= 0)
    # print(EL, A1, A2, RA)
    RA = jax.lax.cond(cond1, plus_pi, keep_as_is, RA)
    RA = jax.lax.cond(cond2, plus_pi, keep_as_is, RA)
    value = sin_el * jnp.sin(EPS)

    # TODO: When debugging, we need JAX_DEBUG_NANS=True to check
    # any nan values that occur
    cond = 1.0 - value**2 >= 0
    declination_ang = jax.lax.cond(cond, calculate_ang, return_nan, value)
    # print(declination_ang)
    # print(jnp.arcsin(value))

    # Calculate siderial time
    two_PI = 2 * PI
    ST = 1.759335 + two_PI * (time_1980 / 365.25 - delyr) + 3.694e-7 * time_1980
    cond = ST >= two_PI
    ST = jax.lax.cond(cond, minus_twopi, keep_as_is, ST)
    S = (
        ST
        - longitude * RADD
        + 1.0027379 * (zone - day_savings_time + hour) * 15.0 * RADD
    )
    # S = ST - longitude * RADD + 1.0027379 * (zone - day_savings_time + hour) * RADD
    # print(S/RADD)
    # print(ST/RADD)
    cond = S >= two_PI
    S = jax.lax.cond(cond, minus_twopi, keep_as_is, S)
    HS = RA - S
    phi_lat_radians = latitude * RADD

    # Calculate direction consine
    SSAS = jnp.sin(phi_lat_radians) * jnp.sin(declination_ang) + jnp.cos(
        phi_lat_radians
    ) * jnp.cos(declination_ang) * jnp.cos(HS)
    cond = 1.0 - SSAS**2 >= 0
    E_ang = jax.lax.cond(cond, calculate_ang, return_nan, SSAS)
    cond = SSAS < 0
    E_ang = jax.lax.cond(cond, plus_pi, keep_as_is, E_ang)
    cond = E_ang < 0
    E_ang = jax.lax.cond(cond, pi_divided_by_2, keep_as_is, E_ang)

    # Calculate the solar angle
    zenith = E_ang / RADD  # in degree

    # Calculate the solar elevation
    beta_deg = 90.0 - zenith
    # elev_ang_deg = 90. - zenith
    # beta_rad = elev_ang_deg * RADD
    # sine_beta = jnp.sin(beta_rad)
    # cos_zenith = jnp.cos(E_ang)
    # beta_deg = beta_rad / RADD

    return beta_deg
    # return beta_deg, beta_rad


def calculate_ang(x):
    return jnp.arctan(x / jnp.sqrt(1.0 - x**2))


def return_nan(x):
    return jnp.nan


def return_one(x):
    return 1.0


def return_zero(x):
    return 0.0


def pi_divided_by_2(x):
    return PI / 2.0


def plus_pi(x):
    return x + PI


def minus_twopi(x):
    return x - PI * 2


def minus_one(x):
    return x - 1


def keep_as_is(x):
    return x