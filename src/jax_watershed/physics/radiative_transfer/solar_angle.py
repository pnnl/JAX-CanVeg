""" 
Calculation of solar elevation angles.

The accuracy of the implementation can be verified against NOAA solar position calculator:
https://gml.noaa.gov/grad/solcalc/azel.html

Author: Peishi Jiang
Date: 2023.03.09
"""
import jax
import jax.numpy as jnp

from ...shared_utilities.constants import PI
from ...shared_utilities.constants import RADIANS as RADD

def calculate_solar_elevation(
    latitude: float, longitude: float, 
    year: int, day: int, hour: float,
    zone: int=8., is_day_saving: bool=False
) -> float:
    """Calculation of solar elevation and zenith angle, based on:
       (1) the algorithms in Walraven. 1978. Solar Energy. 20: 393-397;
       (2) the ANGLE() subroutine of CANOAK.

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        year (int): The year.
        day (int): The day of the year.
        hour (float): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time saving period. Defaults to 0.0.

    Returns:
        float: The solar elevation.
    """

    lat_rad  = latitude*PI/180.  # latitude, radians
    # long_rad = longitude*PI/180. # longitude, radians

    std_meridian = 0.
    delta_long = (longitude - std_meridian) * PI / 180.
    # delta_hours = delta_long * 12. / PI

    # Calculate declination angle
    declin = -23.45*3.1415/180*jnp.cos(2*3.1415*(day+10)/365)
    
    # Calculate hours of day length
    cos_hour = -jnp.tan(lat_rad) * jnp.tan(declin)
    # sunrise = 12 - 12*jnp.arccos(cos_hour) / PI
    # sunset  = 12 + 12*jnp.arccos(cos_hour) / PI
    # daylength = sunset - sunrise
    f = PI * (279.5+0.9856*day)/180

    # Calcuate equation of time in hours
    Et = (-104.7*jnp.sin(f)+596.2*jnp.sin(2*f)+4.3*jnp.sin(3*f)- \
          12.7*jnp.sin(4*f)-429.3*jnp.cos(f)-2.0*jnp.cos(2*f)+19.3*jnp.cos(3*f) \
         ) / 3600
    
    # Perform longitudinal correction
    Lc_deg = longitude + zone*15  # degrees from local meridian
    Lc_hr  = Lc_deg*4./60.  # hours, 4 minutes/per degree
    T0 = 12 - Lc_hr - Et
    hour_rad = PI *(hour-T0) / 12.   # hour angle, radians

    # Calculate sine of solar elevation ,beta
    sin_beta = jnp.sin(lat_rad)*jnp.sin(declin)+jnp.cos(lat_rad)*jnp.cos(declin)*jnp.cos(hour_rad)

    # Calculate solar elevation, radians
    beta_rad = jnp.arcsin(sin_beta)

    # Calculate solar elevation, degrees
    beta_deg = beta_rad * 180 / PI

    return beta_deg


# !!! Note that the following implementation is problematic.
# and does not give the right solution.
def calculate_solar_elevation_Walraven(
    latitude: float, longitude: float, 
    year: int, day: int, hour: float,
    zone: int=8., is_day_saving: bool=False
) -> float:
    """Calculation of solar elevation and zenith angle, based on:
       (1) the algorithms in Walraven. 1978. Solar Energy. 20: 393-397;
       (2) the ANGLE() subroutine of CANOAK.

    Args:
        latitude (float): The latitude.
        longitude (float): The longitude.
        year (int): The year.
        day (int): The day of the year.
        hour (float): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time saving period. Defaults to 0.0.

    Returns:
        float: The solar elevation.
    """

    day_savings_time = jax.lax.cond(
        is_day_saving, return_zero, return_one, 1
    )

    # Calculate the time
    delyr     = year - 1980.0
    delyr4    = delyr/4.
    leap_yr   = jnp.floor(delyr4)
    time_1980 = delyr*365+leap_yr+day-1+hour/24.0

    leap_yr_4 = leap_yr * 4

    cond1 = delyr==leap_yr_4
    time_1980 = jax.lax.cond(
        cond1, minus_one, keep_as_is, time_1980
    )
    cond2 = (delyr<0.0) and (delyr != leap_yr_4)
    time_1980 = jax.lax.cond(
        cond2, minus_one, keep_as_is, time_1980
    )

    # Calculate the longitude of the sun
    theta_angle = 2 * PI * time_1980 / 365.25
    G = -.031272 - 4.53963E-7 * time_1980 + theta_angle
    EL = 4.900968 + 3.6747E-7 * time_1980 + (.033434 - 2.3E-9 * time_1980) * jnp.sin(G) + .000349 * jnp.sin(2. * G) + theta_angle;
    EPS = .40914 - 6.2149E-9 * time_1980
    sin_el = jnp.sin(EL)

    # Calcuate the right ascension and declination
    # A1 = jnp.sin(L)*jnp.cos(eps)
    # A2 = jnp.cos(eps)*jnp.tan(L)
    A1 = sin_el * jnp.cos(EPS)
    A2 = jnp.cos(EL)
    RA = jnp.arctan(A1/A2)
    cond1 = (A1>0) and (A2<=0)
    cond2 = (A1<=0) and (A2<=0)
    RA = jax.lax.cond(
        cond1, plus_pi, keep_as_is, RA
    )
    RA = jax.lax.cond(
        cond2, plus_pi, keep_as_is, RA
    )
    value = sin_el*jnp.sin(EPS)

    # TODO: When debugging, we need JAX_DEBUG_NANS=True to check
    # any nan values that occur
    cond = 1. - value**2 >=0
    declination_ang = jax.lax.cond(
        cond, calculate_ang, return_nan, value
    )
    # print(declination_ang)

    # Calculate siderial time
    two_PI = 2 * PI
    ST = 1.759335 + two_PI * (time_1980 / 365.25 - delyr) + 3.694E-7 * time_1980
    cond = ST >= two_PI
    ST = jax.lax.cond(
        cond, minus_twopi, keep_as_is, ST
    )
    S = ST - longitude * RADD + 1.0027379 * (zone - day_savings_time + hour) * 15. * RADD
    # S = ST - longitude * RADD + 1.0027379 * (zone - day_savings_time + hour) * RADD
    # print(S/RADD)
    # print(ST/RADD)
    cond = S >= two_PI
    S = jax.lax.cond(
        cond, minus_twopi, keep_as_is, S
    )
    HS = RA - S
    phi_lat_radians = latitude * RADD

    # Calculate direction consine
    SSAS = (jnp.sin(phi_lat_radians) * jnp.sin(declination_ang) + jnp.cos(phi_lat_radians) * jnp.cos(declination_ang) * jnp.cos(HS))
    cond = 1. - SSAS**2 >=0
    E_ang = jax.lax.cond(
        cond, calculate_ang, return_nan, SSAS
    )
    cond = SSAS<0
    E_ang = jax.lax.cond(
        cond, plus_pi, keep_as_is, E_ang
    )
    cond = E_ang<0
    E_ang = jax.lax.cond(
        cond, pi_divided_by_2, keep_as_is, E_ang
    )
    
    # Calculate the solar angle
    zenith = E_ang / RADD        # in degree

    # Calculate the solar elevation
    beta_deg = 90. - zenith
    beta_rad = beta_deg * RADD # in radians
    # elev_ang_deg = 90. - zenith
    # beta_rad = elev_ang_deg * RADD
    # sine_beta = jnp.sin(beta_rad)
    # cos_zenith = jnp.cos(E_ang)
    # beta_deg = beta_rad / RADD

    return beta_deg
    # return beta_deg, beta_rad


def calculate_ang(x):
    return jnp.arctan(x/jnp.sqrt(1.-x**2))

def return_nan(x):
    return jnp.nan

def return_one(x):
    return 1.

def return_zero(x):
    return 0.

def pi_divided_by_2(x):
    return PI / 2.

def plus_pi(x):
    return x + PI

def minus_twopi(x):
    return x - PI*2

def minus_one(x):
    return x - 1

def keep_as_is(x):
    return x