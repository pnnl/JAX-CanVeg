"""
This is the main function of calcuating the radiative transfer fluxes, including
- the canopy radiative transfer fluxes
- the longwave radiation fluxes
- the fluxes from the ground surface.

Author: Peishi Jiang
Date: 2023.03.13
"""

from typing import Tuple

from .solar_angle import calculate_solar_elevation
from .solar_radiation_partition import partition_solar_radiation
from .albedo_emissivity import calculate_ground_albedos, calculate_ground_vegetation_emissivity
from .canopy_radiative_transfer import calculate_canopy_fluxes_per_unit_incident
from .radiative_fluxes import calculate_net_radiation_at_the_surface
from .radiative_fluxes import calculate_solar_fluxes, calculate_longwave_fluxes
from .radiative_fluxes import check_solar_energy_conservation, check_longwave_energy_conservation


def main_calculate_solar_fluxes(
    solar_rad: float, pres: float, solar_elev_angle: float, 
    α_g_db_par: float, α_g_dif_par: float, α_g_db_nir: float, α_g_dif_nir: float,
    f_snow: float, f_cansno: float, L: float, S: float, pft_ind: int,
) -> Tuple:
    """A main function for calculating different solar flux components

    Args:
        solar_rad (float): The incoming solar radiation [W m2-1]
        pres (float): The atmospheric pressure [kPa]
        solar_elev_angle (float): The solar elevation [degree]
        f_cansno (float): The canopy snow-covered fraction [-]
        f_snow (float): The fraction of ground covered by snow [-]
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]
        pft_ind (int): The index of plant functional type based on the pft_clm5

    Returns:
        Tuple: Different solar flux components
    """
    # Perform the solar radiation partitioning
    S_db_par, S_dif_par, S_db_nir, S_dif_nir = partition_solar_radiation(
        solar_rad=solar_rad, solar_elev_angle=solar_elev_angle, pres=pres
    )

    # Calculate the canopy radiative transfer (PAR)
    I_up_db_par, I_up_dif_par, I_down_db_par, I_down_dif_par, I_down_trans_can_par, \
    I_can_sun_db_par, I_can_sha_db_par, I_can_sun_dif_par, I_can_sha_dif_par = \
        calculate_canopy_fluxes_per_unit_incident(
            solar_elev_angle=solar_elev_angle, L=L, S=S, f_cansno=f_cansno,
            α_g_db=α_g_db_par, α_g_dif=α_g_dif_par, f_snow=f_snow, rad_type='PAR', pft_ind=pft_ind
        )

    # Calculate the canopy radiative transfer (NIR)
    I_up_db_nir, I_up_dif_nir, I_down_db_nir, I_down_dif_nir, I_down_trans_can_nir, \
    I_can_sun_db_nir, I_can_sha_db_nir, I_can_sun_dif_nir, I_can_sha_dif_nir = \
        calculate_canopy_fluxes_per_unit_incident(
            solar_elev_angle=solar_elev_angle, L=L, S=S, f_cansno=f_cansno,
            α_g_db=α_g_db_nir, α_g_dif=α_g_dif_nir, f_snow=f_snow, rad_type='NIR', pft_ind=pft_ind
        )

    # print(I_can_sun_db_par+I_can_sha_db_par+I_up_db_par+I_down_trans_can_par*(1-α_g_db_par)+I_down_db_par*(1-α_g_dif_par))
    # print(I_can_sun_db_nir+I_can_sha_db_nir+I_up_db_nir+I_down_trans_can_nir*(1-α_g_db_nir)+I_down_db_nir*(1-α_g_dif_nir))
    # print(I_can_sun_dif_par+I_can_sha_dif_par+I_up_dif_par+I_down_dif_par*(1-α_g_dif_par))
    # print(I_can_sun_dif_nir+I_can_sha_dif_nir+I_up_dif_nir+I_down_dif_nir*(1-α_g_dif_nir))

    # Calculate different solar fluxes
    S_v, S_g = calculate_solar_fluxes(
        S_db_par=S_db_par, S_dif_par=S_dif_par, S_db_nir=S_db_nir, S_dif_nir=S_dif_nir,
        I_can_db_par=I_can_sun_db_par+I_can_sha_db_par, I_can_dif_par=I_can_sun_dif_par+I_can_sha_dif_par, 
        I_can_db_nir=I_can_sun_db_nir+I_can_sha_db_nir, I_can_dif_nir=I_can_sun_dif_nir+I_can_sha_dif_nir,
        I_down_db_par=I_down_db_par, I_down_dif_par=I_down_dif_par, I_down_db_nir=I_down_db_nir, I_down_dif_nir=I_down_dif_nir, 
        I_down_trans_can_par=I_down_trans_can_par, I_down_trans_can_nir=I_down_trans_can_nir,
        α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir, 
    )

    # Check the energy balance of solar radiation
    is_solar_rad_balanced = check_solar_energy_conservation(
        S_db_par=S_db_par, S_dif_par=S_dif_par, S_db_nir=S_db_nir, S_dif_nir=S_dif_nir, S_v=S_v, S_g=S_g,
        I_up_db_par=I_up_db_par, I_up_dif_par=I_up_dif_par, I_up_db_nir=I_up_db_nir, I_up_dif_nir=I_up_dif_nir,
    )

    return S_v, S_g, is_solar_rad_balanced


def main_func(
    solar_rad: float, L_down: float, pres: float,
    f_snow: float, f_cansno: float, L: float, S: float, pft_ind: int,
    T_v_t1: float, T_v_t2: float, T_g_t1: float, T_g_t2: float,
    latitude: float, longitude: float, 
    year: int, day: int, hour: float,
    zone: int=8., is_day_saving: bool=False,
) -> Tuple:
    """This is the main function that 
       - takes in the solar radiation, incoming/downward longwave radiations, canopy/ground temperatures and
       - calculates the different fluxes from the top of the canopy to the ground surface.

    Args:
        solar_rad (float): The incoming solar radiation [W m2-1]
        L_down (float): The incoming longwave radiation [W m2-1]
        pres (float): The atmospheric pressure [kPa]
        f_cansno (float): The canopy snow-covered fraction [-]
        f_snow (float): The fraction of ground covered by snow [-]
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]
        pft_ind (int): The index of plant functional type based on the pft_clm5
        T_v_t1 (float): The vegetation temperature at the previous time step [degC]
        T_v_t2 (float): The vegetation temperature at the current time step [degC]
        T_g_t1 (float): The snow/soil surface temperature at the previous time step [degC]
        T_g_t2 (float): The snow/soil surface temperature at the current time step [degC]
        latitude (float): The latitude [degree].
        longitude (float): The longitude [degree].
        year (int): The year.
        day (int): The day of the year.
        hour (float): The fractional hour.
        zone (int, optional): The time zone. Defaults to 8..
        is_day_saving (bool, optional): Whether the current day is in the day time saving period. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    # Calculate the solar elevation angle
    solar_elev_angle = calculate_solar_elevation(
        latitude=latitude, longitude=longitude, year=year, day=day, hour=hour,
        zone=zone, is_day_saving=is_day_saving
    )

    # Perform the solar radiation partitioning
    S_db_par, S_dif_par, S_db_nir, S_dif_nir = partition_solar_radiation(
        solar_rad=solar_rad, solar_elev_angle=solar_elev_angle, pres=pres
    )

    # Calculate the albedos and emissivities
    α_g_db_par, α_g_dif_par = calculate_ground_albedos(f_snow, 'PAR')
    α_g_db_nir, α_g_dif_nir = calculate_ground_albedos(f_snow, 'NIR')
    ε_g, ε_v                = calculate_ground_vegetation_emissivity(
        solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L, S=S, pft_ind=pft_ind
        )

    # Calculate the canopy radiative transfer (PAR)
    I_up_db_par, I_up_dif_par, I_down_db_par, I_down_dif_par, I_down_trans_can_par, \
    I_can_sun_db_par, I_can_sha_db_par, I_can_sun_dif_par, I_can_sha_dif_par = \
        calculate_canopy_fluxes_per_unit_incident(
            solar_elev_angle=solar_elev_angle, L=L, S=S, f_cansno=f_cansno,
            α_g_db=α_g_db_par, α_g_dif=α_g_dif_par, f_snow=f_snow, rad_type='PAR', pft_ind=pft_ind
        )

    # Calculate the canopy radiative transfer (NIR)
    I_up_db_nir, I_up_dif_nir, I_down_db_nir, I_down_dif_nir, I_down_trans_can_nir, \
    I_can_sun_db_nir, I_can_sha_db_nir, I_can_sun_dif_nir, I_can_sha_dif_nir = \
        calculate_canopy_fluxes_per_unit_incident(
            solar_elev_angle=solar_elev_angle, L=L, S=S, f_cansno=f_cansno,
            α_g_db=α_g_db_nir, α_g_dif=α_g_dif_nir, f_snow=f_snow, rad_type='NIR', pft_ind=pft_ind
        )

    # print(I_can_sun_db_par+I_can_sha_db_par+I_up_db_par+I_down_trans_can_par*(1-α_g_db_par)+I_down_db_par*(1-α_g_dif_par))
    # print(I_can_sun_db_nir+I_can_sha_db_nir+I_up_db_nir+I_down_trans_can_nir*(1-α_g_db_nir)+I_down_db_nir*(1-α_g_dif_nir))
    # print(I_can_sun_dif_par+I_can_sha_dif_par+I_up_dif_par+I_down_dif_par*(1-α_g_dif_par))
    # print(I_can_sun_dif_nir+I_can_sha_dif_nir+I_up_dif_nir+I_down_dif_nir*(1-α_g_dif_nir))

    # Calculate different solar fluxes
    S_v, S_g = calculate_solar_fluxes(
        S_db_par=S_db_par, S_dif_par=S_dif_par, S_db_nir=S_db_nir, S_dif_nir=S_dif_nir,
        I_can_db_par=I_can_sun_db_par+I_can_sha_db_par, I_can_dif_par=I_can_sun_dif_par+I_can_sha_dif_par, 
        I_can_db_nir=I_can_sun_db_nir+I_can_sha_db_nir, I_can_dif_nir=I_can_sun_dif_nir+I_can_sha_dif_nir,
        I_down_db_par=I_down_db_par, I_down_dif_par=I_down_dif_par, I_down_db_nir=I_down_db_nir, I_down_dif_nir=I_down_dif_nir, 
        I_down_trans_can_par=I_down_trans_can_par, I_down_trans_can_nir=I_down_trans_can_nir,
        α_g_db_par=α_g_db_par, α_g_dif_par=α_g_dif_par, α_g_db_nir=α_g_db_nir, α_g_dif_nir=α_g_dif_nir, 
    )

    # Calculate different longwave fluxes
    L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg = calculate_longwave_fluxes(
        L_down=L_down, ε_v=ε_v, ε_g=ε_g, 
        T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
        L=L, S=S
    )

    # Calculate the net radiation
    Rnet = calculate_net_radiation_at_the_surface(S_v, S_g, L_v, L_g)

    # Check the energy balance of solar radiation
    is_solar_rad_balanced = check_solar_energy_conservation(
        S_db_par=S_db_par, S_dif_par=S_dif_par, S_db_nir=S_db_nir, S_dif_nir=S_dif_nir, S_v=S_v, S_g=S_g,
        I_up_db_par=I_up_db_par, I_up_dif_par=I_up_dif_par, I_up_db_nir=I_up_db_nir, I_up_dif_nir=I_up_dif_nir,
    )

    # Check the energy balance of longwave radiation
    is_longwave_balanced = check_longwave_energy_conservation(
        L_down=L_down, L_v=L_v, L_g=L_g, L_up=L_up, 
        L_up_g=L_up_g, L_down_v=L_down_v, L_up_v=L_up_v, δ_veg=δ_veg
    )

    # return Rnet, S_v, S_g, L_v, L_g, solar_rad_balanced, longwave_balanced
    return Rnet, S_v, S_g, L_v, L_g, is_solar_rad_balanced
