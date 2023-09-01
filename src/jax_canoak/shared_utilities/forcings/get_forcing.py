import requests
from statistics import mean
import pandas as pd
import jax.numpy as jnp

from typing import Tuple
from .point_data import PointData
from ..types import Float_0D

from ..constants import bprime, mass_air, rugc, cp


def get_modis(
    site: str,
    start: str,
    end: str,
    network: str = "AMERIFLUX",
    product: str = "MCD15A3H",
    sample_freq: str = "D",
) -> pd.core.frame.DataFrame:
    modis_baseurl = "https://modis.ornl.gov/rst/api/v1/"
    urlheader_txt = {"Accept": "text/csv"}
    urlheader_json = {"Accept": "json"}

    # Query the dates in the MODIS product
    query = "".join([modis_baseurl, product, f"/{network}/", site, "/dates"])
    response = requests.get(query, headers=urlheader_txt)
    dates = response.text.split(",")

    # Query the product data
    query = "".join(
        [
            modis_baseurl,
            product,
            "/AMERIFLUX/",
            site,
            "/subset?",
            "&startDate=",
            dates[0],
            "&endDate=",
            dates[-1],
        ]
    )
    response = requests.get(query, headers=urlheader_json)
    data_json = response.json()

    # Get LAI, FparLai_QC, and Fpar_500m
    nt = int(len(data_json["subset"]) / 6)
    # time_set, lai_set, fpar_set, qc_set = [], [], [], []
    time_set, lai_set, fpar_set = [], [], []
    for i in range(nt):
        lai = data_json["subset"][i * 6 + 5]
        fpar = data_json["subset"][i * 6 + 3]
        qc = data_json["subset"][i * 6 + 1]
        assert lai["calendar_date"] == qc["calendar_date"]
        assert fpar["calendar_date"] == qc["calendar_date"]
        assert lai["band"] == "Lai_500m"
        assert fpar["band"] == "Fpar_500m"
        assert qc["band"] == "FparLai_QC"

        # Calculate the mean of lai and fpar on the valid data
        time = lai["calendar_date"]
        lai, fpar, qc = lai["data"], fpar["data"], qc["data"]
        lai = [l for l in lai if (l >= 0) and (l <= 100)]  # noqa: E741
        fpar = [l for l in fpar if (l >= 0) and (l <= 100)]  # noqa: E741

        # if time=='2016-01-05':
        #     print(qc)

        if (len(lai) == 0) or (len(fpar) == 0):
            continue
        else:
            lai, fpar = mean(lai), mean(fpar)
            if lai == 0:
                continue

        # Append them to the sets
        time_set.append(time)
        lai_set.append(lai)
        fpar_set.append(fpar)

    # Convert it to pandas dataframe
    df_modis = pd.DataFrame(
        jnp.array([lai_set, fpar_set]).T, index=time_set, columns=["LAI", "FPAR"]
    )
    df_modis.index = pd.to_datetime(df_modis.index, format="%Y-%m-%d")
    df_modis = df_modis.resample(sample_freq).interpolate()
    df_modis = df_modis[start:end]

    return df_modis


def get_input_t(forcings: PointData, t: Float_0D) -> Tuple:
    """Get the forcings at the time t.

    Args:
        forcings (PointData): _description_
        t (Float_1D): _description_
    """
    forcing_list = forcings.varn_list
    rg_ind, pa_ind, lai_ind = (
        forcing_list.index("Rg"),
        forcing_list.index("PA"),
        forcing_list.index("LAI"),
    )
    ta_ind, ws_ind, us_ind = (
        forcing_list.index("TA"),
        forcing_list.index("WS"),
        forcing_list.index("Ustar"),
    )
    co2_ind, ea_ind, ts_ind, swc_ind = (
        forcing_list.index("CO2A"),
        forcing_list.index("VP"),
        forcing_list.index("TS"),
        forcing_list.index("SWC"),
    )

    # Get the forcing data
    forcing_now = forcings.interpolate_time(t)
    rglobal, press_kpa, lai = (
        forcing_now[rg_ind],
        forcing_now[pa_ind],
        forcing_now[lai_ind],
    )
    ta, ws, ustar = (
        forcing_now[ta_ind],
        forcing_now[ws_ind],
        forcing_now[us_ind],
    )
    co2, ea, ts, swc = (
        forcing_now[co2_ind],
        forcing_now[ea_ind],
        forcing_now[ts_ind],
        forcing_now[swc_ind],
    )

    if ustar < 0.07:
        ustar = ws * 0.095

    # Compute absolute air temperature
    T_Kelvin = ta + 273.15

    # Compute absolute humidity, g ,-3
    rhova_g = ea * 2165 / T_Kelvin

    # Comptue relative humidity
    # est = es(T_Kelvin)
    est = 613 * jnp.exp(17.502 * ta / (240.97 + ta))
    relative_humidity = ea * 10.0 / est

    # Vapor pressure deficit, mb
    vpd = est - ea * 10.0

    # check for bad data
    if rhova_g < 0:
        rhova_g = 0

    # Air pressure
    press_bars = press_kpa / 100.0
    press_Pa = press_kpa * 1000.0

    # Combining gas law constants
    pstat273 = 0.022624 / (273.16 * press_bars)

    # cuticular conductance adjusted for pressure and T, mol m-2 s-1
    gcut = bprime * T_Kelvin * pstat273

    # cuticular resistance
    rcuticle = 1.0 / gcut

    # check for bad CO2 data
    if jnp.abs(co2) >= 998.0:
        co2 = 400.0

    # check for bad Rg
    if rglobal < 0:
        rglobal = 0

    # Get par data
    parin = 4.6 * rglobal / 2.0  # umol m-2 s-1

    # check for bad par data
    if parin < 0:
        parin = 0.0

        # solar.par_beam=0.;
        # solar.par_diffuse=0.;
        # solar.nir_beam=0.;
        # solar.nir_diffuse=0.;

    # set some limits on bad input data to minimize the model from blowing up
    # if (solar.ratrad > 0.9) | (solar.ratrad < 0.2):
    #     solar.ratrad=0.5

    # if humidity is bad, estimate it as some reasonable value, e.g. annual average
    if rhova_g > 30.0:
        rhova_g = 10.0
    rhova_kg = rhova_g / 1000.0

    # air density, kg m-3
    air_density = press_kpa * mass_air / (rugc * T_Kelvin)

    # air density, mole m-3
    air_density_mole = press_kpa / (rugc * T_Kelvin) * 1000.0

    soil_Tave_15cm = ts

    # Heat coefficient
    heatcoef = air_density * cp

    # The initial heat flux
    H_old = 0.0

    return (
        rglobal,
        parin,
        press_kpa,
        lai,
        ta,
        ws,
        ustar,
        co2,
        ea,
        ts,
        swc,
        T_Kelvin,
        rhova_g,
        rhova_kg,
        relative_humidity,
        vpd,
        press_bars,
        press_Pa,
        pstat273,
        gcut,
        rcuticle,
        air_density,
        air_density_mole,
        soil_Tave_15cm,
        heatcoef,
        H_old,
    )
