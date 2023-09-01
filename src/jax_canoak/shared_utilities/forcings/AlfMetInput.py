"""Get the data at AlfMetInput site."""

from pathlib import Path

import pandas as pd
import jax.numpy as jnp

from .point_data import PointData
from ..domain import Time

start, end = "2018-01-01 00:00:00", "2020-10-01 12:00:00"

########################################################################
# forcing data
########################################################################
# dir_data = Path('.')
dir_data = Path(
    # "/Users/jian449/Library/CloudStorage/OneDrive-PNNL/Codes/jax-watershed/src/jax_canoak/shared_utilities/forcings"
    "../../../../data"
)
f_obs = dir_data / "fluxtower" / "Alf" / "AlfMetInput.csv"
df_obs = pd.read_csv(f_obs)
df_obs["year"] = 2018
df_obs.index = pd.to_datetime(
    df_obs["year"] * 1000 + df_obs["day"], format="%Y%j"
) + pd.to_timedelta(df_obs["hour"], unit="h")

########################################################################
# LAI data
########################################################################
df_obs["LAI"] = 4.0

########################################################################
# Turbulance divergence data
########################################################################
f_div = dir_data / "dij" / "Alf" / "AlfDIJ5000.csv"
df_div = pd.read_csv(f_div, header=None)
divergence = df_div.iloc[:, 0].values.reshape([30, 150]).T
divergence = jnp.array(divergence)

########################################################################
# Put everything together to a forcing class
########################################################################
varn_list = [
    "TairC",
    "Rg_Wm-2",
    "ea_kPa",
    "U_ms-1",
    "CO2_ppm",
    "P_kPa",
    "u*_ms-1",
    "TsoilC",
    "soilmoisture",
    "LAI",
]
df_obs = df_obs[varn_list]
df_obs = df_obs.interpolate(axis=0).ffill().bfill()

# Rename
varn_list_new = [
    "TA",  # degC
    "Rg",  # W m-2
    "VP",  # kPa
    "WS",  # m s-1
    "CO2A",  # ppm
    "PA",  # kPa
    "Ustar",  # m s-1
    "TS",  # degC
    "SWC",  # [-]
    "LAI",  # [-]
]
df_obs.rename(columns=dict(zip(varn_list, varn_list_new)), inplace=True)

# Calculate the air density, specific humidity, saturated water vapor pressure,
# and water vapor pressure
# df_obs["VP_SAT"] = esat_from_temp(
#     df_obs["TA"].values
# )  # Saturated water vapor pressure [Pa]
# df_obs["VP"] = df_obs["VP_SAT"] - df_obs["VPD"]  # Water vapor pressure [Pa]
# df_obs["ROA"] = ρ_from_e_pres_temp(
#     pres=df_obs["PA"].values, e=df_obs["VP"].values, T=df_obs["TA"].values
# )  # Air density [kg m-3]
# df_obs["SH"] = q_from_e_pres(
#     pres=df_obs["PA"].values, e=df_obs["VP"].values
# )  # Specific humidity [kg kg-1]

ts = df_obs.index - pd.to_datetime(start, format="%Y-%m-%d %H:%M:%S")
ts = ts + pd.Timedelta(days=1)
# ts = jnp.asarray(ts.days.values)
ts = list(ts.days.values + ts.seconds.values / 86400.0)
ts = Time(
    t0=ts[0], tn=ts[-1], t_list=jnp.array(ts[1:-1]), time_unit="Day", start_time=start
)
ys = jnp.asarray(df_obs.values.T)
forcings = PointData(
    # varn_list=varn_list_new, data=ys, ts=ts
    varn_list=list(df_obs.keys()),
    data=ys,
    ts=ts,
)
