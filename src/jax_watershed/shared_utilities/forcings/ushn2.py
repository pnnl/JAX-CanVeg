"""Get the data at US-Hn2 site."""

import requests
from statistics import mean
from pathlib import Path

import pandas as pd
import jax.numpy as jnp
from diffrax import LinearInterpolation

from .point_data import PointData
from ..constants import C_TO_K as c2k
from ..domain import Time
from ...physics.water_fluxes import esat_from_temp, ρ_from_e_pres_temp, q_from_e_pres

site = 'US-Hn2'
network = 'AMERIFLUX'
product = 'MCD15A3H'

start, end = '2016-01-05', '2017-10-01'

# flux_varn_list = ['PPFD_IN', 'SW_IN_F_MDS', 'LW_IN_F_MDS', 'NETRAD', 'G_F_MDS',
#              'P_F', 'WS_F', 'VPD_F_MDS', 'TA_F_MDS',  'PA_F',
#              'TS_F_MDS_1', 'TS_F_MDS_2', 'SWC_F_MDS_1']

########################################################################
# Flux tower data
########################################################################
dir_data = Path('/Users/jian449/Library/CloudStorage/OneDrive-PNNL/Data/AmeriFlux/AMF_US-Hn2_FLUXNET_FULLSET_2015-2018_3-5')
f_flux   = dir_data / 'AMF_US-Hn2_FLUXNET_FULLSET_DD_2015-2018_3-5.csv'
f_flux_hh= dir_data / 'AMF_US-Hn2_FLUXNET_FULLSET_HH_2015-2018_3-5.csv'

df_flux = pd.read_csv(f_flux, index_col=0)
df_flux.index = pd.to_datetime(df_flux.index, format="%Y%m%d")
df_flux = df_flux[start:end]

# df_flux = df_flux[flux_varn_list]

df_flux[df_flux==-9999] = jnp.nan

########################################################################
# MODIS data
########################################################################
modis_baseurl = "https://modis.ornl.gov/rst/api/v1/"
urlheader_txt = {'Accept': 'text/csv'}
urlheader_json = {'Accept': 'json'}

query = modis_baseurl+'{}/sites'.format(network)
r = requests.get(query, headers=urlheader_txt)

# Query the dates in the MODIS product
query = "".join([
        modis_baseurl, product, 
        "/AMERIFLUX/", site, "/dates"
])
response = requests.get(query, headers=urlheader_txt)
dates = response.text.split(',')

# Query the product data
query = "".join([
        modis_baseurl, product, 
        "/AMERIFLUX/", site, "/subset?",
        "&startDate=", dates[0],
        "&endDate=", dates[-1],
])
response = requests.get(query, headers=urlheader_json)
data_json = response.json()

# Get LAI, FparLai_QC, and Fpar_500m
nt = int(len(data_json['subset'])/6)
time_set, lai_set, fpar_set, qc_set = [], [], [], []
for i in range(nt):
    lai  = data_json['subset'][i*6+5]
    fpar = data_json['subset'][i*6+3]
    qc   = data_json['subset'][i*6+1]
    assert lai['calendar_date'] == qc['calendar_date']
    assert fpar['calendar_date'] == qc['calendar_date']
    assert lai['band'] == 'Lai_500m'
    assert fpar['band'] == 'Fpar_500m'
    assert qc['band'] == 'FparLai_QC'
    
    # Calculate the mean of lai and fpar on the valid data
    time = lai['calendar_date']
    lai, fpar, qc = lai['data'], fpar['data'], qc['data']
    lai  = [l for l in lai if (l>=0) and (l<=100)]
    fpar = [l for l in fpar if (l>=0) and (l<=100)]
    
    # if time=='2016-01-05':
    #     print(qc)
    
    if (len(lai) == 0) or (len(fpar) == 0):
        continue
    else:
        lai, fpar = mean(lai), mean(fpar)
        if lai == 0: continue
    
    # Append them to the sets
    time_set.append(time)
    lai_set.append(lai)
    fpar_set.append(fpar)

# Convert it to pandas dataframe
df_modis = pd.DataFrame(jnp.array([lai_set, fpar_set]).T, index=time_set, columns=['LAI', 'FPAR'])
df_modis.index = pd.to_datetime(df_modis.index, format="%Y-%m-%d")
df_modis = df_modis.resample('D').interpolate()
df_modis = df_modis[start:end]

########################################################################
# Put everything together to a forcing class
########################################################################
varn_list = ['P_F', 'SW_IN_F', 'LW_IN_F', 'NETRAD', 'G_F_MDS', 'LE_F_MDS', 'H_F_MDS', 'USTAR',
             'VPD_F_MDS', 'TA_F_MDS', 'WS_F', 'PA_F', 'TS_F_MDS_2', 'SWC_F_MDS_1', 'LAI']
df_obs = pd.concat([df_flux, df_modis], axis=1)
df_obs = df_obs[varn_list]
df_obs.interpolate(axis=0, inplace=True)

# Rename
varn_list_new =['P', 'SW_IN', 'LW_IN', 'NETRAD', 'G', 'LE', 'H', 'USTAR', 'VPD', 'TA', 'WS', 'PA', 'TS', 'SWC', 'LAI'] 
df_obs.rename(
    columns=dict(zip(varn_list, varn_list_new)), inplace=True
)

# Unit conversion
df_obs['P']      = df_obs['P'] * 1e-3 # [mm] -> [m]
df_obs['NETRAD'] = df_obs['NETRAD']   # [W m-2]
df_obs['SW_IN']  = df_obs['SW_IN']    # [W m-2]
df_obs['LW_IN']  = df_obs['LW_IN']    # [W m-2]
df_obs['G']      = df_obs['G']        # [W m-2]
df_obs['LE']     = df_obs['LE']       # [W m-2]
df_obs['H']      = df_obs['H']        # [W m-2]
df_obs['VPD']    = df_obs['VPD']*100  # [hPa] -> [Pa]
df_obs['PA']     = df_obs['PA']*1000  # [kPa] -> [Pa]
df_obs['TA']     = df_obs['TA'] + c2k # [degC] -> [degK]
df_obs['WS']     = df_obs['WS']*1.    # [m s-1] -> [m s-1]
df_obs['TS']     = df_obs['TS'] + c2k # [degC] -> [degK]
df_obs['SWC']    = df_obs['SWC']*1e-2 # [%] -> [-]
df_obs['LAI']    = df_obs['LAI']*1.   # [-] -> [-]

# Calculate the air density, specific humidity, saturated water vapor pressure, and water vapor pressure
df_obs['VP_SAT'] = esat_from_temp(df_obs['TA'].values)   # Saturated water vapor pressure [Pa]
df_obs['VP'] = df_obs['VP_SAT'] - df_obs['VPD']          # Water vapor pressure [Pa]
df_obs['ROA'] = ρ_from_e_pres_temp(
    pres=df_obs['PA'].values, e=df_obs['VP'].values, T=df_obs['TA'].values)         # Air density [kg m-3]
df_obs['SH'] = q_from_e_pres(
    pres=df_obs['PA'].values, e=df_obs['VP'].values)          # Specific humidity [kg kg-1]


ts = df_obs.index - pd.to_datetime(start, format='%Y-%m-%d')
# ts = jnp.asarray(ts.days.values)
ts = list(ts.days.values)
ts = Time(t0=ts[0], tn=ts[-1], t_list=ts[1:-1], time_unit="Day", start_time=start)
ys = jnp.asarray(df_obs.values.T)
forcings = PointData(
    # varn_list=varn_list_new, data=ys, ts=ts
    varn_list=list(df_obs.keys()), data=ys, ts=ts
)
