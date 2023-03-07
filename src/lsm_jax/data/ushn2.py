"""Get the data at US-Hn2 site."""

import requests
from statistics import mean
from pathlib import Path

import pandas as pd
import jax.numpy as jnp
from diffrax import LinearInterpolation

# from .base import DataBase
from .point_data import PointData

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
varn_list = ['P_F', 'SW_IN_F', 'LW_IN_F', 'NETRAD', 'G_F_MDS', 'LE_F_MDS', 'H_F_MDS', 
             'VPD_F_MDS', 'TA_F_MDS', 'WS_F', 'TS_F_MDS_2', 'SWC_F_MDS_1', 'LAI']
df_obs = pd.concat([df_flux, df_modis], axis=1)
df_obs = df_obs[varn_list]
df_obs.interpolate(axis=0, inplace=True)

# Rename
varn_list_new =['P', 'SW_IN', 'LW_IN', 'NETRAD', 'G', 'LE', 'H', 'VPD', 'TA', 'WS', 'TS', 'SWC', 'LAI'] 
df_obs.rename(
    columns=dict(zip(varn_list, varn_list_new)), inplace=True
)

# Unit conversion
df_obs['P']      = df_obs['P'] * 1e-3                # [mm] -> [m]
df_obs['NETRAD'] = df_obs['NETRAD']*86400*1e-6 # [W m-2] -> [MJ m-2 day-1]
df_obs['SW_IN']  = df_obs['SW_IN']*86400*1e-6 # [W m-2] -> [MJ m-2 day-1]
df_obs['LW_IN']  = df_obs['LW_IN']*86400*1e-6 # [W m-2] -> [MJ m-2 day-1]
df_obs['G']      = df_obs['G']*86400*1e-6     # [W m-2] -> [MJ m-2 day-1]
df_obs['LE']     = df_obs['LE']*86400*1e-6     # [W m-2] -> [MJ m-2 day-1]
df_obs['H']      = df_obs['H']*86400*1e-6     # [W m-2] -> [MJ m-2 day-1]
df_obs['VPD']    = df_obs['VPD']*0.1        # [hPa] -> [kPa]
df_obs['TA']     = df_obs['TA']*1.           # [degC] -> [degC]
df_obs['WS']     = df_obs['WS']*1.               # [m s-1] -> [m s-1]
df_obs['TS']     = df_obs['TS']*1.         # [degC] -> [degC]
df_obs['SWC']    = df_obs['SWC']*1e-2     # [%] -> [-]
df_obs['LAI']    = df_obs['LAI']*1.              # [-] -> [-]

ts = df_obs.index - pd.to_datetime(start, format='%Y-%m-%d')
ts = jnp.asarray(ts.days.values)
ys = jnp.asarray(df_obs.values.T)
forcings = PointData(
    varn_list=varn_list_new, data=ys, ts=ts
)

# # Unit conversion
# obs = {
#     'P': df_flux['P_F']*1e-3,               # [mm] -> [m]
#     'NETRAD': df_flux['NETRAD']*86400*1e-6, # [W m-2] -> [MJ m-2 day-1]
#     'G': df_flux['G_F_MDS']*86400*1e-6,     # [W m-2] -> [MJ m-2 day-1]
#     'VPD': df_flux['VPD_F_MDS']*0.1,        # [hPa] -> [kPa]
#     'TA': df_flux['TA_F_MDS']*1.,           # [degC] -> [degC]
#     'WS': df_flux['WS_F']*1.,               # [m s-1] -> [m s-1]
#     'TS': df_flux['TS_F_MDS_2']*1.,         # [degC] -> [degC]
#     'SWC': df_flux['SWC_F_MDS_1']*1e-2,     # [%] -> [-]
#     'LAI': df_modis['LAI']*1.,              # [-] -> [-]
# }


# # Create the interpolations
# obs_interp = {}
# for varn,df in obs.items():
#     df = df.dropna()
#     t = df.index - pd.to_datetime(start, format='%Y-%m-%d')
#     obs_interp[varn] = LinearInterpolation(
#         ts=jnp.asarray(t.days.values), ys=jnp.asarray(df.values)
#     )

# forcings = DataBase(forcings=obs_interp, dt=1.)


# # The observation resolution
# obs_interp['dt'] = 1.  # [day]

