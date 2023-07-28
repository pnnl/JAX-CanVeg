"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.07.24.
"""

import jax
import jax.numpy as jnp

# import h5py
import numpy as np

from jax_canoak.subjects import Para, Met, Soil
from jax_canoak.subjects import ParNir, Ir, Rnet, SunShadedCan
from jax_canoak.subjects import Qin, Veg
from jax_canoak.physics.energy_fluxes import disp_canveg, diffuse_direct_radiation_mx
from jax_canoak.physics.carbon_fluxes import angle_mx, leaf_angle_mx


f_forcing = "../shared_utilities/forcings/AlfMetBouldinInput.csv"
forcing_data = np.loadtxt(f_forcing, delimiter=",")
forcing_data = jnp.array(forcing_data)
n_time = forcing_data.shape[0]

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
para = Para(
    time_zone=-8,
    latitude=38.0991538,
    longitude=-121.49933,
    stomata=2,
    hypo_amphi=1,
    veg_ht=0.8,
    leafangle=1,
    n_can_layers=30,
    meas_ht=5.0,
    n_hr_per_day=48,
    n_time=n_time,
)


# ---------------------------------------------------------------------------- #
#                     Set the model forcings                                   #
# ---------------------------------------------------------------------------- #
met = Met(forcing_data, para.ntime, para.Mair, para.rugc)
para.set_lai(met.lai)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
dij = disp_canveg(para)


# ---------------------------------------------------------------------------- #
#                     Initialize model states                        #
# ---------------------------------------------------------------------------- #
soil = Soil(met, para)
quantum, nir = ParNir(para.ntime, para.jtot), ParNir(para.ntime, para.jtot)
ir, veg = Ir(para.ntime, para.jtot), Veg(para.ntime)
qin, rnet = Qin(para.ntime, para.jktot), Rnet(para.ntime, para.jktot)
sun, shade = SunShadedCan(para.ntime, para.jktot), SunShadedCan(para.ntime, para.jktot)

dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)
sun.Tsfc = dot(met.T_air_K, sun.Tsfc)
shade.Tsfc = dot(met.T_air_K, shade.Tsfc)


# ---------------------------------------------------------------------------- #
#                     Compute sun angles                                       #
# ---------------------------------------------------------------------------- #
# beta_rad, solar_sin_beta, beta_deg = angle_mx(
sun_ang = angle_mx(para.lat_deg, para.long_deg, para.time_zone, met.day, met.hhour)

# ---------------------------------------------------------------------------- #
#                     Compute direct and diffuse radiations                    #
# ---------------------------------------------------------------------------- #
ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = diffuse_direct_radiation_mx(
    sun_ang.sin_beta, met.rglobal, met.parin, met.P_kPa
)
quantum.inbeam, quantum.indiffuse = par_beam, par_diffuse
quantum.incoming = quantum.inbeam + quantum.indiffuse
nir.inbeam, nir.indiffuse = nir_beam, nir_diffuse
nir.incoming = nir.inbeam + nir.indiffuse


# ---------------------------------------------------------------------------- #
#                     Compute leaf angle                                       #
# ---------------------------------------------------------------------------- #
leaf_ang = leaf_angle_mx(sun_ang, para)


# ---------------------------------------------------------------------------- #
#                     Initialize IR fluxes with air temperature                #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                     Initialize profiles of scalars/sources/sinks             #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                     Compute radiation fields             #
# ---------------------------------------------------------------------------- #
# PAR

# NIR


# ---------------------------------------------------------------------------- #
#                     Iterations                                               #
# ---------------------------------------------------------------------------- #
# compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
# loop again and apply updated Tsfc info until convergence
# This is where things should be jitted as a whole
