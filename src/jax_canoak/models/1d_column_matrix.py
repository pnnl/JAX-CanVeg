"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.07.24.
"""

# import jax
import jax.numpy as jnp

# import h5py
import numpy as np

from jax_canoak.subjects import Para, Met

# from jax_canoak.subjects import ParNir, Ir, Rnet, SunShadedCan
# from jax_canoak.subjects import Qin, Veg
from jax_canoak.subjects import initialize_profile_mx, initialize_model_states
from jax_canoak.shared_utilities.types import HashableArrayWrapper
from jax_canoak.physics.energy_fluxes import disp_canveg, diffuse_direct_radiation_mx
from jax_canoak.physics.energy_fluxes import sky_ir_mx, rad_tran_canopy_mx
from jax_canoak.physics.carbon_fluxes import angle_mx, leaf_angle_mx


f_forcing = "../shared_utilities/forcings/AlfMetBouldinInput.csv"
forcing_data = np.loadtxt(f_forcing, delimiter=",")
forcing_data = jnp.array(forcing_data)
n_time = forcing_data.shape[0]
lai = 4.0
forcing_data = jnp.concatenate([forcing_data, jnp.ones([n_time, 1]) * lai], axis=1)

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
soil, quantum, nir, ir, veg, qin, rnet, sun, shade = initialize_model_states(met, para)


# ---------------------------------------------------------------------------- #
#                     Compute sun angles                                       #
# ---------------------------------------------------------------------------- #
# beta_rad, solar_sin_beta, beta_deg = angle_mx(
sun_ang = angle_mx(para.lat_deg, para.long_deg, para.time_zone, met.day, met.hhour)
mask_night = sun_ang.sin_beta <= 0.0
mask_night_hashable = HashableArrayWrapper(mask_night)

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
ir.ir_in = sky_ir_mx(met.T_air_K, ratrad, para.sigma)


# ---------------------------------------------------------------------------- #
#                     Initialize profiles of scalars/sources/sinks             #
# ---------------------------------------------------------------------------- #
prof = initialize_profile_mx(met, para)

# ---------------------------------------------------------------------------- #
#                     Compute radiation fields             #
# ---------------------------------------------------------------------------- #
# PAR
quantum = rad_tran_canopy_mx(
    sun_ang, leaf_ang, quantum, para, mask_night_hashable, niter=25
)
# NIR
nir = rad_tran_canopy_mx(sun_ang, leaf_ang, nir, para, mask_night_hashable, niter=25)


# ---------------------------------------------------------------------------- #
#                     Iterations                                               #
# ---------------------------------------------------------------------------- #
# compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
# loop again and apply updated Tsfc info until convergence
# This is where things should be jitted as a whole


# # Plot
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 1, figsize=(10,8))
# ax.plot(met.day+met.hhour/24., par_beam, label='par_beam')
# ax.plot(met.day+met.hhour/24., par_diffuse, label='par_diffuse')
# plt.show()
