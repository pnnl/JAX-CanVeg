"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.08.13.
"""

import jax
import jax.numpy as jnp

# import h5py
import numpy as np

import equinox as eqx
from jax_canoak.subjects import Para, Met

from jax_canoak.subjects import initialize_profile_mx, initialize_model_states
from jax_canoak.subjects import update_profile_mx, calculate_veg_mx

# from jax_canoak.subjects.utils import llambda as flambda
from jax_canoak.shared_utilities.types import HashableArrayWrapper
from jax_canoak.shared_utilities.utils import dot
from jax_canoak.physics import energy_carbon_fluxes_mx
from jax_canoak.physics.energy_fluxes import disp_canveg, diffuse_direct_radiation_mx

from jax_canoak.physics.energy_fluxes import rad_tran_canopy_mx, sky_ir_v2_mx

# from jax_canoak.physics.energy_fluxes import rad_tran_canopy_mx, sky_ir_mx
from jax_canoak.physics.energy_fluxes import compute_qin_mx, ir_rad_tran_canopy_mx
from jax_canoak.physics.energy_fluxes import uz_mx, soil_energy_balance_mx
from jax_canoak.physics.carbon_fluxes import angle_mx, leaf_angle_mx


import matplotlib.pyplot as plt
from jax_canoak.shared_utilities.plot import plot_dij, plot_daily
from jax_canoak.shared_utilities.plot import plot_veg_temp

# from jax_canoak.shared_utilities import plot_ir, plot_rad, plot_canopy1, plot_dij
# from jax_canoak.shared_utilities import plot_soil, plot_soiltemp, plot_prof
# from jax_canoak.shared_utilities import plot_totalenergyplot_prof

jax.config.update("jax_enable_x64", True)

f_forcing = "../shared_utilities/forcings/AlfMetBouldinInput.csv"
forcing_data = np.loadtxt(f_forcing, delimiter=",")
forcing_data = jnp.array(forcing_data)
n_time = forcing_data.shape[0]
# lai = 3.6
lai = 5.0
forcing_data = jnp.concatenate([forcing_data, jnp.ones([n_time, 1]) * lai], axis=1)
plot = True

# ---------------------------------------------------------------------------- #
#                     Model parameter/properties settings                      #
# ---------------------------------------------------------------------------- #
time_zone = -8
latitude = 38.0991538
longitude = -121.49933
stomata = 2
hypo_amphi = 1
veg_ht = 0.8
leafangle = 1
n_can_layers = 50
meas_ht = 5.0
n_hr_per_day = 48
n_time = n_time


# ---------------------------------------------------------------------------- #
#                     Set the model forcings                                   #
# ---------------------------------------------------------------------------- #
zl0 = jnp.zeros(n_time)
# met = Met(forcing_data, n_time)
met = Met(forcing_data, n_time, zl0)


# ---------------------------------------------------------------------------- #
#                     Set up model parameter instance                      #
# ---------------------------------------------------------------------------- #
para = Para(
    time_zone=time_zone,
    latitude=latitude,
    longitude=longitude,
    stomata=stomata,
    hypo_amphi=hypo_amphi,
    veg_ht=veg_ht,
    leafangle=leafangle,
    n_can_layers=n_can_layers,
    meas_ht=meas_ht,
    n_hr_per_day=n_hr_per_day,
    n_time=n_time,
    npart=int(1e6),
)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
dij = disp_canveg(para, timemax=5000.0)
# dij = np.loadtxt('Dij_Alfalfa.csv', delimiter=',')
# dij = jnp.array(dij)


# ---------------------------------------------------------------------------- #
#                     Initialize model states                        #
# ---------------------------------------------------------------------------- #
soil, quantum, nir, ir, qin, rnet, sun, shade, lai = initialize_model_states(met, para)


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
quantum = eqx.tree_at(
    lambda t: (t.inbeam, t.indiffuse), quantum, (par_beam, par_diffuse)
)
nir = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), nir, (nir_beam, nir_diffuse))
# quantum.inbeam, quantum.indiffuse = par_beam, par_diffuse
# quantum.incoming = quantum.inbeam + quantum.indiffuse
# nir.inbeam, nir.indiffuse = nir_beam, nir_diffuse
# nir.incoming = nir.inbeam + nir.indiffuse

# ---------------------------------------------------------------------------- #
#                     Compute leaf angle                                       #
# ---------------------------------------------------------------------------- #
leaf_ang = leaf_angle_mx(sun_ang, para, lai)


# ---------------------------------------------------------------------------- #
#                     Initialize IR fluxes with air temperature                #
# ---------------------------------------------------------------------------- #
# ir.ir_in = sky_ir_mx(met.T_air_K, ratrad, para.sigma)
ir_in = sky_ir_v2_mx(met, ratrad, para.sigma)
ir_dn = dot(ir_in, ir.ir_dn)
ir_up = dot(ir_in, ir.ir_up)
ir = eqx.tree_at(lambda t: (t.ir_in, t.ir_dn, t.ir_up), ir, (ir_in, ir_dn, ir_up))


# ---------------------------------------------------------------------------- #
#                     Initialize profiles of scalars/sources/sinks             #
# ---------------------------------------------------------------------------- #
prof = initialize_profile_mx(met, para)


# ---------------------------------------------------------------------------- #
#                     Generate masks for matrix calculations                   #
# ---------------------------------------------------------------------------- #
# Night mask
mask_night = sun_ang.sin_beta <= 0.0
mask_night_hashable = HashableArrayWrapper(mask_night)
# Turbulence mask
nnu_T_P = dot(
    para.nnu * (101.3 / met.P_kPa),
    jnp.power(prof.Tair_K[:, : para.jtot] / 273.16, 1.81),
)
Re = para.lleaf * prof.wind[:, : para.jtot] / nnu_T_P
mask_turbulence = Re > 14000.0
mask_turbulence_hashable = HashableArrayWrapper(mask_turbulence)


# ---------------------------------------------------------------------------- #
#                     Compute radiation fields             #
# ---------------------------------------------------------------------------- #
# PAR
quantum = rad_tran_canopy_mx(
    sun_ang, leaf_ang, quantum, para, lai, mask_night_hashable, niter=5
)
# print(quantum.inbeam)
# NIR
nir = rad_tran_canopy_mx(
    sun_ang, leaf_ang, nir, para, lai, mask_night_hashable, niter=25
)
# jax.debug.print("{a}", a=jnp.mean(nir.dn_flux, 0))
# jax.debug.print("{a}", a=nir.dn_flux[1,:])
# plt.imshow(nir.dn_flux, cmap='Blues', vmin=0., aspect='auto')


# ---------------------------------------------------------------------------- #
#                     Iterations                                               #
# ---------------------------------------------------------------------------- #
# compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
# loop again and apply updated Tsfc info until convergence
# This is where things should be jitted as a whole
def iteration(c, i):
    met, prof, ir, qin, sun, shade, soil = c
    # jax.debug.print("T soil: {a}", a=soil.T_soil[10,:])
    jax.debug.print("T sfc: {a}", a=soil.sfc_temperature[10])

    # Update canopy wind profile with iteration of z/l and use in boundary layer
    # resistance computations
    wind = uz_mx(met, para)
    prof = eqx.tree_at(lambda t: t.wind, prof, wind)

    # Compute IR fluxes with Bonan's algorithms of Norman model
    ir = ir_rad_tran_canopy_mx(leaf_ang, ir, quantum, soil, sun, shade, para)
    # jax.debug.print("ir: {a}", a=ir.ir_dn[10,:])

    # Incoming short and longwave radiation
    qin = compute_qin_mx(quantum, nir, ir, para, qin)

    # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
    # Compute new boundary layer conductances based on new leaf energy balance
    # and delta T, in case convection occurs
    # Different coefficients will be assigned if amphistomatous or hypostomatous
    sun, shade = energy_carbon_fluxes_mx(
        sun, shade, qin, quantum, met, prof, para, mask_turbulence_hashable
    )

    # Compute soil fluxes
    soil = soil_energy_balance_mx(
        quantum, nir, ir, met, prof, para, soil
    )  # type: ignore

    # Compute profiles of C's, zero layer jtot+1 as that is not a dF/dz or
    # source/sink level
    prof = update_profile_mx(met, para, prof, quantum, sun, shade, soil, lai, dij)

    # compute met.zL from HH and met.ustar
    HH = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.H
            + quantum.prob_shade[:, : para.nlayers] * shade.H
        )
        * lai.dff[:, : para.nlayers],
        axis=1,
    )
    zL = -(0.4 * 9.8 * HH * para.meas_ht) / (
        met.air_density * 1005 * met.T_air_K * jnp.power(met.ustar, 3.0)
    )
    met.zL = jnp.clip(zL, a_min=-3, a_max=0.25)

    # # Compute the vegetation overall photosynthesis and respiration
    # veg_Ps = jnp.sum(
    #     (
    #         quantum.prob_beam[:, : para.nlayers] * sun.Ps
    #         + quantum.prob_shade[:, : para.nlayers] * shade.Ps
    #     )
    #     * lai.dff[:, : para.nlayers],
    #     axis=1,
    # )
    # veg_Rd = jnp.sum(
    #     (
    #         quantum.prob_beam[:, : para.nlayers] * sun.Resp
    #         + quantum.prob_shade[:, : para.nlayers] * shade.Resp
    #     )
    #     * lai.dff[:, : para.nlayers],
    #     axis=1,
    # )

    cnew = [met, prof, ir, qin, sun, shade, soil]
    return cnew, None


initials = [met, prof, ir, qin, sun, shade, soil]
finals, _ = jax.lax.scan(iteration, initials, xs=None, length=15)

met, prof, ir, qin, sun, shade, soil = finals

# Compute canopy integrated fluxes
veg = calculate_veg_mx(para, lai, quantum, sun, shade)

# Net radiation budget at top of the canopy
can_rnet = (
    quantum.beam_flux[:, para.jtot] / 4.6
    + quantum.dn_flux[:, para.jtot] / 4.6
    - quantum.up_flux[:, para.jtot] / 4.6
    + nir.beam_flux[:, para.jtot]
    + nir.dn_flux[:, para.jtot]
    - nir.up_flux[:, para.jtot]
    + ir.ir_dn[:, para.jtot]
    + -ir.ir_up[:, para.jtot]
)

# ---------------------------------------------------------------------------- #
#                     Plot                                              #
# ---------------------------------------------------------------------------- #
if plot:
    # plot_rad(quantum, para, lai, "par")
    # plot_rad(nir, para, lai, "nir")
    # plot_ir(ir, para, lai)
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # plot_canopy1(sun, qin, para, "sun", axes[0])
    # plot_canopy1(shade, qin, para, "shade", axes[1])
    # plot_canopy2(sun, para, "sun")
    # plot_canopy2(shade, para, "shade")
    # plot_leafang(leaf_ang, para)
    plot_veg_temp(sun, shade, para)
    plot_dij(dij, para)
    # plot_soil(soil, para)
    # plot_soiltemp(soil, para)
    # plot_totalenergy(soil, veg, can_rnet)
    # plot_prof1(prof)
    # plot_prof2(prof, para)
    plot_daily(met, soil, veg, para)
    plt.show()
