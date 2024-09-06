"""
Function for computing the dispersion matrix.

- disp_canveg()
- get_dispersion_matrix()

Author: Peishi Jiang
Date: 2023.7.25.
"""

import os

# import time
# import jax
import jax.numpy as jnp
import numpy as np

from typing import Optional

# import pandas as pd

from ...subjects import Para, Setup
from ...shared_utilities.types import Float_0D, Float_2D

# from ...shared_utilities.utils import dot
from .dispersion import disp_mx
import logging


def disp_canveg(
    setup: Setup,
    prm: Para,
    timemax: Float_0D = 1000.0,
    ustar: Float_0D = 1.0,
    f_dij: str = "./Dij.txt",
) -> Float_2D:
    logging.info("Generating dispersion matrix ...")
    hh = prm.veg_ht  #  canopy height (m)
    dd = prm.dht

    # dij = np.zeros([prm.jktot3, prm.jktot])
    # dij = np.zeros([prm.nlayers_atmos + 1, prm.jktot])
    dij = np.zeros([setup.n_total_layers + 1, setup.n_can_layers + 1])
    # Generate the Dij file
    disp_mx(
        # int(prm.jktot),
        # int(prm.nlayers_atmos + 1),
        int(setup.n_can_layers + 1),
        int(setup.n_total_layers + 1),
        float(setup.npart),
        float(timemax),
        float(ustar),
        float(hh),
        float(dd),
        f_dij,
        dij,
    )

    # Get the calculated dij
    dij = jnp.array(dij[1:, 1:])

    # Read the Dij file
    # df_div = pd.read_csv(f_dij, header=None)
    # print(f_dij)
    # print(df_div.head())
    # print(df_div.tail())
    # dij = df_div.iloc[:, 0].values.reshape([prm.jtot, prm.jtot3]).T
    # dij = np.loadtxt(f_dij)[:,0]
    # dij = dij.reshape([prm.jtot, prm.jtot3]).T
    # dij = jnp.array(dij)
    # print(dij.shape)
    # print(dij[:,1])

    # Remove the file
    # os.remove(f_dij)

    return dij


def get_dispersion_matrix(
    setup: Setup, para: Para, f_dij: Optional[str] = None, timemax: Float_0D = 1000.0
) -> Float_2D:
    # Get dij from an existing csv file
    if os.path.isfile(f_dij):  # pyright: ignore
        logging.info(f"Reading dispersion matrix from {f_dij}")
        dij = np.loadtxt(f_dij, delimiter=",")  # pyright: ignore
        dij = jnp.array(dij)
    # Otherwise, generate dij
    else:
        if f_dij is None:
            f_dij = "Dij.csv"
        # dij = disp_canveg(setup, para, timemax=timemax, f_dij=f_dij)
        dij = disp_canveg(setup, para, timemax=timemax)
        # Save dij
        # print(f_dij)
        logging.info(f"Saving dispersion matrix to {f_dij}")
        np.savetxt(f_dij, np.array(dij), delimiter=",")
    return dij


# def conc_mx(
#     source: Float_2D,
#     soilflux: Float_1D,
#     delz: Float_1D,
#     dij: Float_2D,
#     met: Met,
#     cref: Float_1D,
#     prm: Para,
#     factor: Float_1D,
# ) -> Float_2D:
#     """Subroutine to compute scalar concentrations from source
#        estimates and the Lagrangian dispersion matrix.

#     Args:
#         source (Float_2D): _description_
#         soilflux (Float_2D): _description_
#         delz (Float_1D): _description_
#         Dij (Float_2D): _description_
#         met (Met): _description_
#         cref (Float_1D): _description_
#         prm (Para): _description_
#         factor (Float_1D): _description_

#     Returns:
#         Float_2D: _description_
#     """
#     ustar = met.ustar

#     # Compute concentration profiles from Dispersion matrix
#     ustar_ref = 1
#     ustfact = ustar_ref / ustar  #   ustar_ref /ustar
#     # ntime = ustfact.size

#     # @jax.vmap
#     # @partial(jax.vmap, )
#     def calculate_ccnc(ustfact_e, zL, source_e, soilflux_e, factor_e, cref_e):
#         # CC is the differential concentration (Ci-Cref)
#         # Ci-Cref = SUM (Dij S DELZ), units mg m-3 or mole m-3
#         # S = dfluxdz/DELZ
#         # note delz values cancel
#         # scale dispersion matrix according to friction velocity
#         disper = ustfact_e * dij[: prm.nlayers_atmos, :]
#         # disperzl=disper
#         # Updated dispersion matrix
#         disperzl = jax.lax.cond(
#             zL < 0,
#             lambda: disper * (0.973 * -0.7182) / (zL - 0.7182),
#             lambda: disper * (-0.31 * zL + 1.00),
#         )  # (nlayers_atmos, jktot)

#         # Compute cncc for all layers
#         # delz[:prm.jtot] * source_e[:prm.jtot] * disperzl[:prm.jtot,:]
#         sumcc = dot(
#             delz[: prm.jtot] * source_e[: prm.jtot],
#             disperzl.T[: prm.jtot, : prm.nlayers_atmos],
#         )
#         sumcc = sumcc.sum(axis=0)  # (nlayers_atmos,)

#         # scale dispersion matrix according to Z/L
#         dispersoil = ustfact_e * dij[: prm.nlayers_atmos, 0]  # (nlayers_atmos,)

#         # add soil flux to the lowest boundary condition
#         soilbnd = soilflux_e * dispersoil / factor_e  # (nlayers_atmos,)
#         cc = sumcc / factor_e + soilbnd  # (nlayers_atmos,)

#         # factor to adjust dij with alternative u* values
#         # compute scalar profile below reference
#         cncc_e = cc + cref_e - cc[-1]

#         return cncc_e

#     cncc = jax.vmap(calculate_ccnc, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)(
#         ustfact, met.zL, source, soilflux, factor, cref
#     )  # (ntime, nlayers_atmos)

#     return cncc


# def disp_canveg(prm: Para, rnd_seed: int=1234) -> Float_3D:
#     """_summary_

#     Args:
#         para (Para): _description_

#     Returns:
#         Float_3D: _description_
#     """
#     ht_atmos=prm.meas_ht-prm.veg_ht
#     dht_canopy=prm.dht_canopy
#     dht_atmos=prm.dht_atmos
#     nlayers_atmos=prm.nlayers_atmos

#     # consider look up tables for functional calls
#     zht=prm.zht  # heights of each layer
#     nlevel=prm.nlayers        # number of canopy layers plus one
#     npart = prm.npart         # Number of parcels 1000000
#     timemax = 1000.0          # time of parcel run
#     ustar = 1.00              #  friction velocity (m s-1)
#     HH = prm.veg_ht           #  canopy height (m)
#     DD=prm.dht
#     delta_z=prm.delz

#     upper_boundary = prm.meas_ht  # upper bound, ? times canopy height
#     ihigh=prm.nlayers_atmos
#     ihigh1 =ihigh + 1

#     sigma_h = 1.25      # sigma w > HH  */
#     sigma_zo = 0.25     # sigmaw over u* at z equal zero for exponential profile, Brunet 2020 BLM review  # noqa: E501
#     sigma_sur= sigma_zo * sigma_h * ustar  # sigma w at zero for linear profile
#     del_sigma = (sigma_h * ustar - sigma_sur) / HH # difference in sigma w

#     #######################################################
#     # Time step length (* TL) and Lagrangian Length scale #
#     #######################################################
#     fract=0.1  # was 0.1 for short crops
#     parcel_delta_t = fract * tl(HH, HH, DD, ustar)                         # time step
#     laglen = sigma_h * ustar * tl(HH, HH, DD, ustar)     # Lagrangian length scale */

#     # nn is the number of time increments that the particle travels
#     nn = jnp.floor((timemax / parcel_delta_t))

#     # pre-allocate random number and see if code is faster
#     key = jax.random.PRNGKey(rnd_seed)
#     rndij=jax.random.normal(key=key,shape=nn+1)

#     # parcel_sumn = 0
#     parcel_wwmean=jnp.zeros(ihigh1+1)
#     parcel_DIJ=jnp.zeros([ihigh,prm.jtot])

#     #######################################################
#     # number of particles per layer, redistribute npart #
#     # with height according to the relative source strength
#     #######################################################
#     nlev = jnp.ones(nlevel) * npart / nlevel
#     parcel_sumn = jnp.sum(nlev)

#     #######################################################
#     # Start the release of particles #
#     #######################################################
#     parcel_sum_w, parcel_move = 0., 1
#     timescale = tl(HH, HH, DD, ustar)

#     # Release of particles is carried out separately for each layer,
#     # starting with the lowest level.
#     # sigmaw = jnp.zeros(ihigh1)
#     sigmaw = sigma_w_zht(zht, HH, sigma_zo, del_sigma, sigma_h, ustar)
#     dvarwdz = dw2dz_w_zht(zht, HH, sigma_zo, del_sigma, ustar)
#     tldz = tl_dz(zht, HH, DD, ustar)

#     # 1-D case. Particles are released from each level z.  We have
#     # a continuous source and a horizontally homogeneous canopy.
#     def calculate_particle_level(c, ilevel):
#         parcel_consum = jnp.zeros(ihigh1)

#         # at each level NLEV(LEVEL) particles are released
#         def calculate_particle_nlev(c2, part):
#             rndij, parcel_consum = c2[0], c2[1]
#             parcel_sum_w = c2[2]
#             parcel_z = ilevel * prm.dht_canopy
#             IZF = jnp.min(jnp.fix(parcel_z/prm.dht_canopy + 1), ihigh)-1
#             parcel_w = sigmaw[IZF] * rndij[0]

#             # number of particle movements
#             parcel_move = parcel_move + 1
#             parcel_sum_w = parcel_sum_w+parcel_w

#             # The following should be a for loop
#             parcel_w = jnp.sign(parcel_z) * parcel_w
#             parcel_z = jnp.sign(parcel_z) * parcel_z
#             IZF = jnp.min(jnp.fix(parcel_z/prm.dht_canopy + 1), ihigh)-1
#             parcel_consum = parcel_consum.at[IZF].set(
#                parcel_consum.at[IZF] + parcel_delta_t
#             )
#             dtimescale=tldz[IZF]
#             dtT=parcel_delta_t / timescale

#             key = jax.random.PRNGKey(rnd_seed)
#             rndij=jax.random.normal(key=key,shape=nn+1)
#             c2new = [rndij, parcel_consum, parcel_sum_w]
#             return c2new, c2new

#         c2final, _ = jax.lax.scan(
#             calculate_particle_nlev, [rndij, parcel_consum, parcel_sum_w],
#             jnp.arange(nlev[ilevel])
#         )

#         parcel_consum, parcel_sum_w = c2final[1], c2final[2]
#         Dij_level = (parcel_consum[:ihigh]-parcel_consum[ihigh-1]) / \
#                     (delta_z[ilevel]*nlev[ilevel])
#         wwmean_level = jnp.mean(parcel_sum_w)
#         y = [Dij_level, wwmean_level]
#         return cnew, y
#     _, out = jax.lax.scan(
#         calculate_particle_level, , jnp.arange(nlevel)
#     )
#     parcel_Dij, parcel_wwmean = out[0], out[1]


# def tl(z: Float_0D, vht: Float_0D, dht: Float_0D, ustar: Float_0D) -> Float_0D:
#     """This function gives the variation of T(L) with height.

#     Args:
#         z (Float_0D): _description_
#         vht (Float_0D): _description_
#         dht (Float_0D): _description_
#         ustar (Float_0D): _description_

#     Returns:
#         Float_0D: _description_
#     """
#     y = jax.lax.cond(
#         z >= 1.5 * vht,
#         lambda:  0.4*(z-dht)/(1.56 * ustar),
#         lambda: 0.25*vht/ ustar
#     )
#     return y


# def sigma_w_zht(
#     z: Float_1D, hh: Float_0D, sigma_zo: Float_0D, del_sigma: Float_0D,
#     sigma_h: Float_0D, ustar: Float_0D
# ) -> Float_1D:
#     """This function gives the s(w) value for height z
#        Use linear decrease in sigma with z/h, as Wilson et al
#        show for a corn canopy.

#     Args:
#         z (Float_1D): _description_
#         hh (Float_0D): _description_
#         sigma_zo (Float_0D): _description_
#         del_sigma (Float_0D): _description_
#         sigma_h (Float_0D): _description_
#         ustar (Float_0D): _description_

#     Returns:
#         Float_1D: _description_
#     """
#     def calculate_each_y(c, z_each):
#         y_each = jax.lax.cond(
#             z_each < hh,
#             lambda: sigma_zo+z_each*del_sigma,
#             lambda: sigma_h * ustar
#         )
#         return c, y_each
#     _, y = jax.lax.scan(
#         calculate_each_y, init=None, xs=z
#     )
#     return y


# def dw2dz_w_zht(
#     z: Float_1D, hh: Float_0D, sigma_zo: Float_0D,
#     del_sigma: Float_0D, ustar: Float_0D
# ) -> Float_1D:
#     def calculate_each_y(c, z_each):
#         y_each = jax.lax.cond(
#             z_each < hh,
#          lambda: 2*z_each*del_sigma*del_sigma*ustar*ustar+2*sigma_zo*del_sigma*ustar,
#             lambda: 0.
#         )
#         return c, y_each
#     _, y = jax.lax.scan(
#         calculate_each_y, init=None, xs=z
#     )
#     return y


# def tl_dz(
#     z: Float_1D, hh: Float_0D, dd: Float_0D, ustar: Float_0D
# ) -> Float_1D:
#     def calculate_each_y(c, z_each):
#         y_each = jax.lax.cond(
#             z_each <= 1.5*hh,
#             lambda: 0.25*hh/ustar,
#             lambda: 0.4*(z_each-dd)/(1.56*ustar)
#         )
#         return c, y_each
#     _, y = jax.lax.scan(
#         calculate_each_y, init=None, xs=z
#     )
#     return y
