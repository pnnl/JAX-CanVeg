"""
One-dimensional hydrobiogeochemical modeling of CANOAK/Canveg.
Modified from CANOAK's matlab code.

Author: Peishi Jiang
Date: 2023.07.24.
"""

# import jax
# import jax.numpy as jnp
import pandas as pd

# import h5py
# import numpy as np

from jax_canoak.subjects import Para, Met
from jax_canoak.physics.energy_fluxes import disp_canveg


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
    n_time=200,
)


# ---------------------------------------------------------------------------- #
#                     Get the model forcings                                   #
# ---------------------------------------------------------------------------- #
f_forcing = ""  # TODO
forcing_data = pd.read_csv(f_forcing).values
met = Met(forcing_data, para)
para.set_lai(met.lai)


# ---------------------------------------------------------------------------- #
#                     Generate or read the Dispersion matrix                   #
# ---------------------------------------------------------------------------- #
dij = disp_canveg(para)


# ---------------------------------------------------------------------------- #
#                     ??? Initialize model states ?????                        #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                     Compute sun angles                                       #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                     Compute direct and diffuse radiations                    #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                     Compute leaf angle                                       #
# ---------------------------------------------------------------------------- #


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
