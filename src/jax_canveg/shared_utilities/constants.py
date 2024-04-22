from math import pow

PI = 3.14159
PI180 = 0.017453292  # pi divided by 180, radians per degree
PI9 = 2.864788976
PI2 = 6.283185307  # 2 time pi

# # Canopy structure variables
# ht = 1.0  # Canopy height, m
# pai = 0.0  # Plant area index
# lai = 4.0  # Leaf area index

vcopt = 170.0  # Carboxylation rate at optimal temperature, umol m-2 s-1
jmopt = 278.0  # Electron transport rate at optimal temperature, umol m-2 s-1
rd25 = 0.22  # Dark respiration at 25 C, rd25= 0.34 umol m-2 s-1

# jtot = 30             # Number of canopy layers
# jktot = 31            # jtot + 1
# jtot3 = 150           # Number of layers in the domain, three times canopy height
# izref = 150           # Array value of reference ht at 2.8 m, jtot/ht
# delz = 0.0175         # Height of each layer, ht/jtot
# zh65 = 0.3575         # 0.65/ht

pi4 = 12.5663706

ustar_ref = 1.0  # Reference u* value for dispersion matrix

# Universal gas constant
rugc = 8.314  # Universal gas constant, J mole-1 K-1
rgc1000 = 8314.0  # Gas constant times 1000.

# Consts for Photosynthesis model and kinetic equations.
# for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)
hkin = 200000.0  # Enthalpy term, J mol-1
skin = 710.0  # Entropy term, J K-1 mol-1
ejm = 55000.0  # Activation energy for electron transport, J mol-1
evc = 55000.0  # Activation energy for carboxylation, J mol-1

# Enzyme constants & partial pressure of O2 and CO2
# Michaelis-Menten K values. From survey of literature.
kc25 = 274.6  # Kinetic coefficient for CO2 at 25 C, microbars
ko25 = 419.8  # Kinetic coefficient for O2 at 25C, millibars

o2 = 210.0  # Oxygen concentration, mmol mol-1

# tau is computed on the basis of the Specificity factor (102.33)
# times Kco2/Kh2o (28.38) to convert for value in solution
# to that based in air/
# The old value was 2321.1.
# New value for Quercus robor from Balaguer et al. 1996
# Similar number from Dreyer et al. 2001, Tree Physiol, tau= 2710
tau25 = 2904.12  # Tau coefficient

# Arrhenius constants
# Eact for Michaelis-Menten const. for KC, KO and dark respiration
# These values are from Harley
ekc = 80500.0  # Activation energy for K of CO2, J mol-1
eko = 14500.0  # Activation energy for K of O2, J mol-1
erd = 38000.0  # Activation energy for dark respiration, eg Q10=2
ektau = -29000.0  # J mol-1 (Jordan and Ogren, 1984)
tk_25 = 298.16  # Absolute temperature at 25 C
toptvc = 298.0  # Optimum temperature for maximum carboxylation
toptjm = 298.0  # Optimum temperature for maximum electron transport
eabole = 45162  # Activation energy for bole respiration for Q10 = 2.02

# Constants for leaf energy balance
sigma = 5.67e-08  # Stefan-Boltzmann constant, W M-2 K-4
cp = 1005.0  # Specific heat of air, J KG-1 K-1
mass_air = 29.0  # Molecular weight of air, g mole-1
mass_CO2 = 44.0  # Molecular weight of CO2, g mole-1
dldt = -2370.0  # Derivative of the latent heat of vaporization

ep = 0.98  # Emissivity of leaves
epm1 = 0.02  # 1 - ep
epsoil = 0.98  # Emissivity of soil
epsigma = 5.5566e-8  # ep * sigma
epsigma2 = 11.1132e-8  # 2 * ep * sigma
epsigma4 = 22.2264e-8  # 4.0 * ep * sigma
epsigma6 = 33.3396e-8  # 6.0 * ep * sigma
epsigma8 = 44.448e-8  # 8.0 * ep * sigma
epsigma12 = 66.6792e-8  # 12.0 * ep * sigma

betfact = 1.5  # Multiplication factor for aerodynamic sheltering

# Constants for the polynomial equation for saturation vapor pressure
# -T function, es=f(t)
a1en = 617.4  # Polynomial coefficient
a2en = 42.22  # Polynomial coefficient
a3en = 1.675  # Polynomial coefficient
a4en = 0.01408  # Polynomial coefficient
a5en = 0.0005818  # Polynomial coefficient

# Ball-Berry stomatal coefficient for stomatal conductance
kball = 9.5  # Ball-Berry stomatal coefficient for stomatal conductance

# Intercept of Ball-Berry model, mol m-2 s-1
bprime = 0.0175  # Intercept for H2O
bprime16 = 0.0109375  # Intercept for CO2

# Minimum stomatal resistance, s m-1
rsm = 145.0  # Minimum stomatal resistance, s m-1
brs = 60.0  # Curvature coefficient for light response

# Leaf quantum yield, electrons
qalpha = 0.22  # Leaf quantum yield, electrons
qalpha2 = 0.0484  # qalpha squared

# Leaf clumping factor
markov = 1.0  # Leaf clumping factor

# Leaf dimension. geometric mean of length and width (m)
lleaf = 0.02  # Leaf dimension, geometric mean of length and width, m

# Diffusivity values for 273 K and 1013 mb (STP) using values from
# Massman (1998) Atmos Environment
# These values are for diffusion in air.  When used these values must be adjusted for
# temperature and pressure
# nu, Molecular viscosity
nuvisc = 13.27  # Molecular viscosity, mm2 s-1
nnu = 0.00001327  # m2 s-1

# Diffusivity of CO2
dc = 13.81  # Diffusivity of CO2, mm2 s-1
ddc = 0.00001381  # m2 s-1

# Diffusivity of heat
dh = 18.69  # Diffusivity of heat, mm2 s-1
ddh = 0.00001869  # m2 s-1

# Diffusivity of water vapor
dv = 21.78  # Diffusivity of water vapor, mm2 s-1
ddv = 0.00002178  # m2 s-1

# Diffusivity of ozone
do3 = 14.44  # Diffusivity of ozone, mm2 s-1
ddo3 = 0.00001444  # m2 s-1

# Isotope ratio of PeeDee Belimdite standard (PDB)
# redefined as 13C/(12C+13C) for PDB, Tans et al 93
Rpdb_CO2 = 0.011115  # Isotope ratio of PeeDee Belimdite standard (PDB) for CO2

# Isotope ratio of PeeDee Belimdite standard (PDB)
# defined as 13C/12C, from Farquhar
Rpdb_12C = 0.01124  # Isotope ratio of PeeDee Belimdite standard (PDB) for 12C


# --------------------- Some dimensionless constants ---------------------
# Constants for leaf boundary layers
lfddf = lleaf / ddh

# Prandtl Number
pr = nuvisc / dh
pr33 = pow(pr, 0.33)

# DIFFUSIVITY OF WATER VAPOR, m2 s-1
lfddv = lleaf / ddv

# SCHMIDT NUMBER FOR VAPOR
sc = nuvisc / dv
sc33 = pow(sc, 0.33)

# SCHMIDT NUMBER FOR CO2
scc = nuvisc / dc
scc33 = pow(scc, 0.33)

# Grasshof Number
grasshof = 9.8 * pow(lleaf, 3) / pow(nnu, 2)
