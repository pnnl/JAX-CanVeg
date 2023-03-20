"""
A list of constants here.
"""

RADIANS = .017453293
PI      = 3.141592653589793
TWOPI   = PI*2

# Physical constants
PO                        = 101.325 # sea level pressure [kPa]
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8 # [W m-2 K-4]
C_TO_K                    = 273.15 # conversion from degC to degK
G                         = 9.81 # the gravitational acceleration [m s-2]
VON_KARMAN_CONSTANT       = 0.41
κ                         = 1.38065e-23 # Boltzmann constant [J K-1 molecule-1]
NA                        = 6.02214e26 # Avogadro's number [molecule kmol-1]
MW_DA                     = 28.966 # Molecular weight of dry air [kg kmol-1]
R_GAS                     = NA * κ # Universal gas constant [J K-1 kmol-1]
R_DA                      = R_GAS / MW_DA # dry air gas constant [J K-1 kg-1]
C_P                       = 1.00464e3 # Specific heat capacity of dry air [J kg-1 K-1]
λ_VAP                     = 2.501e6 # latent heat of vaporization [J kg-1]

# Plant functional type optical properties (from Table 3.1 in CLM5)
pft_clm5 = ["NET Temperature", "NET Boreal", "NDT Boreal", "BET Tropical", "BET temperate", "BDT tropical", "BDT temperate", "BDT boreal",
            "BES temperate", "BDS temperate", "BDS boreal", "C3 arctic grass", "C3 grass", "C4 grass", "C3 Crop",
            "Temp Corn", "Spring Wheat", "Temp Soybean", "Cotton", "Rice", "Sugarcane", "Tropical Corn", "Tropical Soybean"]
χl_clm5 = [0.01]*3 + [0.1]*2 +[0.01, 0.25, 0.25, 0.01, 0.25, 0.25] + [-0.3]*4 + [-0.5]*8
α_leaf_clm5 = {
    "PAR": [0.7]*3 + [0.1]*5 + [0.7, 0.1, 0.1] + [0.11]*12,
    "NIR": [0.35]*3 + [0.45]*5 + [0.35, 0.45, 0.45] + [0.35]*12
}
α_stem_clm5 = {
    "PAR": [0.16]*11 + [0.31]*12,
    "NIR": [0.39]*11 + [0.53]*12
}
τ_leaf_clm5 = {
    "PAR": [0.05]*23,
    "NIR": [0.1]*3 + [0.25]*5 + [0.1,0.25,0.25] + [0.34]*12
}
τ_stem_clm5 = {
    "PAR": [0.001]*11 + [0.12]*12,
    "NIR": [0.001]*11 + [0.25]*12
}

# Intercepted snow optical properties (from Table 3.2 in CLM5)
ω_snow_clm5  = {"PAR": 0.8, "NIR": 0.4}
β_snow_clm5  = {"PAR": 0.5, "NIR": 0.5}
β0_snow_clm5 = {"PAR": 0.5, "NIR": 0.5}

# Soil albedos
α_soil_clm5 = {"PAR": 0.6, "NIR": 0.4}

# Soil and snow emissivities
ε_soil_clm5, ε_snow_clm5 = 0.96, 0.97

# TODO: Dry and saturated soil albedos (from Table 3.3 in CLM5)


# Spectral bands and weights used for snow radiative transfer (from table 3.4 in CLM5)