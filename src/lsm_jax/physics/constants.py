"""Let's define some physics constants here."""

# PI
PI = 3.141592653589793

# Conversion from mm to m
MM_TO_M = 1e-3

# Conversion from seconds to day
SECONDS_TO_DAY = 1./86400.

# Conversion from W (i.e., J s-1) to MJ day-1
W_TO_MJD = 1e-6 / SECONDS_TO_DAY

# Conversion from Celsius to Kelvin
C_TO_K = 273.15

# Stephan-Boltzmann constant
# BOLTZMANN_CONSTANT = 5.67e-8 # [W m-2 K-4]
BOLTZMANN_CONSTANT = 5.67e-8 * W_TO_MJD # [MJ m-2 day-1 K-4]

# The latent heat of vaporization
L  = 2.45     # [MJ kg-1]

# Atmospheric pressure
P_AIR   = 101.3    # [kPa]

# Mean desity of air at 20 degC
RHO_AIR = 1.2      # [kg m-3]

# Mean desity of air at 20 degC
RHO_WATER = 1e3      # [kg m-3]

# Specific heat of air
C_AIR   = 0.001013 # [MJ kg-1 K-1]

# von Karman constant
VONKARMAN_CONSTANT   = 0.41 # [-]
