# Running JAX-CanVeg
Running JAX-CanVeg can be a complicated process as it involves a lot of model configurations and data preparations. To ease this process, we document the two key procedures to execute the model given a flux tower site.

## Step 1: Prepare the atmospheric forcings and ecosystem fluxes in CSV format
**Observation source:** Most atmospheric forcings and fluxes can be available from publicly available flux tower websites (e.g., [AmeriFlux](https://ameriflux.lbl.gov/)). The at-site leaf area index (LAI) data can be downloaded from [MODIS Fixed Sites Subsets Tool](https://modis.ornl.gov/sites/).

**Forcing csv file:** The forcing file is used to drive the model and should be in CSV format with the following column names: *year* (the year of observation), *doy* (the day of year), *hour* (the fractional hour), *ta* (air temperature [C]), *sw_in* (solar radiation [W m-2]), *eair* (ambient vapor pressure [kPa]), *ws* (wind speed [m s-1]), *co2* (ambient CO2 concentration [µmolCO2 mol-1]), *pa* (air pressure [hPa]), *ustar* (friction velocity [m s-1]), *ts* (soil temperature [C]), *swc* (volumetric soil water content [m3 m-3]), *veg_ht* (vegetation height [m]), *lai* (leaf area index [m2 m-2]). A sample of forcing data can be found [here](./data/fluxtower/US-Bi1/US-Bi1-forcings.csv).

**Flux csv file:** The flux file is used to validate/train/compare against the model and is thus **optional**. This CSV file contains the following column names: *P_mm* (precipitation [mm]), *LE* (latent heaf flux, or *LE_F_MDS*, [W M-2]), *H* (sensible heat flu, or *H_F_MDS*, [W M-2]), *G* (ground heat flux, or *G_F_MDS*, [W M-2]), *NETRAD* (net radiation, [W M-2]), *GPP* (gross primary production [µmolCO2 m-2 s-1]), *ALBEDO* (albedo [-]), *FCO2* (net ecosystem exchange, or *NEE_CUT_REF*, [µmolCO2 m-2 s-1]), *Rsoil* (soil respiratio [µmolCO2 m-2 s-1]). A sample of flux data can be found [here](./data/fluxtower/US-Bi1/US-Bi1-fluxes.csv).


## Step 2: Training JAX-CanVeg using a JSON-based configuration file
We suggest training JAX-CanVeg by providing a JSON-based configuration file, which include the following configurations (please see an example of US-Bi1 [here](./examples/US-Bi1/test-model/configs.json)):

- ```site name```: The name of the study site [string]

- ```model configurations```: JAX-CanVeg configurations including [dictionary]
    - ```time zone```: The time zone of the study site [int]
    - ```latitude```: The latitude of the study site [float]
    - ```longitude```: The longitude of the study site [float]
    - ```stomata type```: The stomata type with 0 for hypostomatous leaf and 1 for amphistomatous leaf [int]
    - ```leaf angle type```: The leaf angle type with 0 for planophile, 1 for spherical, 2 for erectophile, 3 for plagiophile, 4 for extremophile, and 5 for uniform leaf angle distributions [int]
    - ```leaf relative humidity module```: The leaf relative humidity module options with 0 process-based model and 1 for the DNN model [int]
    - ```soil respiration module```: The soil respiration module option with 0 for the alfalfa model (only for US-Bi1), 1 for Q10 power model, and 2 for the DNN model [int] 
    - ```canopy height```: The canopy height [m]
    - ```measurement height```: The flux tower height [m]
    - ```soil depth```: The soil depth [m]
    - ```number of canopy layers```: The number of canopy layers from surface to the top of canopy [int]
    - ```number of atmospheric layers```: The number of atmospheric layers from the canopy top to the flux tower sensors [int]
    - ```number of observed steps per day```: The number of observations every day [int]
    - ```number of solver iterations```: The number of iterations used in the fixed-point solver [int]
    - ```dispersion matrix```: the file path where the dispersion matrix locates (if exists) or will be saved (if not exists) [string]

- ```data```: The observed forcings and fluxes [dictionary]
    - ```training forcings```:
    - ```training fluxes```:
    - ```test forcings```:
    - ```test fluxes```:

- ```learning configurations```: The model training configurations [dictionary]
    - ```batch size```: The batch size during training [int]
    - ```number of epochs```: The number of training epochs [int]
    - ```output function```: The function for downselecting the model output for training, with ```canopy le``` for outputing the canopy LE only, ```canopy gpp``` for outputing the canopy GPP only, ```canopy nee``` for outputing the canopy NEE only, and ```canopy le nee``` for outputing both canopy LE and NEE (*more to be added*) [int]
    - ```output scaler```: The type of scaler for normalizing the output for training, with ```standard``` for the standard scaler, ```minmax``` for the minmax scaler, and ```null``` for the identify scaler [string]
    - ```loss function```: The type of loss function, with ```mse``` for mean squared error, ```mspe` for mean square percentage error, ```relative_mse``` for relative mean squared error [string]
    - ```tunable parameters```: The list of the parameters to be tuned where the full list can be found [here](./src/jax_canveg/subjects/parameters.py#L116) [list]
    - ```optimizer```: The optimizer [dictionary]
        - ```type```: The type of optimizer available from [optax](https://optax.readthedocs.io/en/latest/api/optimizers.html), currently only supporting ```Adam``` and ```Adamw```[string]
        - ```args```: The arguments of the selected optax optimizer in dictionary [dictionary]
        - ```learning_scheduler```: The [optax opimizer scheduler](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html) [dictionary]
            - ```type```: the type of optax opimizer scheduler](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html), currently only supporting ```constant``` and ```piecewise constant``` [string]
            - ```args```: The arguments of the selected optax optimizer in dictionary [dictionary]


- ```saving configurations```: The saving configurations [dictionary]
    - ```new model```: The file path where the trained model is saved [string]
    - ```loss values```: The file path where the loss values are saved [string]

Given the configuration file, one can train the model by calling the [train_model](./src/jax_canveg/train_model.py#L51) function. The corresponding example of training US-Bi1 test model can be found in this [file](./examples/US-Bi1/train_testmodel.py).

## Step 3: Run the trained model
Once the model is trained. one can run the model by executing the following code sample (see a more complicated version [here](./examples/US-Bi1/postprocessing.py)).
```python
from jax_canveg import load_model

# Configuration file location
f_config = "./examples/US-Bi1/test-model/configs.json"

 # Load the model, forcings, and observations
model, met_train, met_test, obs_train, obs_test = load_model(f_configs)

# Run the model on both training and test datasets
states_train, drivers_train = model(met_train)
states_test, drivers_test = model(met_test)

# Get the canopy simualtion
can_train, can_test = states_train[-1], states_test[-1]
```