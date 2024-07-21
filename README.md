# JAX-CanVeg: A Differentiable Land Surface Model

## Introduction
Land surface process describes the water, energy, and carbon cycles exchanged among the atmosphere, canopy, and soil. Its complex interacting nature makes it challenging to model due to the associated unknown biophysical and ecophysiological parameters and less-mechanistically represented subprocesses. Differentiable modeling provides a new opportunity to explore the parameter space and capture these complex interactions by seamlessly coupling process-based and deep learning models. Here, we developed a differentiable land surface model by reimplementing an existing simulator, CanVeg, in JAX -- a Google-developed Python package for high-performance machine learning research using automatic differentiation. We applied JAX-CanVeg to simulate the ecohydrological fluxes at two flux tower sites with varying aridity. We implemented a hybrid version of the Ball-Berry equation in JAX-CanVeg to improve the limited capability of the equation in accounting for the influence of water stress on stomatal closure. The hybrid model emulates the impact of water stress on stomatal conductance calculation through a deep neural network parameterized on the observed soil water content. Trained against the latent heat flux observations, the hybrid model improves the water flux simulation over the pure process-based model at both sites, owing to a better delineation of stomatal conductance response to soil moisture. The updated calculation of stomatal conductance further alters the model prediction on canopy carbon fluxes, such as photosynthesis. Our study showcases a new avenue for modeling land-atmospheric interactions by leveraging the benefits of both data-driven learning and process-based modeling.

## References
Jiang, P. et al., (2024). JAX-CanVeg: A Differentiable Land Surface Model. Water Resources Research, *in review*.

## Key data/folder Structure
```
.
+-- src/jax_canveg
+-- examples
+-- data
+-- environment.yml
+-- README.md
```
- `src/jax_canveg`: providing source codes for JAX-CanVeg.
- `examples`: providing example notebooks for running JAX-CanVeg at selected flux tower sites (e.g., US-Bi1 and US-Hn1).
- `data`: providing the observation data (including both flux tower and MODIS) and the simulated lagrangian particles.
- `environment.yml`: the YAML file for creating the conda virtual environment (see Step 1 in the Installation section).
- `README.md`: the readme file.

## Installation
1. Create the conda virtual environment:
```
conda env create -f environment.yml
```

2. Activate the virtual environment:
```
conda activate jax-canveg
```

3. [Install JAX](https://github.com/google/jax#installation) either by (for CPU only)
```
pip install --upgrade "jax[cpu]"
```
, or by (for GPU support)
```
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Install other packages that are only available under pip:
```
pip install equinox diffrax optax pre-commit optimistix lineax hydroeval pyproj
pip install -U scikit-learn
pip3 install torch torchvision torchaudio
```

5. Compile [the C++ code](./src/jax_canveg/physics/energy_fluxes/DispersionMatrix.cpp) with [pybind11](https://github.com/pybind/pybind11) for generating dispersion matrix (make sure you have a suitable compiler installed):
```
cd ./src/jax_canveg/physics/energy_fluxes/

# For Unix
g++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -fPIC $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)

# For MacOS
c++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -undefined dynamic_lookup $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)
```

6. Add the path of the source code [src](./src) into the environment variable `PYTHONPATH`.


## Examples
We provides the codes/notebooks for reproducing the two examples shown in the Jiang et al. (2024) under the folder `examples`. We applied the differentiable JAX-CanVeg at US-Hn1 and US-Bi1 flux tower sites to evaluate the performance of a hybrid version of the Ball-Berry equation. The model were trained against the observed latent heat fluxes. Below we illustrate the application example on [US-Hn1](./examples/US-Hn1) (which is applicable to [US-Bi1](./examples/US-Bi1)).

- Step 1: Preprocess the flux tower and MODIS observations to CSV files (using [US-Hn1.ipynb](./examples/US-Hn1/US-Hn1.ipynb))
- Step 2: Run the default JAX-CanVeg (without training; using [canveg-Hn1_default.ipynb](./examples/US-Hn1/canveg-Hn1_default.ipynb))
- Step 3: Train the process-based JAX-CanVeg (using [canveg-Hn1-purephysics.ipynb](./examples/US-Hn1/canveg-Hn1-purephysics.ipynb))
- Step 4: Train the hybrid JAX-CanVeg (using [canveg-Hn1-hybrid.ipynb](./examples/US-Hn1/canveg-Hn1-hybrid.ipynb))
- Step 5: Train the pure DNN model (using [dnn_US-Hn1.ipynb](./examples/US-Hn1/dnn_US-Hn1.ipynb))
- Step 6: Postprocess the modeling results (using [postprocess.ipynb](./examples/US-Hn1/postprocess.ipynb))

We performed computation time benchmark by running both JAX-CanVeg and [CanVeg-matlabl] --(https://github.com/baldocchi/CanVeg)
- Run [examples/US-Hn1/calculate_execution_time.ipynb](./examples/US-Hn1/calculate_execution_time.ipynb) to calculate the computation time of the US-Hn1 example
- Run [examples/US-Bi1/calculate_execution_time.ipynb](./examples/US-Bi1/calculate_execution_time.ipynb) to calculate the computation time of the US-Bi1 example
- Run [CanVeg-matlab](https://github.com/baldocchi/CanVeg) as follows:
    - Download [CanVeg-matlab](https://github.com/baldocchi/CanVeg)
    - Copy the matlab file [Canveg_pj.m](./examples/Canveg_pj.m) to [CanVeg-matlab](https://github.com/baldocchi/CanVeg)
    - Copy the Dij files [Dij_US-Bi1.csv](./data/dij/Dij_US-Bi1.csv) and [Dij_US-Hn1.csv](./data/dij/Dij_US-Hn1.csv) to [CanVeg-matlab](https://github.com/baldocchi/CanVeg)
    - Copy the forcing files [US-Bi1-forcings.csv](./data/fluxtower/US-Bi1/US-Bi1-forcings.csv) and [US-Hn1-forcings-v2.csv](./data/fluxtower/US-Hn1/US-Hn1-forcings-v2.csv) to [CanVeg-matlab](https://github.com/baldocchi/CanVeg)
    - Run [Canveg_pj.m](./examples/Canveg_pj.m) and the computation time will be printed to the screen
    - Rerun [Canveg_pj.m](./examples/Canveg_pj.m) on another site by changing the variable `Site` to `US-Hn1` or `US-Bi1` in the file (Line 95)
- Plot the executation time using [examples/plot_time_difference.ipynb](./examples/plot_time_difference.ipynb)

## License
See LICENSE for more information.

## Contact
Peishi Jiang (peishi.jiang@pnnl.gov)