# JAX-WATERSHED: Watershed Modeling in JAX

## (TODO) Introduction
A differentiable land surface model using [JAX](https://github.com/google/jax).

## Installation
1. Create the conda virtual environment:
```
conda env create -f environment.yml
```

2. Activate the virtual environment:
```
conda activate jax-watershed
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
pip install equinox diffrax optax pre-commit
```

5. Compile the C++ code for generating dispersion matrix (make sure you have a suitable compiler installed):
```
cd ./src/jax_canoak/physics/energy_fluxes/

# For unix
g++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -fPIC $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)

# For MacOS
c++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -undefined dynamic_lookup $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)
```

## (TODO) Examples

## Contact:
Peishi Jiang (peishi.jiang@pnnl.gov)