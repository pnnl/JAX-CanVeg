# Installation of JAX-CanVeg

1. Close the repository
```
git clone https://github.com/pnnl/JAX-CanVeg/tree/main
```

2. Create the conda virtual environment:
```
conda env create -f environment.yml
```

3. Activate the virtual environment:
```
conda activate jax-canveg
```

4. [Install JAX](https://github.com/google/jax#installation) either by (for CPU only)
```sh
pip install --upgrade "jax[cpu]"
```
, or by (for GPU support)
```sh
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

5. Install other packages that are only available under pip:
```sh
pip install equinox diffrax optax pre-commit optimistix lineax hydroeval pyproj
pip install -U scikit-learn
pip3 install torch torchvision torchaudio
```

6. Compile [the C++ code](./src/jax_canveg/physics/energy_fluxes/DispersionMatrix.cpp) with [pybind11](https://github.com/pybind/pybind11) for generating dispersion matrix (make sure you have a suitable compiler installed):
```sh
cd ./src/jax_canveg/physics/energy_fluxes/

# For Unix
g++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -fPIC $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)

# For MacOS
c++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -undefined dynamic_lookup $(python3 -m pybind11 --includes) DispersionMatrix.cpp -o dispersion$(python3-config --extension-suffix)
```

7. Add the path of the source code [src](./src) into the environment variable `PYTHONPATH`.