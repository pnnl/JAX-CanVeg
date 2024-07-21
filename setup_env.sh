#!/bin/bash

# Set up the virtual environment
conda env create -f environment.yml

# Activate the environment
conda activate jax-canoak

wait

# Install JAX using pip
pip install --upgrade "jax[cpu]"
pip install optimistix
pip install diffrax
pip install lineax
pip install optax
pip install pre-commit
pip install pyproj
pip install -U scikit-learn
pip install hydroeval
pip3 install torch torchvision torchaudio
# pip install equinox