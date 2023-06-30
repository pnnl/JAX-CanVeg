#!/bin/bash

# Set up the virtual environment
conda env create -f environment.yml

# Activate the environment
conda activate jax-watershed

# Install JAX using pip
pip install --upgrade "jax[cpu]"
pip install diffrax
pip install pre-commit
# pip install equinox