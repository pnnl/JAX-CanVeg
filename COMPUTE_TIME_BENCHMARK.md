# Benchmark the computational time of JAX-CanVeg against the legacy CanVeg
We performed computation time benchmark by running both JAX-CanVeg and [CanVeg-matlabl] -- (https://github.com/baldocchi/CanVeg) at the four study sites.

## Calculate the computational time of running the legacy CanVeg
- Run [examples/US-Hn1/calculate_execution_time.ipynb](./examples/US-Hn1/calculate_execution_time.ipynb) to calculate the computation time of the US-Hn1 example
- Run [examples/US-Bi1/calculate_execution_time.ipynb](./examples/US-Bi1/calculate_execution_time.ipynb) to calculate the computation time of the US-Bi1 example
- Run [CanVeg-matlab](https://github.com/baldocchi/CanVeg) as follows:
    - Download [CanVeg-matlab](https://github.com/baldocchi/CanVeg);
    - Copy the matlab file [Canveg_pj.m](./examples/Canveg_pj.m) to [CanVeg-matlab](https://github.com/baldocchi/CanVeg);
    - Copy the Dij files [Dij_US-Bi1_50L.csv](./data/dij/Dij_US-Bi1_50L.csv), [Dij_US-Hn1_50L.csv](./data/dij/Dij_US-Hn1_50L.csv), [Dij_US-Me2_50L.csv](./data/dij/Dij_US-Me2_50L.csv), and [Dij_US-Whs_50L.csv](./data/dij/Dij_US-Whs_50L.csv) to the folder of CanVeg-matlab;
    - Copy the forcing files [US-Bi1-forcings.txt](./data/fluxtower/US-Bi1/US-Bi1-forcings.txt), [US-Hn1-forcings.txt](./data/fluxtower/US-Hn1/US-Hn1-forcings.txt), [US-Me2-forcings.txt](./data/fluxtower/US-Me2/US-Me2-forcings.txt), and [US-Whs-forcings.txt](./data/fluxtower/US-Whs/US-Whs-forcings.txt) to the folder of CanVeg-matlab;
    - Run [Canveg_pj.m](./examples/Canveg_pj.m) and the computation time will be printed to the screen.
    - Rerun [Canveg_pj.m](./examples/Canveg_pj.m) on another site by changing the variable `Site` to `US-Hn1`, `US-Bi1`, `US-Me2`, or `US-Whs` in the file (Lines 94-97)
- Plot the executation time using [examples/plot_time_difference.ipynb](./examples/plot_time_difference.ipynb)


## (TODO) Calculate the computational time of running JAX-CanVeg

## Plot the time difference
Plot the executation time using [examples/plot_time_difference.ipynb](./examples/plot_time_difference.ipynb).