# Benchmark the computational time of JAX-CanVeg against the legacy CanVeg
We performed computation time benchmark by running both JAX-CanVeg and [CanVeg-matlabl] -- (https://github.com/baldocchi/CanVeg) at the four study sites.

## Calculate the computational time of running the legacy CanVeg
Run [CanVeg-matlab](https://github.com/baldocchi/CanVeg) as follows:
1. Download [CanVeg-matlab](https://github.com/baldocchi/CanVeg);
2. Copy the matlab file [Canveg_pj.m](./examples/Canveg_pj.m) to [CanVeg-matlab](https://github.com/baldocchi/CanVeg);
3. Copy the Dij files [Dij_US-Bi1_50L.csv](./data/dij/Dij_US-Bi1_50L.csv), [Dij_US-Hn1_50L.csv](./data/dij/Dij_US-Hn1_50L.csv), [Dij_US-Me2_50L.csv](./data/dij/Dij_US-Me2_50L.csv), and [Dij_US-Whs_50L.csv](./data/dij/Dij_US-Whs_50L.csv) to the folder of CanVeg-matlab;
4. Copy the forcing files [US-Bi1-forcings.txt](./data/fluxtower/US-Bi1/US-Bi1-forcings.txt), [US-Hn1-forcings.txt](./data/fluxtower/US-Hn1/US-Hn1-forcings.txt), [US-Me2-forcings.txt](./data/fluxtower/US-Me2/US-Me2-forcings.txt), and [US-Whs-forcings.txt](./data/fluxtower/US-Whs/US-Whs-forcings.txt) to the folder of CanVeg-matlab;
5. Run [Canveg_pj.m](./examples/Canveg_pj.m) and the computation time will be printed to the screen.
6. Rerun [Canveg_pj.m](./examples/Canveg_pj.m) on another site by changing the variable `Site` to `US-Hn1`, `US-Bi1`, `US-Me2`, or `US-Whs` in the file (Lines 94-97)

> [!NOTE]
> The computational time of running CanVeg will be printed to the screen in the matlabl console.


## Calculate the computational time of running JAX-CanVeg
1. Calculate the computational time of running JAX-CanVeg on CPU
```sh
cd [jax-canveg-folder]/examples
python calculate_computingtime.py cpu
```

2. Calculate the computational time of running JAX-CanVeg on GPU
```sh
cd [jax-canveg-folder]/examples
python calculate_computingtime.py cuda
```

> [!NOTE]
> The computational time of running JAX-CanVeg will be saved into two logging text files, named:
> computation_time-cpu.log and computation_time-cuda.og