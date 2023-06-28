import jax
import jax.numpy as jnp

from .types import Float_0D, Float_1D, Float_2D

# Subroutine to compute scalar concentrations from source estimates
# and the Lagrangian dispersion matrix
def conc(
    cref: Float_0D,
    soilflux: Float_0D,
    factor: Float_0D,
    # sze3: int, jtot: int, jtot3: int,
    met_zl: Float_0D,
    delz: Float_0D,
    izref: int,
    ustar_ref: Float_0D,
    ustar: Float_0D,
    source: Float_1D,
    dispersion: Float_2D,
) -> Float_1D:

    jtot3, jtot = dispersion.shape

    # Compute concentration profiles from Dispersion matrix
    ustfact = ustar_ref / ustar  # factor to adjust Dij with alternative u* values
    # Note that dispersion matrix was computed using u* = 1.00

    def compute_disperzl(disper):
        disperzl = jax.lax.cond(
            met_zl < 0,
            lambda x: x[0] * (0.97 * -0.7182) / (x[1] - 0.7182),
            lambda x: x[0] * (-0.31 * x[1] + 1.00),
            [disper, met_zl],
        )
        return disperzl

    # Calculate disperzl matrix
    disper = ustfact * dispersion
    disperzl = jnp.vectorize(compute_disperzl)(disper)

    # Calculate sumcc
    sumcc = jnp.sum(delz * disperzl * source, axis=1)  # type: ignore[code]

    # Calculate cc
    def compute_each_cc(carry, i):
        # Scale dispersion matrix according to Z/L
        disper = ustfact * dispersion[i, 0]
        disperzl = compute_disperzl(disper)
        # Add soil flux to the lowest boundary condition
        soilbnd = soilflux * disperzl / factor
        each_cc = sumcc[i] / factor + soilbnd
        return carry, each_cc

    _, cc = jax.lax.scan(f=compute_each_cc, init=None, xs=jnp.arange(jtot3))

    # Compute scalar profile below reference
    def compute_each_cncc(carry, i):
        each_cncc = cc[i] + cref - cc[izref]
        return carry, each_cncc

    _, cncc = jax.lax.scan(f=compute_each_cncc, init=None, xs=jnp.arange(jtot3))

    return cncc


# # Subroutine to compute scalar concentrations from source estimates
# # and the Lagrangian dispersion matrix
# def conc(
#     cref, soilflux, factor,
#     sze3, jtot, jtot3, met_zl, delz, izref,
#     ustar_ref, ustar,
#     source, dispersion
# ):

#     sumcc = jnp.zeros(sze3)
#     cc = jnp.zeros(sze3)
#     cncc = jnp.zeros(jtot3)

#     # Compute concentration profiles from Dispersion matrix
#     ustfact = ustar_ref / ustar  # factor to adjust Dij with alternative u* values
#     # Note that dispersion matrix was computed using u* = 1.00

#     for i in range(jtot3):

#         for j in range(jtot):
#             # CC is the differential concentration (Ci-Cref)
#             # Ci-Cref = SUM (Dij * S * DELZ), units mg m-3 or mole m-3
#             # S = dfluxdz / DELZ
#             # Note delz values cancel
#             # Scale dispersion matrix according to friction velocity

#             # disper = ustfact * met.dispersion[i][j]  # units s/m
#             disper = ustfact * dispersion[i,j]  # units s/m

#             # Updated Dispersion matrix (Oct, 2015) for alfalfa runs
#             # for a variety of z?
#             if met_zl < 0:
#                 disperzl = disper * (0.97 * -0.7182) / (met_zl - 0.7182)
#             else:
#                 disperzl = disper * (-0.31 * met_zl + 1.00)

#             # sumcc = jax.ops.index_add(sumcc, i, delz * disperzl * source[j])
#             sumcc = sumcc.at[i].add(delz * disperzl * source[j])
#             # sumcc[i] += delz * disperzl * source[j]

#         # Scale dispersion matrix according to Z/L
#         disper = ustfact * dispersion[i,0]

#         if met_zl < 0:
#             disperzl = disper * (0.97 * -0.7182) / (met_zl - 0.7182)
#         else:
#             disperzl = disper * (-0.31 * met_zl + 1.00)

#         # Add soil flux to the lowest boundary condition
#         soilbnd = soilflux * disperzl / factor
#         # cc = jax.ops.index_update(cc, i, sumcc[i] / factor + soilbnd)
#         cc = cc.at[i].set(sumcc[i] / factor + soilbnd)
#         # cc[i] = sumcc[i] / factor + soilbnd
#         # print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
#         #     cc[i], sumcc[i], factor, disper, dispersion[i,0]))

#     # Compute scalar profile below reference
#     for i in range(jtot3):
#         # cncc[i] = cc[i] + cref - cc[izref]
#         cncc = cncc.at[i].set(cc[i] + cref - cc[izref])
#         # print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(
#         #     cncc[i], cc[i], cref, cc[izref], sumcc[i]))

#     return cncc
