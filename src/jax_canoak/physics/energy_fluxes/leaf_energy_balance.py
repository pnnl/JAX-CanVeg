"""
Leaf energy balance subroutines and runctions, including:
- compute_qin()
- leaf_energy()

Author: Peishi Jiang
Date: 2023.08.01.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

from ...subjects import ParNir, Ir, Para, Qin, BoundLayerRes
from ...subjects import Met, SunShadedCan, Prof
from ...subjects.utils import desdt as fdesdt
from ...subjects.utils import des2dt as fdes2dt
from ...subjects.utils import es as fes
from ...subjects.utils import llambda as fllambda

from ...shared_utilities.utils import dot

# from typing import Tuple

from ...shared_utilities.types import Float_2D


@eqx.filter_jit
def compute_qin(quantum: ParNir, nir: ParNir, ir: Ir, prm: Para, qin: Qin) -> Qin:
    """Available energy on leaves for evaporation.
       Values are average of top and bottom levels of a layer.
       The index refers to the layer.  So layer 3 is the average
       of fluxes at level 3 and 4.  Level 1 is soil and level
       j+1 is the top of the canopy. layer jtot is the top layer

       Sunlit and shaded Absorbed Energy on the Top and Bottom of Leaves

    Args:
        quantum (ParNir): _description_
        nir (ParNir): _description_
        ir (Ir): _description_
        prm (Para): _description_

    Returns:
        Qin: _description_
    """
    # convert umol m-2 s-1 PPFD to W m-2 in visible
    vis_sun_abs = quantum.sun_abs / 4.6
    vis_sh_abs = quantum.sh_abs / 4.6

    ir_shade = ir.shade * prm.ep
    # ir_shade = ir.shade

    # noticed in code ir.shade was multiplied by prm.ep twice, eg in
    # fIR_RadTranCanopy_MatrixV2 and here. removed prm.ep in the IR
    # subroutine
    # qin.sun_abs = nir.sun_abs + vis_sun_abs + ir_shade
    # qin.shade_abs = nir.sh_abs + vis_sh_abs + ir_shade
    sun_abs = nir.sun_abs + vis_sun_abs + ir_shade
    shade_abs = nir.sh_abs + vis_sh_abs + ir_shade

    qin = eqx.tree_at(lambda t: (t.sun_abs, t.shade_abs), qin, (sun_abs, shade_abs))

    return qin


def leaf_energy(
    boundary_layer_res: BoundLayerRes,
    qin: Float_2D,
    met: Met,
    prof: Prof,
    radcan: SunShadedCan,
    prm: Para,
) -> SunShadedCan:
    """_summary_

    Args:
        boundary_layer_res (BoundLayerRes): _description_
        Q_In (Qin): _description_
        met (Met): _description_
        prof (Prof): _description_
        rad (SunShadedCan): _description_
        prm (Para): _description_

    Returns:
        SunShadedCan: _description_
    """
    gb = 1.0 / boundary_layer_res.vapor

    def rwater_hypo():
        return (gb * radcan.gs) / (gb + radcan.gs)

    def rwater_amphi():
        # gs_1side = radcan.gs/2.
        rs_1side = 2 / radcan.gs
        rtop = boundary_layer_res.vapor + rs_1side
        rbottom = rtop
        return rtop * rbottom / (rtop + rbottom)

    # gw = (gb * radcan.gs) / (gb + radcan.gs)
    rwater = jax.lax.switch(prm.stomata, [rwater_hypo, rwater_amphi])
    gw = 1.0 / rwater
    # met.P_Pa=1000 * met.P_kPa   # air pressure, Pa

    # Compute products of air temperature, K
    tk2 = prof.Tair_K[:, : prm.jtot] * prof.Tair_K[:, : prm.jtot]
    tk3 = tk2 * prof.Tair_K[:, : prm.jtot]
    tk4 = tk3 * prof.Tair_K[:, : prm.jtot]

    # Longwave emission at air temperature, W m-2
    llout = prm.epsigma * tk4
    d2est = fdes2dt(prof.Tair_K[:, : prm.jtot])
    dest = fdesdt(prof.Tair_K[:, : prm.jtot])
    est = fes(prof.Tair_K[:, : prm.jtot])
    vpd_Pa = est - prof.eair_Pa[:, : prm.jtot]
    llambda = fllambda(prof.Tair_K[:, : prm.jtot])
    air_density = dot(met.P_kPa, prm.Mair / (prm.rugc * prof.Tair_K[:, : prm.jtot]))
    vpd_Pa = jnp.clip(vpd_Pa, a_min=0.0, a_max=5000.0)
    # jax.debug.print("vpd: {a}", a=jnp.isnan(vpd_Pa).sum())

    # gas, heat and thermodynamic coeficients
    lecoef = dot(1.0 / met.P_Pa, air_density * 0.622 * llambda * gw)
    hcoef = air_density * prm.Cp / boundary_layer_res.heat
    hcoef2 = 2 * hcoef
    repeat = hcoef2 + prm.epsigma8 * tk3
    llout2 = 2 * llout
    # jax.debug.print("qin: {a}", a=qin[:2,:])
    # jax.debug.print("Tair_K: {a}", a=prof.Tair_K[:2,:])
    # jax.debug.print("d2est: {a}", a=d2est[:2,:])

    # coefficients analytical solution
    def calculate_coef_hypo():
        Acoef1 = lecoef * d2est / (2.0 * repeat)
        Acoef = Acoef1 / 2.0
        Bcoef = -repeat - lecoef * dest + Acoef * (-2.0 * qin + 4.0 * llout)
        Ccoef = Acoef * (qin * qin - 4 * qin * llout + 4 * llout * llout) + lecoef * (
            vpd_Pa * repeat + dest * (qin - 2 * llout)
        )
        return Acoef, Bcoef, Ccoef

    def calculate_coef_amphi():
        Acoef = lecoef * d2est / (2 * repeat)
        Bcoef = (
            -repeat
            - lecoef * dest
            - (qin / repeat) * (2 * Acoef)
            + 2 * Acoef * (2.0 * llout / repeat)
        )
        Ccoef = (
            repeat * lecoef * vpd_Pa
            + lecoef * dest * (qin - llout2)
            + lecoef
            * d2est
            / 2
            * (qin * qin - 4 * qin * llout + 4 * llout * llout)
            / repeat
        )
        return Acoef, Bcoef, Ccoef

    Acoef, Bcoef, Ccoef = jax.lax.switch(
        prm.stomata, [calculate_coef_hypo, calculate_coef_amphi]
    )
    # jax.debug.print("Acoef: {a}", a=Acoef[:2,:])
    # jax.debug.print("Bcoef: {a}", a=Bcoef[:2,:])
    # jax.debug.print("Ccoef: {a}", a=Ccoef[:2,:])

    #  solve for LE
    #  a LE^2 + bLE + c = 0
    # solve for both roots, but LE tends to be second root, le2
    product = Bcoef * Bcoef - 4.0 * Acoef * Ccoef
    # le1 = (-Bcoef + jnp.sqrt(Bcoef*Bcoef - 4.*Acoef*Ccoef)) / (2.*Acoef)
    le2 = (-Bcoef - jnp.sqrt(product)) / (2.0 * Acoef)
    # jax.debug.print("{a}", a=product.mean(axis=1))
    LE = jnp.real(le2)
    del_Tk = (qin - LE - llout2) / repeat
    # del_Tk -= 0.2
    Tsfc_K = prof.Tair_K[:, : prm.jtot] + del_Tk
    # Tsfc_K = prof.Tair_K[:, : prm.nlayers] + del_Tk
    # jax.debug.print("qin: {a}", a=qin.mean(axis=0))
    # jax.debug.print("LE: {a}", a=LE.mean(axis=0))
    # jax.debug.print("llout2: {a}", a=llout2.mean(axis=0))
    # jax.debug.print("del_Tk: {a}", a=del_Tk.mean(axis=0))

    # H is sensible heat flux density from both sides of leaf
    H = hcoef2 * jnp.real(del_Tk)

    # Lout is longwave emitted energy from both sides of leaves
    Lout = 2 * prm.epsigma * jnp.power(Tsfc_K, 4)
    Rnet = qin - Lout  # net radiation as a function of longwave energy emitted
    Tsfc = Tsfc_K
    # esTsfc=fes(Tsfc_K)
    closure = Rnet - H - LE  # test energy balance closure

    radcan = eqx.tree_at(
        lambda t: (t.LE, t.H, t.Tsfc, t.Lout, t.Rnet, t.vpd_Pa, t.closure),
        radcan,
        (LE, H, Tsfc, Lout, Rnet, vpd_Pa, closure),
    )
    # radcan.LE = LE
    # radcan.H = H
    # radcan.Tsfc = Tsfc
    # radcan.Lout = Lout
    # radcan.Rnet = qin - Lout
    # radcan.vpd_Pa = vpd_Pa
    # radcan.closure = closure

    return radcan
