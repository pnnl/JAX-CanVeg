import jax.numpy as jnp
import matplotlib.pyplot as plt

import pandas as pd

# from ..subjects import ParNir, Ir, Para


def plot_rad(rad, setup, lai, waveband: str, ax=None):
    jtot = setup.n_can_layers
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sumlai = jnp.mean(lai.sumlai, 0)
    ax.plot(jnp.mean(rad.up_flux[:, :jtot], 0), sumlai, label="up")
    # ax=gca;
    # set(ax, 'ydir','reverse');
    ax.plot(jnp.mean(rad.dn_flux[:, :jtot], 0), sumlai, label="down")
    ax.plot(jnp.mean(rad.beam_flux[:, :jtot], 0), sumlai, label="beam")
    ax.plot(jnp.mean(rad.total[:, :jtot], 0), sumlai, label="total")
    ax.plot(jnp.mean(rad.sun_abs[:, :jtot], 0), sumlai, label="sun abs")
    ax.plot(jnp.mean(rad.sh_abs[:, :jtot], 0), sumlai, label="shade abs")
    ax.legend()
    ax.invert_yaxis()
    ax.set(
        xlabel="Radiation Flux Density", ylabel="Canopy Cumulative LAI", title=waveband
    )

    return ax


# def plot_ir(ir: Ir, prm: Para, ax=None):
def plot_ir(ir, setup, lai, ax=None):
    jtot = setup.n_can_layers
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sumlai = jnp.mean(lai.sumlai, 0)
    ax.plot(jnp.nanmean(ir.ir_up[:, :jtot], 0), sumlai, "+-", label="up")
    # ax=gca;
    # set(ax, 'ydir','reverse');
    ax.plot(jnp.nanmean(ir.ir_dn[:, :jtot], 0), sumlai, label="down")
    ax.legend()
    ax.invert_yaxis()
    ax.set(
        xlabel="Radiation Flux Density",
        ylabel="Canopy Cumulative LAI",
        title="IR flux density, W m-2",
    )
    # xlabel('Radiation Flux Density')
    # ylabel('Canopy Depth')
    # title('IR flux density, W m-2');
    return ax


def plot_veg_temp(sun, shade, prm, setup, ax=None):
    jtot = setup.n_can_layers
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(jnp.nanmean(sun.Tsfc, 0), prm.zht[:jtot], label="Sun")
    ax.plot(jnp.nanmean(shade.Tsfc, 0), prm.zht[:jtot], label="Shade")
    ax.set(xlabel="Temperature [degK]", ylabel="canopy layers")
    ax.legend()


def plot_canopy1(can, sunorshade: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # sumlai = jnp.mean(prm.sumlai, 0)
    ax.plot(jnp.nanmean(can.LE, 1), label="LE")
    ax.plot(jnp.nanmean(can.H, 1), label="H")
    ax.plot(jnp.nanmean(can.Rnet, 1), label="Rnet")
    # ax.plot(jnp.nanmean(can.Lout, 1), label='Lout')
    # if sunorshade == 'sun':
    #     ax.plot(jnp.nanmean(qin.sun_abs, 1), label='Qin')
    # elif sunorshade == 'shade':
    #     ax.plot(jnp.nanmean(qin.shade_abs, 1), label='Qin')
    ax.legend()
    ax.set(xlabel="Time", ylabel="Radiation", title=sunorshade)
    return ax


def plot_canopy2(can, waveband: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(
        jnp.nanmean(can.Rnet, 1),
        jnp.nanmean(can.LE + can.H, 1),
    )
    ax.legend()
    ax.set(xlabel="Rnet", ylabel="LE+H", title=waveband)
    return ax


# def plot_canopy3(can, prm, waveband: str, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#     ax.scatter(
#         jnp.nanmean(can.Rnet, 1),
#         jnp.nanmean(can.LE+can.H, 1),
#     )
#     ax.legend()
#     ax.set(
#         xlabel="Rnet", ylabel="LE+H", title=waveband
#     )
#     return ax


def plot_leafang(leafang, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(leafang.integ_exp_diff.T, aspect="auto", cmap="Blues")
    ax.set(ylabel="Time", xlabel="Canopy layers")
    plt.colorbar(im)


def plot_soil(soil, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(soil.rnet, soil.evap + soil.heat + soil.gsoil)
    ax.set(ylabel="Heat+Evap+Gsoil", xlabel="Rnet", title="Soil energy balance")


def plot_soiltemp(soil, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(soil.T_soil[:, :-2].T, cmap="copper_r", aspect="auto")
    ax.set(xlabel="Time", ylabel="Soil layer")
    plt.colorbar(im)


def plot_totalenergy(soil, veg, cantop_rnet, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    ax = axes[0]
    ax.scatter(
        soil.rnet + veg.Rnet, soil.evap + veg.LE + soil.heat + veg.H + soil.gsoil
    )
    ax.set(ylabel="Heat+Evap+Gsoil", xlabel="Rnet", title="Total energy balance")
    ax = axes[1]
    ax.scatter(soil.rnet + veg.Rnet, cantop_rnet)
    ax.set(ylabel="Rnet calculated, top of canopy", xlabel="can rnet")


def plot_dij(dij, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i in range(dij.shape[1]):
        ax.plot(dij[: prm.zht.size, i], prm.zht)
    ax.set(ylabel="Height [m]", xlabel="Dij [s/m]")


def plot_prof1(prof, axes=None):
    if axes is None:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    ax = axes[0, 0]
    im = ax.imshow(prof.Ps.T, aspect="auto")
    ax.set(title="Ps")
    plt.colorbar(im)
    ax = axes[0, 1]
    im = ax.imshow(prof.LE.T, aspect="auto")
    ax.set(title="LE")
    plt.colorbar(im)
    ax = axes[1, 0]
    im = ax.imshow(prof.H.T, aspect="auto")
    ax.set(title="H")
    plt.colorbar(im)
    ax = axes[1, 1]
    im = ax.imshow(prof.Rnet.T, aspect="auto")
    ax.set(title="Rnet")
    plt.colorbar(im)
    ax = axes[2, 0]
    im = ax.imshow(prof.Tsfc.T, aspect="auto")
    ax.set(title="Tsfc")
    plt.colorbar(im)
    ax = axes[2, 1]
    im = ax.imshow(prof.Tair_K.T, aspect="auto")
    ax.set(title="Tair_K")
    plt.colorbar(im)


def plot_prof2(prof, prm, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    # Tair ~ ht
    ax = axes[0]
    ax.plot(jnp.nanmean(prof.Tair_K, 0), prm.zht)
    ax.set(xlabel="Tair [degK]", ylabel="Ht [m]")
    # eair ~ ht
    ax = axes[1]
    ax.plot(jnp.nanmean(prof.eair_Pa, 0), prm.zht)
    ax.set(xlabel="eair [Pa]", ylabel="Ht [m]")
    # co2 ~ ht
    ax = axes[2]
    ax.plot(jnp.nanmean(prof.co2, 0), prm.zht)
    ax.set(xlabel="CO2 [ppm]", ylabel="Ht [m]")


def plot_daily(met, soil, veg, prm, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    # Tsfc, Tair
    ax = axes[0]
    vegtsfc_mean = compute_daily_average(veg.Tsfc, met.hhour)
    soiltsfc_mean = compute_daily_average(soil.sfc_temperature, met.hhour)
    tair_mean = compute_daily_average(met.T_air_K, met.hhour)
    ax.plot(vegtsfc_mean.index, vegtsfc_mean.values, label="veg-tsfc")
    ax.plot(soiltsfc_mean.index, soiltsfc_mean.values, label="soil-tsfc")
    ax.plot(tair_mean.index, tair_mean.values, label="tair")
    ax.set(xlabel="Hr", ylabel="Temperature [degK]")
    ax.legend()

    # Energy fluxes
    Rnet_mean = compute_daily_average(soil.rnet + veg.Rnet, met.hhour)
    LE_mean = compute_daily_average(soil.evap + veg.LE, met.hhour)
    H_mean = compute_daily_average(soil.heat + veg.H, met.hhour)
    G_mean = compute_daily_average(soil.gsoil, met.hhour)
    ax = axes[1]
    ax.plot(Rnet_mean.index, Rnet_mean.values, label="Rnet")
    ax.plot(LE_mean.index, LE_mean.values, label="LE")
    ax.plot(H_mean.index, H_mean.values, label="H")
    ax.plot(G_mean.index, G_mean.values, label="G")
    ax.set(xlabel="Hr", ylabel="Energy fluxes [W m-2]")
    ax.legend()


def plot_obs_1to1(obs, can, lim, varn="varn", ax=None):
    # ------ Obs versus can ------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    l2 = jnp.mean((obs - can) ** 2)
    ax.scatter(obs, can)
    ax.plot(lim, lim, "k--")
    ax.set(
        xlim=lim,
        ylim=lim,
        xlabel="Measured",
        ylabel="Simulated",
        title=f"{varn} (L2: {l2:.3f})",
    )


def plot_obs_comparison(obs, can, axes=None):
    # ------ Obs versus can ------
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    LE_lim, H_lim = [-10, 750], [-220, 400]
    rnet_lim, gsoil_lim = [-50, 800], [-60, 70]
    ax = axes[0, 0]
    plot_obs_1to1(obs.LE, can.LE, LE_lim, varn="LE", ax=ax)
    ax = axes[0, 1]
    plot_obs_1to1(obs.H, can.H, H_lim, varn="H", ax=ax)
    ax = axes[1, 0]
    plot_obs_1to1(obs.rnet, can.rnet, rnet_lim, varn="Rnet", ax=ax)
    ax = axes[1, 1]
    plot_obs_1to1(obs.gsoil, can.gsoil, gsoil_lim, varn="Gsoil", ax=ax)


def plot_obs_energy_closure(obs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    rnet_lim = [-50, 800]
    ax.scatter(obs.rnet, obs.H + obs.LE + obs.gsoil)
    ax.plot(rnet_lim, rnet_lim, "k--")
    ax.set(
        xlabel="Rnet, measured",
        ylabel="H+LE+Gsoil, measured",
        xlim=rnet_lim,
        ylim=rnet_lim,
    )


def compute_daily_average(values, hhours):
    d = pd.DataFrame([hhours, values]).T.astype(float)
    d = d.groupby(0).mean()
    return d
