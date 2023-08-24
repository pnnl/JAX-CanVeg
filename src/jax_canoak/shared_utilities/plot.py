import jax.numpy as jnp
import matplotlib.pyplot as plt

import pandas as pd

# from ..subjects import ParNir, Ir, Para

# Figure configurations parameters
# rc('text', usetex=False)
small_size = 15
medium_size = 25
bigger_size = 30
plt.rc("font", size=small_size)  # controls default text sizes
plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
plt.rc("axes", labelsize=small_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
plt.rc("legend", fontsize=small_size)  # legend fontsize
plt.rc("figure", titlesize=small_size)  # fontsize of the figure title
plt.rc("text", usetex=False)

figsize_1 = (5, 5)
figsize_1b = (8, 5)
figsize_1_2 = (12, 5)
figsize_2_1 = (5, 12)
figsize_2_2 = (10, 10)


# Plotting functions
def plot_timeseries(
    array, timesteps=None, ax=None, title=None, label=None, tunit="[day of year]"
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1b)
    if timesteps is None:
        ax.plot(array, label=label)
    else:
        ax.plot(timesteps, array, label=label)
    ax.set(xlabel=f"Time {tunit}", title=title)
    return ax


def plot_imshow(
    array,
    met,
    verticals,
    cmap="bwr",
    axes=None,
    title=None,
    tunit="[day of year]",
    is_canopy=True,
):
    times = get_time_doy(met)
    if is_canopy:
        extent = [times[0], times[-1], verticals[0], verticals[-1]]
        origin = "lower"
    else:
        extent = [times[0], times[-1], verticals[-1], verticals[0]]
        origin = "upper"
    ylabel = "Aboveground height [m]" if is_canopy else "Soil depth [m]"
    if axes is None:
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=(3, 1),
            height_ratios=(6, 1),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.05,
            hspace=0.05,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        axbar = fig.add_subplot(gs[1, 0], sharey=ax1)
    else:
        ax1, ax2, axbar = axes[0], axes[1], axes[2]
    im = ax1.imshow(array, extent=extent, origin=origin, cmap=cmap, aspect="auto")
    ax1.set(xlabel=f"Time {tunit}", ylabel=ylabel, title=title)
    plt.colorbar(im, cax=axbar, orientation="horizontal")
    # mean = compute_daily_average(array.mean(axis=0), met.hhour)
    if is_canopy:
        ax2.plot(array.mean(axis=1), verticals)
    else:
        ax2.plot(array.mean(axis=1)[::-1], verticals[::-1])
    ax2.set(xlabel=title, title="Averaged")
    return axes, im


def plot_imshow2(
    array,
    met,
    verticals,
    key="T",
    cmap="bwr",
    axes=None,
    title=None,
    tunit="[day of year]",
    is_canopy=True,
):
    times = get_time_doy(met)
    if axes is None:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=(3, 1),
            height_ratios=(2, 6, 0.2),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.1,
            hspace=0.5,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
        axbar = fig.add_subplot(gs[2, 0])
        axes_imshow = [ax1, ax2, axbar]
    else:
        ax0 = axes[0]
        axes_imshow = axes[1:]
    if key == "T":
        met_array, met_title = met.T_air_K, "Air temperature [degK]"
    elif key == "e":
        met_array, met_title = met.eair, "Air pressure [Pa]"
    elif key == "co2":
        met_array, met_title = met.CO2, "Air CO2 [ppm]"
    else:
        raise Exception(f"Unknown key: {key}")
    plot_timeseries(met_array, times, ax=ax0, title=met_title, tunit="[day of year]")
    ax0.set(xlabel=None)
    plot_imshow(
        array,
        met,
        verticals,
        cmap=cmap,
        axes=axes_imshow,
        title=title,
        tunit=tunit,
        is_canopy=is_canopy,
    )


def plot_obs_1to1(obs, can, lim, varn="varn", ax=None):
    # ------ observation versus simulation ------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
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
    return ax


def plot_timeseries_obs_1to1(obs, sim, lim, timesteps=None, varn="varn", axes=None):
    if axes is None:
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=(3, 1),
            left=0.1,
            right=0.9,
            bottom=0.1,
            top=0.9,
            wspace=0.1,
            hspace=0.5,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    else:
        ax1, ax2 = axes[0], axes[1]
    plot_timeseries(obs, timesteps=timesteps, ax=ax1, title=None, label="observation")
    plot_timeseries(sim, timesteps=timesteps, ax=ax1, title=None, label="simulation")
    ax1.legend()
    ax1.set(title=varn)
    plot_obs_1to1(obs, sim, lim, ax=ax2)
    plt.subplots_adjust(wspace=0.5)
    return [ax1, ax2]


def plot_rad(rad, setup, lai, waveband: str, ax=None):
    jtot = setup.n_can_layers
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
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
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
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


def plot_veg_temp(sun, shade, prm, met, axes=None, cmap="bwr", vlim=None, cbar=True):
    times = get_time_doy(met)
    extent = [times[0], times[-1], prm.zht1[0], prm.zht1[-1]]
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize_2_1, sharex=True)
    if vlim is None:
        vlim = (sun.Tsfc.min(), sun.Tsfc.max())
    ax = axes[0]
    ax.imshow(
        sun.Tsfc.T,
        origin="lower",
        cmap=cmap,
        extent=extent,
        aspect="auto",
        vmin=vlim[0],
        vmax=vlim[1],
    )
    ax.set(title="Sunlit leaf temperature [degK]", ylabel="Canopy height [m]")
    ax = axes[1]
    im = ax.imshow(
        shade.Tsfc.T,
        origin="lower",
        cmap=cmap,
        extent=extent,
        aspect="auto",
        vmin=vlim[0],
        vmax=vlim[1],
    )
    ax.set(
        title="Shaded leaf temperature [degK]",
        ylabel="Canopy height [m]",
        xlabel="Day of year",
    )
    if cbar:
        plt.colorbar(im, orientation="horizontal", anchor=(0.5, 1.0))
    return axes, im


def plot_veg_temp_mean(sun, shade, prm, setup, ax=None):
    jtot = setup.n_can_layers
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    ax.plot(jnp.nanmean(sun.Tsfc, 0), prm.zht[:jtot], label="Sun")
    ax.plot(jnp.nanmean(shade.Tsfc, 0), prm.zht[:jtot], label="Shade")
    ax.set(xlabel="Temperature [degK]", ylabel="canopy layers")
    ax.legend()
    return ax


def plot_canopy1(can, sunorshade: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
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
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
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


def plot_leafang(leafang, ax=None, cmap="Blues"):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    im = ax.imshow(leafang.integ_exp_diff.T, aspect="auto", cmap=cmap)
    ax.set(ylabel="Time", xlabel="Canopy layers")
    plt.colorbar(im)
    return ax


def plot_soil(soil, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    ax.scatter(soil.rnet, soil.evap + soil.heat + soil.gsoil)
    ax.set(ylabel="Heat+Evap+Gsoil", xlabel="Rnet", title="Soil energy balance")
    return ax


def plot_soil_temp(soil, met, ax=None, vlim=None, cmap="copper_r", cbar=False):
    times = get_time_doy(met)
    extent = [times[0], times[-1], soil.z_soil[-1], soil.z_soil[1]]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    if vlim is None:
        vlim = (soil.T_soil.min(), soil.T_soil.max())
    im = ax.imshow(
        soil.T_soil[:, :-2].T,
        cmap=cmap,
        extent=extent,
        vmin=vlim[0],
        vmax=vlim[1],
        aspect="auto",
    )
    ax.set(xlabel="Time", ylabel="Soil depth [m]", title="Soil tempeture [degK]")
    if cbar:
        plt.colorbar(im)
    return ax, im


def plot_totalenergy(soil, veg, cantop_rnet, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize_1_2)
    ax = axes[0]
    ax.scatter(
        soil.rnet + veg.Rnet, soil.evap + veg.LE + soil.heat + veg.H + soil.gsoil
    )
    ax.set(ylabel="Heat+Evap+Gsoil", xlabel="Rnet", title="Total energy balance")
    ax = axes[1]
    ax.scatter(soil.rnet + veg.Rnet, cantop_rnet)
    ax.set(ylabel="Rnet calculated, top of canopy", xlabel="can rnet")
    return axes


def plot_temp(soil, sun, shade, prm, met, axes=None, vlim=None, cmap="bwr"):
    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(5, 15), sharex=True)
    times = get_time_doy(met)
    if vlim is None:
        vlim = (sun.Tsfc.min(), sun.Tsfc.max())
    # Air temperature
    ax = axes[0]
    ax = plot_timeseries(met.T_air_K, times, ax, title="Air temperature [degK]")
    ax.set(xlabel="")
    # Leaf temperature
    axes_leaf = axes[1:3]
    axes_leaf, im = plot_veg_temp(
        sun, shade, prm, met, axes_leaf, vlim=vlim, cbar=False, cmap=cmap
    )
    axes_leaf[1].set(xlabel="")
    # Soil temperature
    ax = axes[-1]
    ax, im = plot_soil_temp(soil, met, ax, vlim, cbar=False, cmap=cmap)
    ax.set(xlabel="Time [day of year]")
    plt.colorbar(im, orientation="horizontal", anchor=(0.5, 0.0))


def plot_dij(dij, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    for i in range(dij.shape[1]):
        ax.plot(dij[: prm.zht.size, i], prm.zht)
    ax.set(ylabel="Height [m]", xlabel="Dij [s/m]")
    return ax


def plot_prof1(prof, axes=None, cmap="bwr"):
    if axes is None:
        fig, axes = plt.subplots(3, 2, figsize=figsize_2_2)
    ax = axes[0, 0]
    im = ax.imshow(prof.Ps.T, aspect="auto", cmap=cmap)
    ax.set(title="Ps")
    plt.colorbar(im)
    ax = axes[0, 1]
    im = ax.imshow(prof.LE.T, aspect="auto", cmap=cmap)
    ax.set(title="LE")
    plt.colorbar(im)
    ax = axes[1, 0]
    im = ax.imshow(prof.H.T, aspect="auto", cmap=cmap)
    ax.set(title="H")
    plt.colorbar(im)
    ax = axes[1, 1]
    im = ax.imshow(prof.Rnet.T, aspect="auto", cmap=cmap)
    ax.set(title="Rnet")
    plt.colorbar(im)
    ax = axes[2, 0]
    im = ax.imshow(prof.Tsfc.T, aspect="auto", cmap=cmap)
    ax.set(title="Tsfc")
    plt.colorbar(im)
    ax = axes[2, 1]
    im = ax.imshow(prof.Tair_K.T, aspect="auto", cmap=cmap)
    ax.set(title="Tair_K")
    plt.colorbar(im)
    return axes


def plot_prof2(prof, prm, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize_1_2)
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
    return axes


def plot_daily(met, soil, veg, prm, axes=None, wspace=0.5, hspace=0.0):
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize_1_2)
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

    if axes is None:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return axes


def plot_obs_comparison(obs, can, axes=None):
    # ------ Obs versus can ------
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize_2_2)
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
    return axes


def plot_obs_energy_closure(obs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize_1)
    rnet_lim = [-50, 800]
    ax.scatter(obs.rnet, obs.H + obs.LE + obs.gsoil)
    ax.plot(rnet_lim, rnet_lim, "k--")
    ax.set(
        xlabel="Rnet, measured",
        ylabel="H+LE+Gsoil, measured",
        xlim=rnet_lim,
        ylim=rnet_lim,
    )
    return ax


# Some utility functions
def compute_daily_average(values, hhours):
    d = pd.DataFrame([hhours, values]).T.astype(float)
    d = d.groupby(0).mean()
    return d


def get_time_doy(met):
    day, hour = met.day, met.hhour / 24.0
    return day + hour
