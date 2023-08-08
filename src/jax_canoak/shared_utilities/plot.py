import jax.numpy as jnp
import matplotlib.pyplot as plt

# from ..subjects import ParNir, Ir, Para


def plot_rad(rad, prm, waveband: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sumlai = jnp.mean(prm.sumlai, 0)
    ax.plot(jnp.mean(rad.up_flux[:, : prm.jtot], 0), sumlai, label="up")
    # ax=gca;
    # set(ax, 'ydir','reverse');
    ax.plot(jnp.mean(rad.dn_flux[:, : prm.jtot], 0), sumlai, label="down")
    ax.plot(jnp.mean(rad.beam_flux[:, : prm.jtot], 0), sumlai, label="beam")
    ax.plot(jnp.mean(rad.total[:, : prm.jtot], 0), sumlai, label="total")
    ax.plot(jnp.mean(rad.sun_abs[:, : prm.jtot], 0), sumlai, label="sun abs")
    ax.plot(jnp.mean(rad.sh_abs[:, : prm.jtot], 0), sumlai, label="shade abs")
    ax.legend()
    ax.invert_yaxis()
    ax.set(
        xlabel="Radiation Flux Density", ylabel="Canopy Cumulative LAI", title=waveband
    )

    return ax


# def plot_ir(ir: Ir, prm: Para, ax=None):
def plot_ir(ir, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sumlai = jnp.mean(prm.sumlai, 0)
    ax.plot(jnp.nanmean(ir.ir_up[:, : prm.jtot], 0), sumlai, "+-", label="up")
    # ax=gca;
    # set(ax, 'ydir','reverse');
    ax.plot(jnp.nanmean(ir.ir_dn[:, : prm.jtot], 0), sumlai, label="down")
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


def plot_canopy1(can, qin, prm, sunorshade: str, ax=None):
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


def plot_canopy2(can, prm, waveband: str, ax=None):
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


def plot_leafang(leafang, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(leafang.integ_exp_diff.T, aspect="auto", cmap="Blues")
    ax.set(ylabel="Time", xlabel="Canopy layers")
    plt.colorbar(im)


def plot_soil(soil, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(soil.rnet, soil.evap + soil.heat + soil.gsoil)
    ax.set(ylabel="Heat+Evap+Gsoil", xlabel="Rnet", title="Soil energy balance")


def plot_soiltemp(soil, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(soil.T_soil[:, :-2].T, cmap="copper_r", aspect="auto")
    ax.set(xlabel="Time", ylabel="Soil layer")
    plt.colorbar(im)
