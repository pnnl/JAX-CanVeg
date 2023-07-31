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


def plot_leafang(leafang, prm, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(leafang.integ_exp_diff.T, aspect="auto", cmap="Blues")
    ax.set(ylabel="Time", xlabel="Canopy layers")
    plt.colorbar(im)
