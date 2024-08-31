"""
Canopy structure functions, including:
- angle()
- leaf_angle()

Author: Peishi Jiang
Date: 2023.06.28.
"""

import jax
import jax.numpy as jnp

from ...subjects import Para, SunAng, LeafAng, Lai
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.types import Int_0D, Int_1D
from ...shared_utilities.constants import PI

# @jax.jit
def angle(
    latitude: Float_0D,
    longitude: Float_0D,
    time_zone: Int_0D,
    day: Int_1D,
    hour: Float_1D,
    # ) -> Tuple[Float_0D, Float_0D, Float_0D]:
) -> SunAng:
    """ANGLE computes solar elevation angles,
       This subroutine is based on algorithms in
       Walraven. 1978. Solar Energy. 20: 393-397.

    Args:
        latitude (Float_0D): _description_
        longitude (Float_0D): _description_
        zone (int): _description_
        day (int): _description_
        hour (Float_0D): _description_

    Returns:
        Tuple[Float_1D, Float_1D, Float_1D]: _description_
    """
    # ntime = hour.size

    RADD = PI / 180.0
    lat_rad = latitude * RADD  # latitude, radians
    # long_rad = longitude*RADD # longitude, radians

    # delta_hours = delta_long * 12. / PI
    #
    # Calculate declination angle
    declin = -23.45 * PI / 180 * jnp.cos(2 * PI * (day + 10) / 365)
    # print(declin)

    # Calculate hours of day length
    # -jnp.tan(lat_rad) * jnp.tan(declin)
    # sunrise = 12 - 12*jnp.arccos(cos_hour) / PI
    # sunset  = 12 + 12*jnp.arccos(cos_hour) / PI
    # daylength = sunset - sunrise
    f = PI * (279.5 + 0.9856 * day) / 180
    # print(sunrise, sunset)

    # Calcuate equation of time in hours
    Et = (
        -104.7 * jnp.sin(f)
        + 596.2 * jnp.sin(2 * f)
        + 4.3 * jnp.sin(3 * f)
        - 12.7 * jnp.sin(4 * f)
        - 429.3 * jnp.cos(f)
        - 2.0 * jnp.cos(2 * f)
        + 19.3 * jnp.cos(3 * f)
    ) / 3600
    # print(Et*60)

    # Perform longitudinal correction
    # Lc_deg = longitude + zone * 15  # degrees from local meridian
    Lc_deg = longitude - time_zone * 15  # degrees from local meridian
    Lc_hr = Lc_deg * 4.0 / 60.0  # hours, 4 minutes/per degree
    T0 = 12 - Lc_hr - Et
    hour_rad = PI * (hour - T0) / 12.0  # hour angle, radians

    # Calculate sine of solar elevation ,beta
    # print(lat_rad, declin, hour_rad, hour, T0)
    sin_beta = jnp.sin(lat_rad) * jnp.sin(declin) + jnp.cos(lat_rad) * jnp.cos(
        declin
    ) * jnp.cos(hour_rad)

    # Added by Peishi to avoid numerical instability
    @jnp.vectorize
    def truncate_small_numbers(v):
        thres = 0.01
        return jax.lax.cond(jnp.abs(v) > thres, lambda: v, lambda: jnp.sign(v) * thres)

    sin_beta = truncate_small_numbers(sin_beta)

    # Calculate solar elevation, radians
    beta_rad = jnp.arcsin(sin_beta)

    # # Calculate solar elevation, degrees
    # beta_deg = beta_rad * 180 / PI

    # theta_rad = PI / 2.0 - beta_rad
    # theta_deg = theta_rad * 180.0 / PI

    sun_ang = SunAng(
        sin_beta,
        beta_rad,
        # beta_deg,
        # theta_rad,
        # theta_deg,
    )
    # sun_ang.sin_beta = sin_beta
    # sun_ang.beta_rad = beta_rad
    # sun_ang.beta_deg = beta_deg
    # sun_ang.theta_rad = PI / 2.0 - beta_rad
    # sun_ang.theta_deg = sun_ang.theta_rad * 180.0 / PI

    return sun_ang
    # return beta_rad, sin_beta, beta_deg


# @jax.jit
def leaf_angle(
    sunang: SunAng, prm: Para, leafangle: int, lai: Lai, num_leaf_class: int = 50
) -> LeafAng:
    # leafang = LeafAng(setup.ntime, setup.jtot, num_leaf_class)
    # estimate leaf angle for 50 classes between 0 and pi/2
    # at midpoint between each angle class.  This is a big improvement over
    # the older code that divided the sky into 9 classes.
    thetaLeaf = (
        jnp.linspace(PI / 100, PI / 2, num=num_leaf_class) - PI / 200
    )  # solar zenith angle    # noqa: E501
    branches = [
        lambda: 2.0 * (1 - jnp.cos(2 * thetaLeaf)) / PI,  # planophile
        lambda: jnp.sin(thetaLeaf),  # spherical
        lambda: 2 * (1 + jnp.cos(2 * thetaLeaf)) / PI,  # erectophile
        lambda: 2 * (1 - jnp.cos(4 * thetaLeaf)) / PI,  # plagiophile
        lambda: 2 * (1 + jnp.cos(4 * thetaLeaf)) / PI,  # extremophile
        lambda: jnp.ones(num_leaf_class) * 2 / PI,  # uniform
        lambda: jnp.sin(thetaLeaf),  # others
    ]  # (50,)
    # jax.debug.print("leaf angle: {a}", a=branches[setup.leafangle]())
    # pdf = jax.lax.switch(setup.leafangle, branches)
    # jax.debug.print("leaf angle: {a}", a=pdf)
    pdf = jax.lax.switch(leafangle, branches)

    # using the algorithm from Warren Wilson and Wang et al
    # Wang, W. M., Z. L. Li, and H. B. Su. 2007.
    # Comparison of leaf angle distribution functions:
    # Effects on extinction coefficient and fraction of sunlit foliage.
    # Agricultural and Forest Meteorology 143:106-122.
    # call function for G function, the direction cosine, for sun zenith angle
    # compute a matrix of Gfunc for all the inputs
    thetaSun = sunang.theta_rad
    Gfunc = Gfunc_dir(thetaSun, thetaLeaf, pdf)  # (ntime,)

    @jnp.vectorize
    def tune_zero(g):
        return jax.lax.cond(g < 0.0, lambda: 0.5, lambda: g)

    Gfunc = tune_zero(Gfunc)
    # Gfunc = Gfunc.at[Gfunc < 0.0].set(0.5)
    # leafang.Gfunc = jnp.clip(leafang.Gfunc, a_min=0.5)
    # leafang.Gfunc(leafang.Gfunc < 0)=0.5

    # call function for Gfunction for each sky sector of the hemisphere
    thetaSky = thetaLeaf  # (nclass,)
    DA = PI / (2.0 * thetaSky.size)  # azimuth increment -> (pi/2)

    # Compute G function for all sky sectors
    Gfunc_Sky = Gfunc_diff(thetaSky, thetaLeaf, pdf)  # (nclass,)

    # compute probability of beam transfer with a markov function for clumped leaves
    dff_Markov = (
        # prm.dff * prm.markov
        lai.dff
        * prm.markov
    )  # the new LAI profile data of Belane (ntime, nlayers)  # noqa: E501
    dff_Markov = jnp.expand_dims(dff_Markov, axis=-1)  # (ntime, nlayers, 1)
    dff_Markov = jnp.tile(dff_Markov, num_leaf_class)  # (ntime, nlayers, nclass)
    g = Gfunc_Sky / jnp.cos(thetaSky)  # (nclass)
    exp_diffuse = -dff_Markov * g  # (ntime, nlayers, nclass)
    exp_diffuse = jnp.exp(exp_diffuse)  # (ntime, nlayers, nclass)
    XX = jnp.sum(
        exp_diffuse * (jnp.cos(thetaSky) * jnp.sin(thetaSky)), axis=2
    )  # noqa: E501 (ntime, nlayers)
    integ_exp_diff = 2.0 * XX * DA  # (ntime, nlayers)

    leafang = LeafAng(pdf, Gfunc, thetaSky, Gfunc_Sky, integ_exp_diff)

    return leafang


def Gfunc_dir(theta_rad: Float_1D, theta_leaf: Float_1D, pdf: Float_1D) -> Float_1D:
    """Evaluate G function, the direction cosine as a function of solar
       zenith angle, theta_rad

    Args:
        theta_rad (Float_1D): _description_
        theta_leaf (Float_1D): _description_
        pdf (Float_1D): _description_

    Returns:
        Float_1D: _description_
    """
    product1 = jnp.outer(1.0 / jnp.tan(theta_rad), 1.0 / jnp.tan(theta_leaf))
    # jax.debug.print('product: {x}', x=product1)
    # jax.debug.print('theta_rad: {x}', x=theta_rad)
    # jax.debug.print('theta_leaf: {x}', x=theta_leaf)

    # @jnp.vectorize
    # def calculate_psi(p_e):
    #     return jax.lax.cond(jnp.abs(p_e) > 1.0, lambda: 0.0, lambda: jnp.arccos(p_e))
    # psi = calculate_psi(product1)
    product1 = jnp.clip(product1, a_min=-1.0, a_max=1.0)
    psi = jnp.arccos(product1)

    product2 = jnp.outer(jnp.cos(theta_rad), jnp.cos(theta_leaf))

    @jnp.vectorize
    def calculate_A(p_e1, p_e2, psi_e):
        return jax.lax.cond(
            jnp.abs(p_e1) > 1.0,
            lambda: p_e2,
            lambda: p_e2 * (1.0 + 2.0 / PI * (jnp.tan(psi_e) - psi_e)),
        )

    A = calculate_A(product1, product2, psi)

    # ntime, nclass = theta_rad.size, pdf.size
    # A, psi = jnp.zeros([ntime, nclass]), jnp.zeros([ntime, nclass])
    # mask = jnp.abs(jnp.outer(1.0 / jnp.tan(theta_rad), 1.0/jnp.tan(theta_leaf))) > 1.0
    # psi = psi.at[~mask].set(
    #     jnp.arccos(jnp.outer(1.0 / jnp.tan(theta_rad), 1.0 /jnp.tan(theta_leaf)))[
    #         ~mask
    #     ]
    # )

    # A = A.at[mask].set(jnp.outer(jnp.cos(theta_rad), jnp.cos(theta_leaf))[mask])
    # A = A.at[~mask].set(
    #     jnp.outer(jnp.cos(theta_rad), jnp.cos(theta_leaf))[~mask]
    #     * (1.0 + 2.0 / PI * (jnp.tan(psi[~mask]) - psi[~mask]))
    # )

    F = A * pdf  # (ntime, nclass)
    Gfunc = jax.vmap(jnp.trapezoid, in_axes=[0, None])(F, theta_leaf)  # pyright: ignore
    return Gfunc


def Gfunc_diff(thetaSky: Float_1D, thetaLeaf: Float_1D, pdf: Float_1D) -> Float_1D:
    """evaluate Gfunc Sky for all the sky zones of the hemisphere and
       assume azimuthal symmetry.
       Use Gfunc_sky to compute the exp Beers law probability of
       penetration from the sky for an incremental layer of leaves.

    Args:
        thetaSky (Float_1D): _description_
        thetaLeaf (Float_1D): _description_
        pdf (Float_1D): _description_

    Returns:
        Float_1D: _description_
    """
    product1 = jnp.outer(1.0 / jnp.tan(thetaSky), 1.0 / jnp.tan(thetaLeaf))

    # @jnp.vectorize
    # def calculate_psi(p_e):
    #     return jax.lax.cond(jnp.abs(p_e) > 1.0, lambda: 0.0, lambda: jnp.arccos(p_e))
    # psi = calculate_psi(product1)
    product1 = jnp.clip(product1, a_min=-1.0, a_max=1.0)
    psi = jnp.arccos(product1)

    product2 = jnp.outer(jnp.cos(thetaSky), jnp.cos(thetaLeaf))

    @jnp.vectorize
    def calculate_A(p_e1, p_e2, psi_e):
        return jax.lax.cond(
            jnp.abs(p_e1) > 1.0,
            lambda: p_e2,
            lambda: p_e2 * (1.0 + 2.0 / PI * (jnp.tan(psi_e) - psi_e)),
        )

    Adiff = calculate_A(product1, product2, psi)

    # n1, n2 = thetaSky.size, thetaLeaf.size

    # Adiff = jnp.zeros([n1, n2])
    # psi = jnp.zeros([n1, n2])

    # mask = jnp.abs(jnp.outer(1.0 / jnp.tan(thetaSky), 1.0 /jnp.tan(thetaLeaf))) > 1.0

    # psi = psi.at[~mask].set(
    #     jnp.arccos(jnp.outer(1.0 / jnp.tan(thetaSky), 1.0 /jnp.tan(thetaLeaf)))[~mask]
    # )

    # Adiff = Adiff.at[mask].set(jnp.outer(jnp.cos(thetaSky), jnp.cos(thetaLeaf))[mask])
    # Adiff = Adiff.at[~mask].set(
    #     jnp.outer(jnp.cos(thetaSky), jnp.cos(thetaLeaf))[~mask]
    #     * (1.0 + 2.0 / PI * (jnp.tan(psi[~mask]) - psi[~mask]))
    # )

    F = Adiff * pdf  # (ntime, nclass)
    Gfunc_Sky = jax.vmap(jnp.trapezoid, in_axes=[0, None])(  # pyright: ignore
        F, thetaLeaf
    )
    return Gfunc_Sky

    # def calculate_each_GfuncSky(theta_sky_each):
    #     Adiff = jnp.zeros(n2)
    #     psi = jnp.zeros(n2)
    #     mask = jnp.abs(
    #         1./jnp.tan(theta_sky_each)*1./jnp.tan(thetaLeaf)
    #     ) > 1.
    #     Adiff = Adiff.at[mask].set(
    #         jnp.cos(theta_sky_each)*jnp.cos(thetaLeaf[mask])
    #     )
    #     psi = psi.at[~mask].set(
    #         jnp.arccos(1./jnp.tan(theta_sky_each)*1./jnp.tan(thetaLeaf[~mask]))
    #     )
    #     Adiff = Adiff.at[~mask].set(
    #         jnp.cos(theta_sky_each)*jnp.cos(thetaLeaf[~mask])*\
    #             (1.+2./PI*(jnp.tan(psi[~mask])-psi[~mask]))
    #     )
    #     F = Adiff * pdf
    #     G_sky = jnp.trapz(F, thetaLeaf)
    #     return G_sky

    # Gfunc_Sky = jax.vmap(calculate_each_GfuncSky, in_axes=[0])(thetaSky)
    # return Gfunc_Sky
