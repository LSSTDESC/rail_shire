#!/bin/env python3

import os
from collections import namedtuple
from functools import partial

import jax_cosmo as jc
from jax import jit
from jax import numpy as jnp

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid

# pi
# pi = 3.14159265359 #on utilise np.pi
# c
ckms = jc.constants.c  # en km/s
c = ckms * 1.0e13  # 2.99792458e18 # en A/s
# h
hplanck = 6.62606957e-34
# k Boltzmann
kboltzmann = 1.3806488e-23
# L solar
Lsol = 3.826e33
# pc en cm
pc = 3.086e18
# hc from Cedric
hc = 12398.42  # [eV.A]
# f_ga from Cedric
f_ga = 1

Cosmo = namedtuple("Cosmo", ["h0", "om0", "l0", "omt"])
PriorParams = namedtuple("PriorParams", ["mod", "zot", "kt", "alpt0", "pcal", "ktf", "ft", "nuv_range"])

DATALOC = os.path.join(os.environ["FORS2DATALOC"], "photoz")

# P(T|m0)
ktf = jnp.array([0.47165, 0.30663, 0.12715, -0.34437])
ft = jnp.array([0.43199, 0.07995, 0.31162, 0.21220])

prior_params_set = (
    PriorParams(0, 0.45181, 0.13677, 3.33078, 0.89744, ktf[0], ft[0], (4.25, jnp.inf)),
    PriorParams(1, 0.16560, 0.12983, 1.42815, 0.90868, ktf[1], ft[1], (3.19, 4.25)),
    PriorParams(2, 0.21072, 0.14008, 1.58310, 0.89747, ktf[2], ft[2], (1.9, 3.19)),
    PriorParams(3, 0.20418, 0.13773, 1.34500, 0.91760, ktf[3], ft[3], (-jnp.inf, 1.9)),
)

prior_pars_E_S0, prior_pars_Sbc, prior_pars_Scd, prior_pars_Irr = prior_params_set  # noqa: N816


@jit
def prior_mod(nuvk):
    """prior_mod Determines the model (galaxy morphology) for which to compute the prior value.

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: Model Id
    :rtype: int
    """
    val = (
        prior_pars_Irr.mod
        + (prior_pars_Scd.mod - prior_pars_Irr.mod) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.mod - prior_pars_Scd.mod) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.mod - prior_pars_Sbc.mod) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val.astype(int)


@jit
def prior_zot(nuvk):
    """prior_zot Determines the z0 value of the prior function

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: z0
    :rtype: float
    """
    val = (
        prior_pars_Irr.zot
        + (prior_pars_Scd.zot - prior_pars_Irr.zot) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.zot - prior_pars_Scd.zot) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.zot - prior_pars_Sbc.zot) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


@jit
def prior_alpt0(nuvk):
    """prior_alpt0 Determines the alpha0 value in the prior function (power law)

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: alpha0
    :rtype: float
    """
    val = (
        prior_pars_Irr.alpt0
        + (prior_pars_Scd.alpt0 - prior_pars_Irr.alpt0) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.alpt0 - prior_pars_Scd.alpt0) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.alpt0 - prior_pars_Sbc.alpt0) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


@jit
def prior_kt(nuvk):
    """prior_kt Determines the k value in the prior function

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: k
    :rtype: float
    """
    val = (
        prior_pars_Irr.kt
        + (prior_pars_Scd.kt - prior_pars_Irr.kt) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.kt - prior_pars_Scd.kt) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.kt - prior_pars_Sbc.kt) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


@jit
def prior_pcal(nuvk):
    """prior_pcal Determines p_cal value in the prior function

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: p_cal
    :rtype: float
    """
    val = (
        prior_pars_Irr.pcal
        + (prior_pars_Scd.pcal - prior_pars_Irr.pcal) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.pcal - prior_pars_Scd.pcal) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.pcal - prior_pars_Sbc.pcal) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


@jit
def prior_ktf(nuvk):
    """prior_ktf Determines the Ktf value in the prior function

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: k_tf
    :rtype: float
    """
    val = (
        prior_pars_Irr.mod
        + (prior_pars_Scd.ktf - prior_pars_Irr.ktf) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.ktf - prior_pars_Scd.ktf) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.ktf - prior_pars_Sbc.ktf) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


@jit
def prior_ft(nuvk):
    """prior_ft Determines the ft value in the prior function

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: ft
    :rtype: float
    """
    val = (
        prior_pars_Irr.mod
        + (prior_pars_Scd.ft - prior_pars_Irr.ft) * jnp.heaviside(nuvk - prior_pars_Scd.nuv_range[0], 0)
        + (prior_pars_Sbc.ft - prior_pars_Scd.ft) * jnp.heaviside(nuvk - prior_pars_Sbc.nuv_range[0], 0)
        + (prior_pars_E_S0.ft - prior_pars_Sbc.ft) * jnp.heaviside(nuvk - prior_pars_E_S0.nuv_range[0], 0)
    )
    return val


# Compute the metric distance dmet in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2
@partial(jit, static_argnums=0)
def distMet(cosmo, z):
    """distMet Computes the metric (comoving) distance dmet in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2

    :param cosmo: Cosmology within which to compute the distances.
    :type cosmo: Cosmology object
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :raises RuntimeError: the specified cosmology does not allow the computation of a distance.
    :return: Metric (comoving) distance un Mpc
    :rtype: float
    """
    dmet, ao = 0.0, 1.0
    # case without the cosmological constant
    if cosmo.l0 == 0:
        # ao = c/(self.h0*np.sqrt(np.abs(1-self.omt)))
        # in fact we use x = ao * x(z) with x(z) from eq 8 of
        # Moscardini et al.  So we don't need to compute ao
        if cosmo.om0 > 0:
            dmet = cosmo.om0 * z - (cosmo.om0 - 2) * (1 - jnp.sqrt(1 + cosmo.om0 * z))
            dmet = 2 * ckms / (ao * cosmo.h0 * cosmo.om0 * cosmo.om0 * (1 + z)) * dmet
        else:
            dmet = ckms * z * (1.0 + z / 2.0) / (cosmo.h0 * (1 + z))

    elif cosmo.om0 < 1 and cosmo.l0 != 0:
        # _sum = 0.
        dz = z / 50.0
        zi = jnp.linspace(0.5 * dz, z, num=50)
        Ez = jnp.power((cosmo.om0 * jnp.power((1.0 + zi), 3.0) + (1 - cosmo.om0 - cosmo.l0) * jnp.power((1.0 + zi), 2.0) + cosmo.l0), -0.5)
        _sum = trapezoid(Ez, zi)
        # for i in range(50):
        #    zi = (i+0.5)*dz
        #    Ez = jnp.sqrt(cosmo.om0*jnp.power((1.+zi),3.)+(1-cosmo.om0-cosmo.l0)*jnp.power((1.+zi),2.)+cosmo.l0)
        #    _sum = _sum + dz/Ez
        dmet = ckms / (cosmo.h0 * ao) * _sum
    else:
        raise RuntimeError(f"Cosmology not included : h0={cosmo.h0}, Om0={cosmo.om0}, l0={cosmo.l0}")
    return dmet


@partial(jit, static_argnums=0)
def distLum(cosmo, z):
    """distLum Computes the luminosity distance dlum in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2

    :param cosmo: Cosmology within which to compute the distances.
    :type cosmo: Cosmology object
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Luminosity distance un Mpc
    :rtype: float
    """
    return distMet(cosmo, z) * (1 + z)


@partial(jit, static_argnums=0)
def distAng(cosmo, z):
    """distAng Computes the anguar diameter distance dang in Mpc : dlum = dmet*(1+z), dang = dmet/(1+z) = dlum/(1+z)^2

    :param cosmo: Cosmology within which to compute the distances.
    :type cosmo: Cosmology object
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Angular diameter distance un Mpc
    :rtype: float
    """
    return distMet(cosmo, z) / (1 + z)


# Compute the distance modulus
@partial(jit, static_argnums=0)
def distMod(cosmo, z):
    r"""distMod Compute the distance modulus $d = 5 \log (d_{lum}[pc])-5$, i.e. the difference between absolute and observed magnitudes.

    :param cosmo: Cosmology within which to compute the distances.
    :type cosmo: Cosmology object
    :param z: Redshift value for which to compute the distance modulus.
    :type z: float
    :return: Distance modulus in units of magnitudes
    :rtype: float
    """
    # funz = 0.
    # if (z >= 1.e-10):
    return 5.0 * jnp.log10(distLum(cosmo, z) * 1.0e6) - 5.0


## Compute cosmological time from z=infinty  to z
## as a function of cosmology.  Age given in year !!
## Note : use lambda0 non zero if Omega_o+lambda_o=1
def time(cosmo, z):
    """time Compute cosmological time from redshift=infinity to z as a function of cosmology.
    Age given in year !!
    Note : use lambda0 non zero if Omega_o+lambda_o=1

    :param cosmo: Cosmology within which to compute the distances.
    :type cosmo: Cosmology object
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :raises RuntimeError: the specified cosmology does not allow the computation of a distance.
    :return: Lookback time corresponding to the redshift in the specified cosmology
    :rtype: float
    """
    timy = 0.0
    val = 0.0
    hy = cosmo.h0 * 1.0224e-12

    if jnp.abs(cosmo.om0 - 1) < 1.0e-6 and cosmo.l0 == 0:
        timy = 2.0 * jnp.power((1 + z), -1.5) / (3 * hy)
    elif cosmo.om0 == 0 and cosmo.l0 == 0:
        timy = 1.0 / (hy * (1 + z))
    elif cosmo.om0 < 1 and cosmo.om0 > 0 and cosmo.l0 == 0:
        val = (cosmo.om0 * z - cosmo.om0 + 2.0) / (cosmo.om0 * (1 + z))
        timy = 2.0 * jnp.sqrt((1 - cosmo.om0) * (cosmo.om0 * z + 1)) / (cosmo.om0 * (1 + z))
        timy = timy - jnp.log10(val + jnp.sqrt(val * val - 1))
        timy = timy * cosmo.om0 / (2.0 * hy * jnp.power((1 - cosmo.om0), 1.5))

    elif cosmo.om0 > 1 and cosmo.l0 == 0:
        timy = jnp.arccos((cosmo.om0 * z - cosmo.om0 + 2.0) / (cosmo.om0 * (1 + z)))
        timy = timy - 2 * jnp.sqrt((cosmo.om0 - 1) * (cosmo.om0 * z + 1)) / (cosmo.om0 * (1 + z))
        timy = timy * cosmo.om0 / (2 * hy * jnp.power((cosmo.om0 - 1), 1.5))

    elif cosmo.om0 < 1 and jnp.abs(cosmo.om0 + cosmo.l0 - 1) < 1.0e-5:
        val = jnp.sqrt(1 - cosmo.om0) / (jnp.sqrt(cosmo.om0) * jnp.power((1 + z), 1.5))
        timy = jnp.log(val + jnp.sqrt(val * val + 1))
        timy = timy * 2.0 / (3.0 * hy * jnp.sqrt(1 - cosmo.om0))

    else:
        raise RuntimeError("Not the right cosmology to derive the time.")
    return timy


def make_jcosmo(H0):
    """make_jcosmo Generates the cosmology object for the given H0 value, as implemented in the `jax_cosmo` module.

    :param H0: Current value for the Universe's expansion rate, aka Hubble's constant.
    :type H0: float
    :return: Cosmology for the specified H0 value, according to Planck 2015 data.
    :rtype: jax_cosmo.Planck15
    """
    return jc.Planck15(h=H0 / 100.0)


# JAX versions
# @partial(jit, static_argnums=0)
def calc_distM(cosm, z):
    """calc_distM Computes the radial comoving distance at given redshift within the specified `jax_cosmo` cosmology.

    :param cosm: Cosmology object
    :type cosm: jax_cosmo.Planck15
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Radial comoving distance un Mpc
    :rtype: float
    """
    return jc.background.radial_comoving_distance(cosm, jc.utils.z2a(z)) / cosm.h


# @partial(jit, static_argnums=0)
def calc_distLum(cosm, z):
    """calc_distLum Computes the luminosity distance at given redshift within the specified `jax_cosmo` cosmology.

    :param cosm: Cosmology object
    :type cosm: jax_cosmo.Planck15
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Luminosity distance un Mpc
    :rtype: float
    """
    return (1.0 + z) * jc.background.radial_comoving_distance(cosm, jc.utils.z2a(z)) / cosm.h


# @partial(jit, static_argnums=0)
def calc_distAng(cosm, z):
    """calc_distAng Computes the angular diameter distance at given redshift within the specified `jax_cosmo` cosmology.

    :param cosm: Cosmology object
    :type cosm: jax_cosmo.Planck15
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Angular diameter distance un Mpc
    :rtype: float
    """
    return jc.background.angular_diameter_distance(cosm, jc.utils.z2a(z)) / cosm.h


# Compute the distance modulus
# @partial(jit, static_argnums=0)
def calc_distMod(cosm, z):
    r"""calc_distMod Compute the distance modulus $d = 5 \log (d_{lum}[pc])-5$, i.e. the difference between absolute and observed magnitudes.

    :param cosm: Cosmology object
    :type cosm: jax_cosmo.Planck15
    :param z: Redshift value for which to compute the distance.
    :type z: float
    :return: Distance modulus in units of magnitudes
    :rtype: float
    """
    return 5.0 * jnp.log10(calc_distLum(cosm, z) * 1.0e6) - 5.0


@partial(jit, static_argnums=0)
def nz_prior_params(nuvk):
    """nz_prior_params Returns all parameters values to compute the parametric n(z) prior for a galaxy SED template, from its UV-IR color index.

    :param nuvk: Emitted UV-IR color index of the galaxy
    :type nuvk: float
    :return: Prior parameters
    :rtype: tuple(floats)
    """
    if nuvk > 4.25:
        # Case E/S0
        # Color UV-K of PHOTO_230506/El_cww.sed.resample.new.resample15.inter
        mod = 0
        zot = 0.45181
        kt = 0.13677
        alpt0 = 3.33078
        pcal = 0.89744
    elif nuvk > 3.19:
        # Case Sbc
        # Color UV-K of PHOTO_230506/Sbc_cww.sed.resample.new.resample8.inter
        mod = 1
        zot = 0.16560
        kt = 0.12983
        alpt0 = 1.42815
        pcal = 0.90868
    elif nuvk > 1.9:
        # Case Scd
        # Color UV-K of PHOTO_230506/Scd_cww.sed.resample.new.resample7.inter  -19.4878 + 21.1501
        mod = 2
        zot = 0.21072
        kt = 0.14008
        alpt0 = 1.58310
        pcal = 0.89747
    else:
        # Case Irr
        mod = 3  # noqa: F841
        zot = 0.20418
        kt = 0.13773
        alpt0 = 1.34500
        pcal = 0.91760
    return alpt0, zot, kt, pcal


@jit
def nz_prior_core(z, imag, alpt0, zot, kt, pcal, ktf_m, ft_m):
    """nz_prior_core Computes the n(z) prior value in function of input parameters, as done in LEPHARE for the COSMOS2020 catalog, from VVDS data.

    :param z: Redshift at which the prior is evaluated
    :type z: float
    :param imag: AB-magnitude in i band at which the prior is evaluated
    :type imag: float
    :param alpt0: alpt0 prior parameter value
    :type alpt0: float
    :param zot: zot prior parameter value
    :type zot: float
    :param kt: kt prior parameter value
    :type kt: float
    :param pcal: pcal prior parameter value
    :type pcal: float
    :param ktf_m: ktf_m prior parameter value
    :type ktf_m: float
    :param ft_m: ft_m prior parameter value
    :type ft_m: float
    :return: Prior probability density at z, i_mag for the given galaxy SED template.
    :rtype: float
    """
    # kk = imag-20.
    # zmax = zot + kt*(imag-20.)
    # pz = jnp.power(z,alpt0)*jnp.exp(-jnp.power((z/(zot + kt*(imag-20.))),alpt0))

    # Ratio for each type
    # rappSum = jnp.sum(ft*jnp.exp(-ktf*(imag-20.)))
    # rapp = ft_m*jnp.exp(-ktf_m*(imag-20.))

    # Normalisation of the probability function
    # _pcal=jnp.power(zot + kt*(imag-20.),alpt0+1)/alpt0*pcal

    # Final value
    # val = pz/_pcal * rapp/rappSum
    return (
        (jnp.power(z, alpt0) * jnp.exp(-jnp.power((z / (zot + kt * (imag - 20.0))), alpt0)))
        / (jnp.power(zot + kt * (imag - 20.0), alpt0 + 1) / alpt0 * pcal)
        * (ft_m * jnp.exp(-ktf_m * (imag - 20.0)))
        / (jnp.sum(ft * jnp.exp(-ktf * (imag - 20.0))))
    )
    # if not jnp.isfinite(val):
    #    val = 1.
    # return (val-1.)*jnp.isfinite(val).astype(float)+1.


"""E(B-V) prior not in use
ebv_prior_file = os.path.join(DATALOC, 'NoType_NoLaw_ebv_prior_dataframe_fromFORS2.pkl') #'NoType_ebv_prior_dataframe_fromFORS2.pkl'  #'BothExt_ebv_prior_dataframe_fromFORS2.pkl'
ebv_prior_df = pd.read_pickle(ebv_prior_file)

cols_to_stack = tuple(ebv_prior_df[col].values for col in ebv_prior_df.columns)
ebv_prior_arr = jnp.column_stack(cols_to_stack)

@jit
def ebv_prior_2laws(ebv, mod, law):
    kde_val = jnp.interp(ebv, ebv_prior_arr[:, 0], ebv_prior_arr[:, 1+2*mod+law])
    return kde_val

@jit
def ebv_prior_1law(ebv, mod):
    kde_val = jnp.interp(ebv, ebv_prior_arr[:, 0], ebv_prior_arr[:, 1+mod])
    return kde_val

@jit
def ebv_prior_notype(ebv, law):
    kde_val = jnp.interp(ebv, ebv_prior_arr[:, 0], ebv_prior_arr[:, 1+law])
    return kde_val

@jit
def ebv_prior_notype_nolaw(ebv):
    kde_val = jnp.interp(ebv, ebv_prior_arr[:, 0], ebv_prior_arr[:, 1])
    return kde_val

@jit
def ebv_prior(ebv, mod, law):
    if ebv_prior_file == 'BothExt_ebv_prior_dataframe_fromFORS2.pkl':
        kde_val = ebv_prior_2laws(ebv, mod, law)
    elif ebv_prior_file == 'NoType_NoLaw_ebv_prior_dataframe_fromFORS2.pkl':
        kde_val = ebv_prior_notype_nolaw(ebv)
    elif ebv_prior_file == 'NoType_ebv_prior_dataframe_fromFORS2.pkl':
        kde_val = ebv_prior_notype(ebv, law)
    else:
        kde_val = ebv_prior_1law(ebv, mod)
    return kde_val
"""
