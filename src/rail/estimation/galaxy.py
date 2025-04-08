#!/bin/env python3

from collections import namedtuple

import jax.numpy as jnp
from jax import jit, vmap

from rail.dsps_fors2_pz import nz_prior_core, prior_alpt0, prior_ft, prior_kt, prior_ktf, prior_pcal, prior_zot

Observation = namedtuple("Observation", ["num", "ref_i_AB", "AB_colors", "AB_colerrs", "valid_filters", "valid_colors", "z_spec"])


def load_galaxy(photometry, ismag, id_i_band=3):
    """load_galaxy _summary_

    :param photometry: fluxes or magnitudes and corresponding errors as read from an ASCII input file
    :type photometry: list or array-like
    :param ismag: whether photometry is provided as AB-magnitudes or fluxes
    :type ismag: bool
    :param id_i_band: index of i-band in the photometry. The default is 3 for LSST u, g, r, i, z, y.
    :type id_i_band: int, optional
    :return: Tuple containing the i-band AB magnitude, the array of color indices for the observations (in AB mag units), the corresponding errors
    and the array of booleans indicating which filters were used for this observation.
    :rtype: tuple
    """
    assert len(photometry) % 2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."
    _phot = jnp.array([photometry[2 * i] for i in range(len(photometry) // 2)])
    _phot_errs = jnp.array([photometry[2 * i + 1] for i in range(len(photometry) // 2)])

    if ismag:
        c_ab = _phot[:-1] - _phot[1:]
        c_ab_err = jnp.power(jnp.power(_phot_errs[:-1], 2) + jnp.power(_phot_errs[1:], 2), 0.5)
        i_ab = _phot[id_i_band]
        filters_to_use = jnp.logical_and(_phot > -15.0, _phot < 50.0)
    else:
        c_ab = -2.5 * jnp.log10(_phot[:-1] / _phot[1:])
        c_ab_err = 2.5 / jnp.log(10) * jnp.power(jnp.power(_phot_errs[:-1] / _phot[:-1], 2) + jnp.power(_phot_errs[1:] / _phot[1:], 2), 0.5)
        i_ab = -2.5 * jnp.log10(_phot[id_i_band]) - 48.6
        filters_to_use = jnp.logical_and(_phot > 0.0, _phot_errs > 0.0)
    colors_to_use = jnp.array([b1 and b2 for (b1, b2) in zip(filters_to_use[:-1], filters_to_use[1:], strict=True)])
    return i_ab, c_ab, c_ab_err, filters_to_use, colors_to_use


def load_magnitudes(photometry, ismag):
    """load_magnitudes Returns the magnitudes and associated errors from photometric data that can be either fluxes or magnitudes.
    Contrary to `load_galaxy`, this function does not compute color indices, identifies i-mag or returns the booleans.
    Instead, it returns the data as JAX arrays with `nan` wherever data is missing.
    This is more a helper to build on, especially aimed at converting ASCII data to a `DataFrame` or `HDF5` file,
    and cannot be used directly to built an `Observation` object.

    :param photometry: fluxes or magnitudes and corresponding errors as read from an ASCII input file
    :type photometry: list or array-like
    :param ismag: whether photometry is provided as AB-magnitudes or fluxes
    :type ismag: bool
    :return: Tuple containing the AB magnitudes (in AB mag units) and the corresponding errors.
    :rtype: tuple
    """
    assert len(photometry) % 2 == 0, "Missing data in observations : check that magnitudes/fluxes and errors are available\n and listed as M (or F), error, M (or F), error, etc."

    if ismag:
        _phot_ab = jnp.array([photometry[2 * i] for i in range(len(photometry) // 2)])
        _phot_ab_errs = jnp.array([photometry[2 * i + 1] for i in range(len(photometry) // 2)])
        filters_to_use = jnp.logical_and(_phot_ab > -15.0, _phot_ab_errs < 50.0)
    else:
        _phot = jnp.array([photometry[2 * i] for i in range(len(photometry) // 2)])
        _phot_errs = jnp.array([photometry[2 * i + 1] for i in range(len(photometry) // 2)])
        _phot_ab = -2.5 * jnp.log10(_phot) - 48.6
        _phot_ab_errs = (2.5 / jnp.log(10)) * (_phot_errs / _phot)
        filters_to_use = jnp.logical_and(_phot > 0.0, _phot_errs > 0.0)

    mag_ab = jnp.where(filters_to_use, _phot_ab, jnp.nan)
    mag_ab_err = jnp.where(filters_to_use, _phot_ab_errs, jnp.nan)

    return mag_ab, mag_ab_err


vmap_load_magnitudes = vmap(load_magnitudes, in_axes=(0, None))


def mags_to_i_and_colors(mags_arr, mags_err_arr, id_i_band):
    """mags_to_i_and_colors Extracts magnitude in i-band and computes color indices and associated errors for photo-z estimation for a single observed galaxy (input).

    :param mags_arr: AB-magnitudes of the galaxy from the input catalog
    :type mags_arr: jax.array
    :param mags_err_arr: Errors in AB-magnitudes of the galaxy from the input catalog
    :type mags_err_arr: jax.array
    :param id_i_band: Identifier of the i-band in the input catalog's photometric system. Starts from 0, such as `i_mag = mags_arr[id_i_band]`.
    :type id_i_band: int
    :return: 3-tuple of JAX arrays containing processed input data : (i_mag ; color indices ; errors on color indices)
    :rtype: tuple of jax.array
    """
    c_ab = mags_arr[:-1] - mags_arr[1:]
    c_ab_err = jnp.power(jnp.power(mags_err_arr[:-1], 2) + jnp.power(mags_err_arr[1:], 2), 0.5)
    i_ab = mags_arr[id_i_band]

    filters_to_use = jnp.logical_and(jnp.isfinite(mags_arr), jnp.isfinite(mags_err_arr))

    colors_to_use = jnp.array(tuple(jnp.logical_and(filters_to_use[i], filters_to_use[i + 1]) for i in range(len(filters_to_use) - 1)))

    ab_colors = jnp.where(colors_to_use, c_ab, jnp.nan)
    ab_cols_errs = jnp.where(colors_to_use, c_ab_err, jnp.nan)

    use_i = filters_to_use[id_i_band]
    i_mag_ab = jnp.where(use_i, i_ab, jnp.nan)

    return i_mag_ab, ab_colors, ab_cols_errs


vmap_mags_to_i_and_colors = vmap(mags_to_i_and_colors, in_axes=(0, 0, None))


@jit
def col_to_fluxRatio(obs, ref, err):
    r"""col_to_fluxRatio Computes the equivalent data in flux (linear) space from the input in magnitude (logarithmic) space.
    Useful to switch from a $\chi^2$ minimisation in color-space or in flux space.

    :param obs: Observed color index
    :type obs: float or array
    :param ref: Reference (template) color index
    :type ref: float or array
    :param err: Observed noise (aka errors, dispersion)
    :type err: float or array
    :return: $\left( 10^{-0.4 obs}, 10^{-0.4 ref}, 10^{-0.4 err} \right)$
    :rtype: 3-tuple of (float or array)
    """
    obs_f = jnp.power(10.0, -0.4 * obs)
    ref_f = jnp.power(10.0, -0.4 * ref)
    err_f = obs_f * (jnp.power(10.0, -0.4 * err) - 1)  # coindetable
    return obs_f, ref_f, err_f


@jit
def chi_term(obs, ref, err):
    r"""chi_term Compute one term in the $\chi^2$ formula, *i.e.* for one photometric band.

    :param obs: Observed color index
    :type obs: float or array
    :param ref: Reference (template) color index
    :type ref: float or array
    :param err: Observed noise (aka errors, dispersion)
    :type err: float or array
    :return: $\left( \frac{obs-ref}{err} \right)^2$
    :rtype: float or array
    """
    return jnp.power((obs - ref) / err, 2.0)


vmap_chi_term = vmap(chi_term, in_axes=(None, 0, None))  # vmap version to compute the chi value for all colors of a single template, i.e. for all redshifts values


@jit
def z_prior_val(i_mag, zp, nuvk):
    """z_prior_val Computes the prior value for the given combination of observation, template and redshift.

    :param i_mag: Observed magnitude in reference (i) band
    :type i_mag: float
    :param zp: Redshift at which the probability, here the prior, is evaluated
    :type zp: float
    :param nuvk: Templates' NUV-NIR color index, in restframe
    :type nuvk: float
    :return: Prior probability of the redshift zp for this observation, if represented by the given template
    :rtype: float
    """
    alpt0, zot, kt, pcal, ktf_m, ft_m = prior_alpt0(nuvk), prior_zot(nuvk), prior_kt(nuvk), prior_pcal(nuvk), prior_ktf(nuvk), prior_ft(nuvk)
    val_prior = nz_prior_core(zp, i_mag, alpt0, zot, kt, pcal, ktf_m, ft_m)
    return val_prior


vmap_nz_prior = vmap(
    vmap(z_prior_val, in_axes=(0, None, None)),  # vmap version to compute the prior value for a certain observation and a certain SED template at all redshifts
    in_axes=(None, 0, 0),
)


@jit
def val_neg_log_posterior(z_val, templ_cols, gal_cols, gel_colerrs, gal_iab, templ_nuvk):
    r"""val_neg_log_posterior Computes the negative log posterior (posterior = likelihood * prior) probability of the redshift for an observation, given a template galaxy.
    This corresponds to a reduced $\chi^2$ value in which the prior has been injected.

    :param z_val: Redshift at which the probability, here the posterior, is evaluated
    :type z_val: float
    :param templ_cols: Color indices of the galaxy template
    :type templ_cols: array of floats
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :param gal_iab: Observed magnitude in reference (i) band
    :type gal_iab: float
    :param templ_nuvk: Templates' NUV-NIR color index, in restframe
    :type templ_nuvk: float
    :return: Posterior probability of the redshift zp for this observation, if represented by the given template
    :rtype: float
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    chi = jnp.where(jnp.isfinite(_chi), _chi, 0.0)
    _count = jnp.sum(jnp.where(jnp.isfinite(_chi), 1.0, 0.0))
    _prior = z_prior_val(gal_iab, z_val, templ_nuvk)
    return jnp.sum(chi) / _count - 2 * jnp.log(_prior)


vmap_neg_log_posterior = vmap(
    vmap(
        val_neg_log_posterior,
        in_axes=(None, None, 0, 0, 0, None),  # Same as above but for all observations...
    ),
    in_axes=(0, 0, None, None, None, 0),  # ... and for all redshifts
)


@jit
def neg_log_posterior(sps_temp, obs_ab_colors, obs_ab_colerrs, obs_iab, z_grid, nuvk):
    r"""neg_log_posterior Computes the posterior distribution of redshifts (negative log posterior, similar to a $\chi^2$) with one template for all observations.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: negative log posterior values along the redshift grid
    :rtype: jax array
    """
    neglog_post = vmap_neg_log_posterior(z_grid, sps_temp, obs_ab_colors, obs_ab_colerrs, obs_iab, nuvk)
    return neglog_post


@jit
def val_neg_log_likelihood(templ_cols, gal_cols, gel_colerrs):
    r"""val_neg_log_likelihood Computes the negative log likelihood of the redshift with one template for all observations.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_cols: Color indices of the galaxy template
    :type templ_cols: array of floats
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp, if represented by the given template.
    :rtype: float
    """
    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    chi = jnp.where(jnp.isfinite(_chi), _chi, 0.0)
    _count = jnp.sum(jnp.where(jnp.isfinite(_chi), 1.0, 0.0))
    return jnp.sum(chi) / _count


vmap_neg_log_likelihood = vmap(
    vmap(val_neg_log_likelihood, in_axes=(None, 0, 0)),  # Same as above but for all observations...
    in_axes=(0, None, None),  # ... and for all redshifts
)


@jit
def neg_log_likelihood(sps_temp, obs_ab_colors, obs_ab_colerrs):
    r"""neg_log_likelihood Computes the negative log likelihood of redshifts (aka $\chi^2$) with one template for all observations.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: likelihood values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right)$) along the redshift grid
    :rtype: jax array
    """
    neglog_lik = vmap_neg_log_likelihood(sps_temp, obs_ab_colors, obs_ab_colerrs)
    return neglog_lik


@jit
def likelihood(sps_temp, obs_ab_colors, obs_ab_colerrs):
    r"""likelihood Computes the likelihood of redshifts with one template for all observations.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: likelihood values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right)$) along the redshift grid
    :rtype: jax array
    """
    neglog_lik = vmap_neg_log_likelihood(sps_temp, obs_ab_colors, obs_ab_colerrs)
    return jnp.exp(-0.5 * neglog_lik)


def likelihood_fluxRatio(sps_temp, obs_ab_colors, obs_ab_colerrs):
    r"""likelihood_fluxRatio Computes the likelihood of redshifts with one template for all observations.
    Uses the $\chi^2$ distribution in flux-ratio space instead of color space.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: likelihood values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right)$) along the redshift grid
    :rtype: jax array
    """
    obs, ref, err = col_to_fluxRatio(obs_ab_colors, sps_temp, obs_ab_colerrs)
    neglog_lik = vmap_neg_log_likelihood(ref, obs, err)
    return jnp.exp(-0.5 * neglog_lik)


# @jit
def posterior(sps_temp, obs_ab_colors, obs_ab_colerrs, obs_iab, z_grid, nuvk):
    r"""posterior Computes the posterior distribution of redshifts with one template for all observations.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: posterior probability values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right) \times prior$) along the redshift grid
    :rtype: jax array
    """
    chi2_arr = vmap_neg_log_likelihood(sps_temp, obs_ab_colors, obs_ab_colerrs)
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    prior_val = vmap_nz_prior(obs_iab, z_grid, nuvk)
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_val
    return res


def posterior_fluxRatio(sps_temp, obs_ab_colors, obs_ab_colerrs, obs_iab, z_grid, nuvk):
    r"""posterior_fluxRatio Computes the posterior distribution of redshifts with one template for all observations.
    Uses the $\chi^2$ distribution in flux-ratio space instead of color space.

    :param sps_temp: Colors of SPS template to be used as reference
    :type sps_temp: array of shape (n_redshift, n_colors)
    :param obs_ab_colors: Observed colors for all galaxies
    :type obs_ab_colors: array of shape (n_galaxies, n_colors)
    :param obs_ab_colerrs: Observed noise of colors for all galaxies
    :type obs_ab_colerrs: array of shape (n_galaxies, n_colors)
    :return: posterior probability values (*i.e.* $\exp \left( - \frac{\chi^2}{2} \right) \times prior$) along the redshift grid
    :rtype: jax array
    """
    obs, ref, err = col_to_fluxRatio(obs_ab_colors, sps_temp, obs_ab_colerrs)
    chi2_arr = vmap_neg_log_likelihood(ref, obs, err)
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    prior_val = vmap_nz_prior(obs_iab, z_grid, nuvk)
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_val
    return res
