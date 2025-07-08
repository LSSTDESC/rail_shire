#!/bin/env python3

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from .cosmology import nz_prior_core, prior_alpt0, prior_ft, prior_kt, prior_ktf, prior_pcal, prior_zot


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


@jit
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
    i_ab = mags_arr.at[id_i_band].get()

    filters_to_use = jnp.logical_and(jnp.isfinite(mags_arr), jnp.isfinite(mags_err_arr))

    colors_to_use = jnp.array(tuple(jnp.logical_and(filters_to_use[i], filters_to_use[i + 1]) for i in range(len(filters_to_use) - 1)))

    ab_colors = jnp.where(colors_to_use, c_ab, jnp.nan)
    ab_cols_errs = jnp.where(colors_to_use, c_ab_err, jnp.nan)

    use_i = filters_to_use.at[id_i_band].get()
    i_mag_ab = jnp.where(use_i, i_ab, jnp.nan)

    return i_mag_ab, ab_colors, ab_cols_errs


vmap_mags_to_i_and_colors = vmap(mags_to_i_and_colors, in_axes=(0, 0, None))


@jit
def mags_to_i_and_icolors(mags_arr, mags_err_arr, id_i_band):
    """mags_to_i_and_icolors Extracts magnitude in i-band and computes color indices ( := mag - mag_i ) and associated errors for photo-z estimation for a single observed galaxy (input).

    :param mags_arr: AB-magnitudes of the galaxy from the input catalog
    :type mags_arr: jax.array
    :param mags_err_arr: Errors in AB-magnitudes of the galaxy from the input catalog
    :type mags_err_arr: jax.array
    :param id_i_band: Identifier of the i-band in the input catalog's photometric system. Starts from 0, such as `i_mag = mags_arr[id_i_band]`.
    :type id_i_band: int
    :return: 3-tuple of JAX arrays containing processed input data : (i_mag ; color indices ; errors on color indices)
    :rtype: tuple of jax.array
    """
    filters_to_use = jnp.logical_and(jnp.isfinite(mags_arr), jnp.isfinite(mags_err_arr))
    use_i = filters_to_use.at[id_i_band].get()
    # colors_to_use = jnp.logical_and(filters_to_use, [k != id_i_band for k in range(len(mags_arr))])

    i_ab = mags_arr.at[id_i_band].get()
    i_ab_err = mags_err_arr.at[id_i_band].get()
    c_ab = mags_arr - i_ab
    c_ab_err = jnp.power(jnp.power(mags_err_arr, 2) + jnp.power(i_ab_err, 2), 0.5)

    ab_colors = jnp.where(filters_to_use, c_ab, jnp.nan)
    ab_cols_errs = jnp.where(filters_to_use, c_ab_err, jnp.nan)
    i_mag_ab = jnp.where(use_i, i_ab, jnp.nan)

    return i_mag_ab, ab_colors, ab_cols_errs


vmap_mags_to_i_and_icolors = vmap(mags_to_i_and_icolors, in_axes=(0, 0, None))


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
    vmap(
        vmap(z_prior_val, in_axes=(0, None, None)),  # vmap version to compute the prior value for all observations
        in_axes=(None, None, 0),  # and a certain SED template at all dust attenuations
    ),
    in_axes=(None, 0, None),  # and at all redshifts
)


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
    vmap(
        vmap(val_neg_log_likelihood, in_axes=(None, 0, 0)),  # Same as above but for all observations...
        in_axes=(0, None, None),  # ... and all dust attenuations
    ),
    in_axes=(0, None, None),  # ... and for all redshifts
)


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
    pz = jnp.exp(-0.5 * neglog_lik)
    return jnp.nanmax(pz, axis=1) #, or jnp.nanargmax(pz, axis=1) sps_temp[:, jnp.nanargmax(pz, axis=1), 0]


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
    return jnp.nanmin(neglog_lik, axis=1)


@jit
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
    return jnp.nanmax(jnp.exp(-0.5 * neglog_lik), axis=1)


@jit
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
    return jnp.nanmax(res, axis=1)


@jit
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
    return jnp.nanmax(res, axis=1)


## Free A_nu (dust) and fit on SPS params instead of template colors
@jit
def val_neg_log_likelihood_pars_z_av(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data):
    r"""val_neg_log_likelihood_pars_z_av Computes the negative log likelihood of the redshift with one template for all observations.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_pars: SPS parameters of the galaxy template
    :type templ_pars: array of floats
    :param z: Redshift value
    :type z: float
    :param av: Dust law attenuation parameter
    :type z: float
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp, if represented by the given template.
    :rtype: float
    """
    from .template import mean_colors

    pars = templ_pars.at[13].set(av)
    templ_cols = mean_colors(pars, wls, filt_trans_arr, z, ssp_data)

    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    chi = jnp.where(jnp.isfinite(_chi), _chi, 0.0)
    _count = jnp.sum(jnp.where(jnp.isfinite(_chi), 1.0, 0.0))
    return jnp.sum(chi) / _count


vmap_obs_nllik = vmap(val_neg_log_likelihood_pars_z_av, in_axes=(None, None, None, 0, 0, None, None, None))
vmap_av_nllik = vmap(vmap_obs_nllik, in_axes=(None, None, 0, None, None, None, None, None))


@partial(vmap, in_axes=(None, 0, None, None, None, None, None, None))
def vmap_z_nllik(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data):
    r"""vmap_z_nllik Computes the negative log likelihood of the redshift with one template for all observations, vmapped on redshifts and dust absorptions.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_pars: SPS parameters of the galaxy template
    :type templ_pars: array of floats
    :param z: Redshift value
    :type z: float
    :param av: Dust law attenuation parameter
    :type z: float
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp, if represented by the given template.
    :rtype: float
    """
    return jnp.nanmin(vmap_av_nllik(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data), axis=0)


vmap_templ_nllik = vmap(vmap_z_nllik, in_axes=(0, None, None, None, None, None, None, None))


## Free A_nu (dust) and fit on SPS params instead of template colors
@jit
def val_neg_log_likelihood_pars_z_av_iclrs(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data, iband_num):
    r"""val_neg_log_likelihood_pars_z_av_iclrs Computes the negative log likelihood of the redshift with one template for all observations.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_pars: SPS parameters of the galaxy template
    :type templ_pars: array of floats
    :param z: Redshift value
    :type z: float
    :param av: Dust law attenuation parameter
    :type z: float
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp, if represented by the given template.
    :rtype: float
    """
    from .template import mean_icolors

    pars = templ_pars.at[13].set(av)
    templ_cols = mean_icolors(pars, wls, filt_trans_arr, z, ssp_data, iband_num)

    _chi = chi_term(gal_cols, templ_cols, gel_colerrs)
    chi = jnp.where(jnp.isfinite(_chi), _chi, 0.0)
    _count = jnp.sum(jnp.where(jnp.isfinite(_chi), 1.0, 0.0))
    return jnp.sum(chi) / _count


vmap_obs_nllik_iclrs = vmap(val_neg_log_likelihood_pars_z_av_iclrs, in_axes=(None, None, None, 0, 0, None, None, None, None))
vmap_av_nllik_iclrs = vmap(vmap_obs_nllik_iclrs, in_axes=(None, None, 0, None, None, None, None, None, None))


@partial(vmap, in_axes=(None, 0, None, None, None, None, None, None, None))
def vmap_z_nllik_iclrs(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data, iband_num):
    r"""vmap_z_nllik_iclrs Computes the negative log likelihood of the redshift with one template for all observations, vmapped on redshifts and dust absorptions.
    This is a reduced $\chi^2$ and does not use a prior probability distribution.

    :param templ_pars: SPS parameters of the galaxy template
    :type templ_pars: array of floats
    :param z: Redshift value
    :type z: float
    :param av: Dust law attenuation parameter
    :type z: float
    :param gal_cols: Color indices of the observed object
    :type gal_cols: array of floats
    :param gel_colerrs: Color errors/dispersion/noise of the observed object
    :type gel_colerrs: array of floats
    :return: Likelihood of the redshift zp, if represented by the given template.
    :rtype: float
    """
    return jnp.nanmin(vmap_av_nllik_iclrs(templ_pars, z, av, gal_cols, gel_colerrs, wls, filt_trans_arr, ssp_data, iband_num), axis=0)


vmap_templ_nllik_iclrs = vmap(vmap_z_nllik_iclrs, in_axes=(0, None, None, None, None, None, None, None, None))


@jit
def val_prior_pars_z_av(templ_pars, z, av, gal_iab, wls, nuvk_trans_arr, ssp_data):
    """val_prior_pars_z_av _summary_

    :param templ_pars: _description_
    :type templ_pars: _type_
    :param z: _description_
    :type z: _type_
    :param av: _description_
    :type av: _type_
    :param gal_iab: _description_
    :type gal_iab: _type_
    :param wls: _description_
    :type wls: _type_
    :param nuvk_trans_arr: _description_
    :type nuvk_trans_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    from .template import ssp_spectrum_fromparam, vmap_calc_rest_mag

    _pars = templ_pars.at[13].set(av)
    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(_pars, z, ssp_data)
    _mags = vmap_calc_rest_mag(ssp_wave, sed_attenuated, wls, nuvk_trans_arr)
    nuvk = _mags.at[0].get() - _mags.at[1].get()
    nz_prior = z_prior_val(gal_iab, z, nuvk)
    return nz_prior


vmap_obs_prior_pars_zav = vmap(val_prior_pars_z_av, in_axes=(None, None, None, 0, None, None, None))
vmap_av_prior_pars_zav = vmap(vmap_obs_prior_pars_zav, in_axes=(None, None, 0, None, None, None, None))
vmap_z_prior_pars_zav = vmap(vmap_av_prior_pars_zav, in_axes=(None, 0, None, None, None, None, None))
vmap_templ_prior_pars_zav = vmap(vmap_z_prior_pars_zav, in_axes=(0, None, None, None, None, None, None))


@jit
def likelihood_pars_z_av(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr, ssp_data):
    """likelihood_pars_z_av _summary_

    :param templ_pars_arr: _description_
    :type templ_pars_arr: _type_
    :param z_arr: _description_
    :type z_arr: _type_
    :param av_arr: _description_
    :type av_arr: _type_
    :param obs_i_cols_arr: _description_
    :type obs_i_cols_arr: _type_
    :param obs_i_colerrs_arr: _description_
    :type obs_i_colerrs_arr: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param iband_num: _description_
    :type iband_num: _type_
    :return: _description_
    :rtype: _type_
    """
    neglog_lik = vmap_templ_nllik(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr, ssp_data)
    return jnp.exp(-0.5 * neglog_lik)


@jit
def posterior_pars_z_av(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, obs_iab, wls, filt_trans_arr, ssp_data):
    """likelihood_pars_z_av _summary_

    :param templ_pars_arr: _description_
    :type templ_pars_arr: _type_
    :param z_arr: _description_
    :type z_arr: _type_
    :param av_arr: _description_
    :type av_arr: _type_
    :param obs_i_cols_arr: _description_
    :type obs_i_cols_arr: _type_
    :param obs_i_colerrs_arr: _description_
    :type obs_i_colerrs_arr: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param iband_num: _description_
    :type iband_num: _type_
    :return: _description_
    :rtype: _type_
    """
    chi2_arr = vmap_templ_nllik(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr[:-2, :], ssp_data)
    prior_arr = vmap_templ_prior_pars_zav(templ_pars_arr, z_arr, av_arr, obs_iab, wls, filt_trans_arr[-2:, :], ssp_data)
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_arr
    return res


@jit
def likelihood_pars_z_av_iclrs(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr, ssp_data, iband_num):
    """likelihood_pars_z_av_iclrs _summary_

    :param templ_pars_arr: _description_
    :type templ_pars_arr: _type_
    :param z_arr: _description_
    :type z_arr: _type_
    :param av_arr: _description_
    :type av_arr: _type_
    :param obs_i_cols_arr: _description_
    :type obs_i_cols_arr: _type_
    :param obs_i_colerrs_arr: _description_
    :type obs_i_colerrs_arr: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param iband_num: _description_
    :type iband_num: _type_
    :return: _description_
    :rtype: _type_
    """
    neglog_lik = vmap_templ_nllik_iclrs(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr, ssp_data, iband_num)
    return jnp.exp(-0.5 * neglog_lik)


@jit
def posterior_pars_z_av_iclrs(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, obs_iab, wls, filt_trans_arr, ssp_data, iband_num):
    """posterior_pars_z_av_iclrs _summary_

    :param templ_pars_arr: _description_
    :type templ_pars_arr: _type_
    :param z_arr: _description_
    :type z_arr: _type_
    :param av_arr: _description_
    :type av_arr: _type_
    :param obs_i_cols_arr: _description_
    :type obs_i_cols_arr: _type_
    :param obs_i_colerrs_arr: _description_
    :type obs_i_colerrs_arr: _type_
    :param obs_iab: _description_
    :type obs_iab: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :param iband_num: _description_
    :type iband_num: _type_
    :return: _description_
    :rtype: _type_
    """
    chi2_arr = vmap_templ_nllik_iclrs(templ_pars_arr, z_arr, av_arr, obs_i_cols_arr, obs_i_colerrs_arr, wls, filt_trans_arr[:-2, :], ssp_data, iband_num)
    prior_arr = vmap_templ_prior_pars_zav(templ_pars_arr, z_arr, av_arr, obs_iab, wls, filt_trans_arr[-2:, :], ssp_data)
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    neglog_lik = _n1 * chi2_arr
    res = jnp.power(jnp.exp(-0.5 * neglog_lik), 1 / _n1) * prior_arr
    return res
