#!/usr/bin/env python3
"""
Module to specify and use SED templates for photometric redshifts estimation algorithms.
Insipired by previous developments in [process_fors2](https://github.com/JospehCeh/process_fors2).

Created on Thu Aug 1 12:59:33 2024

@author: joseph
"""

from collections import namedtuple
from functools import partial

import jax
from jax import jit, vmap
from jax.tree_util import tree_map
from jax import numpy as jnp

from diffmah.defaults import DiffmahParams
from diffstar import calc_sfh_singlegal  # sfh_singlegal
from diffstar.defaults import DiffstarUParams  # , DEFAULT_Q_PARAMS
from rail.dsps import calc_obs_mag, calc_rest_mag, DEFAULT_COSMOLOGY, age_at_z
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from interpax import interp1d
try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid
from astropy import constants as const

from .met_weights_age_dep import calc_rest_sed_sfh_table_lognormal_mdf_agedep

TODAY_GYR = 13.8
T_ARR = jnp.linspace(0.1, TODAY_GYR, 100)
C_KMS = (const.c).to("km/s").value  # km/s

BaseTemplate = namedtuple("BaseTemplate", ["name", "flux", "z_sps"])
SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"])


@jit
def mean_sfr(params):
    """Model of the SFR

    :param params: Fitted parameter dictionnary
    :type params: float as a dictionnary

    :return: array of the star formation rate
    :rtype: float

    """
    # decode the parameters
    MAH_lgmO = params["MAH_lgmO"]
    MAH_logtc = params["MAH_logtc"]  # DEFAULT_MAH_PARAMS[1]
    MAH_early_index = params["MAH_early_index"]  # DEFAULT_MAH_PARAMS[2]
    MAH_late_index = params["MAH_late_index"]
    list_param_mah = [MAH_lgmO, MAH_logtc, MAH_early_index, MAH_late_index]

    MS_lgmcrit = params["MS_lgmcrit"]
    MS_lgy_at_mcrit = params["MS_lgy_at_mcrit"]  # DEFAULT_MS_PARAMS[1]
    MS_indx_lo = params["MS_indx_lo"]
    MS_indx_hi = params["MS_indx_hi"]
    MS_tau_dep = params["MS_tau_dep"]  # DEFAULT_MS_PARAMS[4]
    list_param_ms = [MS_lgmcrit, MS_lgy_at_mcrit, MS_indx_lo, MS_indx_hi, MS_tau_dep]

    Q_lg_qt = params["Q_lg_qt"]
    Q_qlglgdt = params["Q_qlglgdt"]
    Q_lg_drop = params["Q_lg_drop"]
    Q_lg_rejuv = params["Q_lg_rejuv"]
    list_param_q = [Q_lg_qt, Q_qlglgdt, Q_lg_drop, Q_lg_rejuv]

    # compute SFR
    tup_param_sfh = DiffstarUParams(tuple(list_param_ms), tuple(list_param_q))
    tup_param_mah = DiffmahParams(*list_param_mah)

    # tarr = np.linspace(0.1, TODAY_GYR, 100)
    # sfh_gal = sfh_singlegal(tarr, list_param_mah , list_param_ms, list_param_q,\
    #                        ms_param_type="unbounded", q_param_type="unbounded"\
    #                       )

    sfh_gal = calc_sfh_singlegal(tup_param_sfh, tup_param_mah, T_ARR)

    # clear sfh in future
    # sfh_gal = jnp.where(tarr<t_obs, sfh_gal, 0)

    return sfh_gal


@jit
def ssp_spectrum_fromparam(params, z_obs, ssp_data):
    """Return the SED of SSP DSPS with original wavelength range wihout and with dust

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float

    :param ssp_data: SSP library as loaded from DSPS
    :type ssp_data: namedtuple

    :return: the wavelength and the spectrum with dust and no dust
    :rtype: float

    """
    # compute the SFR
    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
    t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

    gal_sfr_table = mean_sfr(params)

    # age-dependant metallicity
    gal_lgmet_young = 2.0  # log10(Z)
    gal_lgmet_old = -3.0  # params["LGMET_OLD"] # log10(Z)
    gal_lgmet_scatter = 0.2  # params["LGMETSCATTER"] # lognormal scatter in the metallicity distribution function

    # compute the SED_info object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf_agedep(
        T_ARR, gal_sfr_table, gal_lgmet_young, gal_lgmet_old, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs
    )
    # dust attenuation parameters
    Av = params["AV"]
    uv_bump = params["UV_BUMP"]
    plaw_slope = params["PLAW_SLOPE"]
    # list_param_dust = [Av, uv_bump, plaw_slope]

    # compute dust attenuation
    wave_spec_micron = ssp_data.ssp_wave / 10000
    k = sbl18_k_lambda(wave_spec_micron, uv_bump, plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k, Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    return ssp_data.ssp_wave, sed_info.rest_sed, sed_attenuated


@partial(vmap, in_axes=(None, None, 0, 0, None))
def _calc_mag(ssp_wls, sed_fnu, filt_wls, filt_transm, z_obs):
    return calc_obs_mag(ssp_wls, sed_fnu, filt_wls, filt_transm, z_obs, *DEFAULT_COSMOLOGY)


@jit
def mean_mags(X, params, z_obs, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library as loaded from DSPS
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    mags_predictions = jax.tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters, list_transm_filters)

    mags_predictions = jnp.array(mags_predictions)

    return mags_predictions


@jit
def mean_colors(X, params, z_obs, ssp_data):
    """mean_colors returns the photometric magnitudes for the given filters transmission in X : predict the magnitudes in filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library as loaded from DSPS
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float
    """
    mags = mean_mags(X, params, z_obs, ssp_data)
    return mags[:-1] - mags[1:]


@jit
def mean_spectrum(wls, params, z_obs, ssp_data):
    """Return the Model of SSP spectrum including Dust at the wavelength wls

    :param wls: wavelengths of the spectrum in rest frame
    :type wls: float

    :param params: parameters for the fit
    :type params: dictionnary of parameters

    :param z_obs: redshift at which the model SSP should be calculated
    :type z_obs: float

    :param ssp_data: SSP library as loaded from DSPS
    :type ssp_data: namedtuple

    :return: the spectrum
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # interpolate with interpax which is differentiable
    # Fobs = jnp.interp(wls, ssp_data.ssp_wave, sed_attenuated)
    Fobs = interp1d(wls, ssp_wave, sed_attenuated, method="cubic")

    return Fobs


@partial(vmap, in_axes=(None, None, 0))
def calc_eqw(sur_wls, sur_spec, lin):
    r"""
    Computes the equivalent width of the specified spectral line.

    Parameters
    ----------
    p : array
        SPS parameters' values - should be an output of a fitting procedure, *e.g.* `results.params`.
    sur_wls : array
        Wavelengths in angstrom - should be oversampled so that spectral lines can be sampled with a sufficiently high resolution (step of 0.1 angstrom is recommended)
    sur_spec : array
        Flux densities in Lsun/Hz - should be oversampled to match `sur_wls`.
    lin : int or float
        Central wavelength (in angstrom) of the line to be studied.

    Returns
    -------
    float
        Value of the nequivalent width of spectral line at $\lambda=$`lin`.
    """
    line_wid = lin * 300 / C_KMS / 2
    cont_wid = lin * 1500 / C_KMS / 2
    nancont = jnp.where((sur_wls > lin - cont_wid) * (sur_wls < lin - line_wid) + (sur_wls > lin + line_wid) * (sur_wls < lin + cont_wid), sur_spec, jnp.nan)
    height = jnp.nanmean(nancont)
    vals = jnp.where((sur_wls > lin - line_wid) * (sur_wls < lin + line_wid), sur_spec / height - 1.0, 0.0)
    ew = trapezoid(vals, x=sur_wls)
    return ew


@jit
def templ_mags(X, params, z_obs, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    obs_mags = tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters[:-2], list_transm_filters[:-2])
    rest_mags = tree_map(lambda x, y: calc_rest_mag(ssp_wave, sed_attenuated, x, y), list_wls_filters[-2:], list_transm_filters[-2:])

    mags_predictions = jnp.concatenate((jnp.array(obs_mags), jnp.array(rest_mags)))

    return mags_predictions


v_mags = vmap(templ_mags, in_axes=(None, None, 0, None))


# @jit
def calc_nuvk(wls, params_dict, zobs, ssp_data):
    """calc_nuvk Computes the theoretical emitted NUV-IR color index of a reference galaxy.

    :param wls: Wavelengths
    :type wls: array
    :param params_dict: DSPS input parameters to compute the restframe NUV and NIR photometry.
    :type params_dict: dict
    :param zobs: Redshift value
    :type zobs: float
    :return: NUV-NIR color index
    :rtype: float
    """
    from .filter import NIR_filt, NUV_filt, ab_mag

    rest_sed = mean_spectrum(wls, params_dict, zobs, ssp_data)
    nuv = ab_mag(NUV_filt.wavelengths, NUV_filt.transmission, wls, rest_sed)
    nir = ab_mag(NIR_filt.wavelengths, NIR_filt.transmission, wls, rest_sed)
    return nuv - nir


v_nuvk = vmap(calc_nuvk, in_axes=(None, None, 0, None))


def make_sps_templates(params_dict, filt_tup, redz, ssp_data, id_imag=3):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_dict: DSPS input parameters
    :type params_dict: dict
    :param filt_tup: Filters in which to compute the photometry of the templates, given as a tuple of two arrays : one for wavelengths, one for transmissions.
    :type filt_tup: tuple of arrays
    :param redz: redshift grid on which to compute the templates photometry
    :type redz: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: SPS_Templates object (namedtuple)
    """
    name = params_dict.pop("tag")
    z_sps = params_dict.pop("redshift")
    template_mags = v_mags(filt_tup, params_dict, redz, ssp_data)
    nuvk = template_mags[:, -2] - template_mags[:, -1]
    colors = template_mags[:, :-3] - template_mags[:, 1:-2]
    i_mag = template_mags[:, id_imag]
    return SPS_Templates(name, z_sps, redz, i_mag, colors, nuvk)


@jit
def templ_mags_legacy(X, params, z_ref, z_obs, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters

    :param X: Tuple of filters to be used (Galex, SDSS, Vircam)
    :type X: a 2-tuple of lists (one element is a list of wavelengths and the other is a list of corresponding transmissions - each element of these lists corresponds to a filter).

    :param params: Model parameters
    :type params: Dictionnary of parameters

    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float

    :param z_obs: redshift of the observations
    :type z_obs: float

    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: float

    """

    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_ref, ssp_data)

    # decode the two lists
    list_wls_filters = X[0]
    list_transm_filters = X[1]

    obs_mags = tree_map(lambda x, y: calc_obs_mag(ssp_wave, sed_attenuated, x, y, z_obs, *DEFAULT_COSMOLOGY), list_wls_filters[:-2], list_transm_filters[:-2])
    rest_mags = tree_map(lambda x, y: calc_rest_mag(ssp_wave, sed_attenuated, x, y), list_wls_filters[-2:], list_transm_filters[-2:])

    mags_predictions = jnp.concatenate((jnp.array(obs_mags), jnp.array(rest_mags)))

    return mags_predictions


v_mags_legacy = vmap(templ_mags_legacy, in_axes=(None, None, None, 0, None))


def make_legacy_templates(params_dict, filt_tup, redz, ssp_data, id_imag=3):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.
    Contrary to `make_sps_template`, this methods only shifts the restframe SED and does not reevaluate the stellar population at each redshift.
    Mainly used for comparative studies as other existing photoZ codes such as BPZ and LEPHARE will do this and more.

    :param params_dict: DSPS input parameters
    :type params_dict: dict
    :param filt_tup: Filters in which to compute the photometry of the templates, given as a tuple of two arrays : one for wavelengths, one for transmissions.
    :type filt_tup: tuple of arrays
    :param redz: redshift grid on which to compute the templates photometry
    :type redz: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, NOT accounting for the Star Formation History up to the redshift value.
    :rtype: SPS_Templates object (namedtuple)
    """
    name = params_dict.pop("tag")
    z_sps = params_dict.pop("redshift")
    template_mags = v_mags_legacy(filt_tup, params_dict, z_sps, redz, ssp_data)
    nuvk = template_mags[:, -2] - template_mags[:, -1]
    colors = template_mags[:, :-3] - template_mags[:, 1:-2]
    i_mag = template_mags[:, id_imag]
    return SPS_Templates(name, z_sps, redz, i_mag, colors, nuvk)
