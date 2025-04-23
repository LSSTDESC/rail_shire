#!/usr/bin/env python3
"""
Module to specify and use SED templates for photometric redshifts estimation algorithms.
Insipired by previous developments in [process_fors2](https://github.com/JospehCeh/process_fors2).

Created on Thu Aug 1 12:59:33 2024

@author: joseph
"""

from functools import partial

import jax
from jax import jit, vmap
from jax.tree_util import tree_map
from jax import numpy as jnp

from diffmah.defaults import DiffmahParams
from diffstar import calc_sfh_singlegal  # sfh_singlegal
from diffstar.defaults import DiffstarUParams  # , DEFAULT_Q_PARAMS
from rail.dsps import calc_obs_mag, calc_rest_mag, DEFAULT_COSMOLOGY, age_at_z
from dsps.cosmology import age_at_z0
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from interpax import interp1d
try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid

from .met_weights_age_dep import calc_rest_sed_sfh_table_lognormal_mdf_agedep
from .io_utils import istuple
from .analysis import C_KMS, lsunPerHz_to_flam_noU
from .filter import NUV_filt, NIR_filt

jax.config.update("jax_enable_x64", True)

TODAY_GYR = age_at_z0(*DEFAULT_COSMOLOGY)  # 13.8
T_ARR = jnp.linspace(0.1, TODAY_GYR, 100)


@jit
def mean_sfr(params):
    """Model of the SFR

    :param params: Fitted parameters array
    :type params: array of floats

    :return: array of the star formation rate
    :rtype: array

    """
    # decode the parameters
    param_mah = params[:4]
    param_ms = params[4:9]
    param_q = params[9:13]

    # compute SFR
    tup_param_sfh = DiffstarUParams(param_ms, param_q)
    tup_param_mah = DiffmahParams(*param_mah)

    sfh_gal = calc_sfh_singlegal(tup_param_sfh, tup_param_mah, T_ARR)

    return sfh_gal


vmap_mean_sfr = vmap(mean_sfr)


@jit
def ssp_spectrum_fromparam(params, z_obs, ssp_data):
    """ssp_spectrum_fromparam _summary_

    :param params: _description_
    :type params: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # compute the SFR
    # need age of universe when the light was emitted
    t_obs = age_at_z(z_obs, *DEFAULT_COSMOLOGY)  # age of the universe in Gyr at z_obs
    t_obs = t_obs[0]  # age_at_z function returns an array, but SED functions accept a float for this argument

    gal_sfr_table = mean_sfr(params)

    # age-dependant metallicity, log10(Z)
    gal_lgmet_young = params.at[16].get()  # 2.0
    gal_lgmet_old = params.at[17].get()  # -3.0  # params["LGMET_OLD"]
    gal_lgmet_scatter = 0.2  # params["LGMETSCATTER"] # lognormal scatter in the metallicity distribution function

    # compute the SED_info object
    sed_info = calc_rest_sed_sfh_table_lognormal_mdf_agedep(
        T_ARR, gal_sfr_table, gal_lgmet_young, gal_lgmet_old, gal_lgmet_scatter, ssp_data.ssp_lgmet, ssp_data.ssp_lg_age_gyr, ssp_data.ssp_flux, t_obs
    )
    # dust attenuation parameters
    Av = params.at[13].get()
    uv_bump = params.at[14].get()
    plaw_slope = params.at[15].get()
    # list_param_dust = [Av, uv_bump, plaw_slope]

    # compute dust attenuation
    wave_spec_micron = ssp_data.ssp_wave / 10000
    k = sbl18_k_lambda(wave_spec_micron, uv_bump, plaw_slope)
    dsps_flux_ratio = _frac_transmission_from_k_lambda(k, Av)

    sed_attenuated = dsps_flux_ratio * sed_info.rest_sed

    return ssp_data.ssp_wave, sed_info.rest_sed, sed_attenuated


@jit
def mean_spectrum(wls, params, z_obs, ssp_data):
    """mean_spectrum _summary_

    :param wls: _description_
    :type wls: _type_
    :param params: _description_
    :type params: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    # interpolate with interpax which is differentiable
    # Fobs = jnp.interp(wls, ssp_data.ssp_wave, sed_attenuated)
    Fobs = interp1d(wls, ssp_wave, sed_attenuated, method="akima", extrap=False)

    return Fobs


vmap_mean_spectrum = vmap(mean_spectrum, in_axes=(None, 0, 0, None))


@partial(vmap, in_axes=(None, None, None, 0, None))
def vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs):
    """vmap_calc_obs_mag _summary_

    :param ssp_wave: _description_
    :type ssp_wave: _type_
    :param sed_attenuated: _description_
    :type sed_attenuated: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :return: _description_
    :rtype: _type_
    """
    return calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs, *DEFAULT_COSMOLOGY)


@partial(vmap, in_axes=(None, None, None, 0))
def vmap_calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr):
    """vmap_calc_obs_mag _summary_

    :param ssp_wave: _description_
    :type ssp_wave: _type_
    :param sed_attenuated: _description_
    :type sed_attenuated: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :return: _description_
    :rtype: _type_
    """
    return calc_rest_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr)


@jit
def mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data):
    """mean_mags _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    # get the restframe spectra without and with dust attenuation
    ssp_wave, rest_sed, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)

    mags_predictions = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs)
    # mags_predictions = tree_map(
    #    lambda trans : calc_obs_mag(
    #        ssp_wave,
    #        sed_attenuated,
    #        wls,
    #        trans,
    #        z_obs,
    #        *DEFAULT_COSMOLOGY
    #    ),
    #    tuple(t for t in filt_trans_arr)
    # )

    return jnp.array(mags_predictions)


@jit
def mean_colors(params, wls, filt_trans_arr, z_obs, ssp_data):
    """mean_colors _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    mags = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    return mags[:-1] - mags[1:]


vmap_mean_mags = vmap(mean_mags, in_axes=(0, None, None, 0, None))

vmap_mean_colors = vmap(mean_colors, in_axes=(0, None, None, 0, None))


@jit
def mean_icolors(params, wls, filt_trans_arr, z_obs, ssp_data, iband_num):
    """mean_icolors _summary_

    :param params: _description_
    :type params: _type_
    :param wls: _description_
    :type wls: _type_
    :param filt_trans_arr: _description_
    :type filt_trans_arr: _type_
    :param z_obs: _description_
    :type z_obs: _type_
    :param ssp_data: _description_
    :type ssp_data: _type_
    :return: _description_
    :rtype: _type_
    """
    mags = mean_mags(params, wls, filt_trans_arr, z_obs, ssp_data)
    imag = mags.at[iband_num].get()
    return mags - imag


vmap_mean_icolors = vmap(mean_icolors, in_axes=(0, None, None, 0, None, None))


@jit
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
    sur_flam = lsunPerHz_to_flam_noU(sur_wls, sur_spec, 0.001)
    nancont = jnp.where(jnp.logical_or(jnp.logical_and(sur_wls > lin - cont_wid, sur_wls < lin - line_wid), jnp.logical_and(sur_wls > lin + line_wid, sur_wls < lin + cont_wid)), sur_flam, jnp.nan)
    height = jnp.nanmean(nancont)
    vals = jnp.where(jnp.logical_and(sur_wls > lin - line_wid, sur_wls < lin + line_wid), sur_flam / height - 1.0, 0.0)
    ew = trapezoid(vals, x=sur_wls)
    return ew


vmap_calc_eqw = vmap(calc_eqw, in_axes=(None, None, 0))


@jit
def templ_mags(params, wls, filt_trans_arr, z_obs, av, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    in X : predict the magnitudes in Filters
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: 1D JAX-array of floats of length (nb bands+2)
    """
    _pars = params.at[13].set(av)
    # get the restframe spectra without and with dust attenuation
    ssp_wave, _, sed_attenuated = ssp_spectrum_fromparam(_pars, z_obs, ssp_data)
    _mags = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs)
    _nuvk = calc_rest_mag(ssp_wave, sed_attenuated, NUV_filt.wavelength, NUV_filt.transmission) - calc_rest_mag(ssp_wave, sed_attenuated, NIR_filt.wavelength, NIR_filt.transmission)

    mags_predictions = jnp.concatenate((_mags, _nuvk))

    return mags_predictions


vmap_mags_av = vmap(templ_mags, in_axes=(None, None, None, None, 0, None))
vmap_mags_zobs = vmap(vmap_mags_av, in_axes=(None, None, None, 0, None, None))
vmap_mags_pars = vmap(vmap_mags_zobs, in_axes=(0, None, None, None, None, None))


def templ_clrs_nuvk(params, wls, filt_trans_arr, z_obs, av, ssp_data):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands-1), float)
    """
    _mags = templ_mags(params, wls, filt_trans_arr, z_obs, av, ssp_data)
    return _mags[:-3] - _mags[1:-2], _mags[-2] - _mags[-1]


vmap_clrs_av = vmap(templ_clrs_nuvk, in_axes=(None, None, None, None, 0, None))
vmap_clrs_zobs = vmap(vmap_clrs_av, in_axes=(None, None, None, 0, None, None))
vmap_clrs_pars = vmap(vmap_clrs_zobs, in_axes=(0, None, None, None, None, None))


def templ_iclrs_nuvk(params, wls, filt_trans_arr, z_obs, av, ssp_data, id_imag):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands), float)
    """
    _mags = templ_mags(params, wls, filt_trans_arr, z_obs, av, ssp_data)
    return _mags[:-2] - _mags[id_imag], _mags[-2] - _mags[-1]


vmap_iclrs_av = vmap(templ_iclrs_nuvk, in_axes=(None, None, None, None, 0, None, None))
vmap_iclrs_zobs = vmap(vmap_iclrs_av, in_axes=(None, None, None, 0, None, None, None))
vmap_iclrs_pars = vmap(vmap_iclrs_zobs, in_axes=(0, None, None, None, None, None, None))


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
    from .filter import ab_mag

    rest_sed = mean_spectrum(wls, params_dict, zobs, ssp_data)
    nuv = ab_mag(NUV_filt.wavelengths, NUV_filt.transmission, wls, rest_sed)
    nir = ab_mag(NIR_filt.wavelengths, NIR_filt.transmission, wls, rest_sed)
    return nuv - nir


v_nuvk = vmap(calc_nuvk, in_axes=(None, None, 0, None))


def get_colors_templates(params, wls, z_obs, transm_arr, ssp_data):
    ssp_wave, _, sed_attenuated = ssp_spectrum_fromparam(params, z_obs, ssp_data)
    _mags = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, transm_arr, z_obs)
    return _mags[:-1]-_mags[1:]

vmap_cols_zo = vmap(get_colors_templates, in_axes=(None, None, 0, None, None))
vmap_cols_templ = vmap(vmap_cols_zo, in_axes=(0, None, None, None, None))

def make_sps_templates(params_arr, wls, transm_arr, redz_arr, av_arr, ssp_data):
    """make_sps_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param av_arr: Attenuation grid on which to compute the templates photometry
    :type av_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-3] - template_mags[:, :, :, 1:-2]
    templ_tupl = [tuple(_pars) for _pars in params_arr]
    reslist_of_tupl = tree_map(lambda partup: vmap_clrs_zobs(jnp.array(partup), wls, transm_arr, redz_arr, av_arr, ssp_data), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_clrs_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    return reslist_of_tupl


def make_sps_itemplates(params_arr, wls, transm_arr, redz_arr, av_arr, ssp_data, id_imag=3):
    """make_sps_itemplates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param av_arr: Attenuation grid on which to compute the templates photometry
    :type av_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # i_mag = template_mags[:, :, :, id_imag]
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-2] - i_mag
    templ_tupl = [tuple(_pars) for _pars in params_arr]
    reslist_of_tupl = tree_map(lambda partup: vmap_iclrs_zobs(jnp.array(partup), wls, transm_arr, redz_arr, av_arr, ssp_data, id_imag), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_iclrs_pars(params_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data, id_imag)
    return reslist_of_tupl


@jit
def templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, av, ssp_data):
    """Return the photometric magnitudes for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: array the predicted magnitude for the SED spectrum model represented by its parameters.
    :rtype: 1D JAX-array of floats of length (nb bands+2)

    """
    _pars = params.at[13].set(av)
    # get the restframe spectra without and with dust attenuation
    ssp_wave, _, sed_attenuated = ssp_spectrum_fromparam(_pars, z_ref, ssp_data)
    _mags = vmap_calc_obs_mag(ssp_wave, sed_attenuated, wls, filt_trans_arr, z_obs)
    _nuvk = calc_rest_mag(ssp_wave, sed_attenuated, NUV_filt.wavelength, NUV_filt.transmission) - calc_rest_mag(ssp_wave, sed_attenuated, NIR_filt.wavelength, NIR_filt.transmission)

    mags_predictions = jnp.concatenate((_mags, _nuvk))

    return mags_predictions


vmap_mags_av_legacy = vmap(templ_mags_legacy, in_axes=(None, None, None, None, None, 0, None))
vmap_mags_zobs_legacy = vmap(vmap_mags_av_legacy, in_axes=(None, None, None, None, 0, None, None))
vmap_mags_pars_legacy = vmap(vmap_mags_zobs_legacy, in_axes=(0, 0, None, None, None, None, None))


def templ_clrs_nuvk_legacy(params, z_ref, wls, filt_trans_arr, z_obs, av, ssp_data):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands-1), float)
    """
    _mags = templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, av, ssp_data)
    return _mags[:-3] - _mags[1:-2], _mags[-2] - _mags[-1]


vmap_clrs_av_legacy = vmap(templ_clrs_nuvk_legacy, in_axes=(None, None, None, None, None, 0, None))
vmap_clrs_zobs_legacy = vmap(vmap_clrs_av_legacy, in_axes=(None, None, None, None, 0, None, None))
vmap_clrs_pars_legacy = vmap(vmap_clrs_zobs_legacy, in_axes=(0, 0, None, None, None, None, None))


def templ_iclrs_nuvk_legacy(params, z_ref, wls, filt_trans_arr, z_obs, av, ssp_data, id_imag):
    """Return the photometric color indices for the given filters transmission
    :param params: Model parameters
    :type params: Dictionnary of parameters
    :param z_ref: redshift of the galaxy used as template
    :type z_ref: float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: Jax array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param z_obs: Redshift of the observations
    :type z_obs: float
    :param av: Attenuation parameter in dust law
    :type av: float
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional

    :return: tuple of arrays the predicted colors for the SED spectrum model represented by its parameters.
    :rtype: tuple(array of floats of length (nb bands), float)
    """
    _mags = templ_mags_legacy(params, z_ref, wls, filt_trans_arr, z_obs, av, ssp_data)
    return _mags[:-2] - _mags[id_imag], _mags[-2] - _mags[-1]


vmap_iclrs_av_legacy = vmap(templ_iclrs_nuvk_legacy, in_axes=(None, None, None, None, None, 0, None, None))
vmap_iclrs_zobs_legacy = vmap(vmap_iclrs_av_legacy, in_axes=(None, None, None, None, 0, None, None, None))
vmap_iclrs_pars_legacy = vmap(vmap_iclrs_zobs_legacy, in_axes=(0, 0, None, None, None, None, None, None))


def make_legacy_templates(params_arr, zref_arr, wls, transm_arr, redz_arr, av_arr, ssp_data):
    """make_legacy_templates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param z_ref: array of redshift of the galaxy used as template
    :type z_ref: JAX-array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param av_arr: Attenuation grid on which to compute the templates photometry
    :type av_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, av_arr, ssp_data)
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-3] - template_mags[:, :, :, 1:-2]
    templ_tupl = [tuple(_pars) + tuple([z]) for _pars, z in zip(params_arr, zref_arr, strict=True)]
    reslist_of_tupl = tree_map(lambda partup: vmap_clrs_zobs_legacy(jnp.array(partup[:-1]), partup[-1], wls, transm_arr, redz_arr, av_arr, ssp_data), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_clrs_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    return reslist_of_tupl


def make_legacy_itemplates(params_arr, zref_arr, wls, transm_arr, redz_arr, av_arr, ssp_data, id_imag=3):
    """make_legacy_itemplates Creates the set of templates for photo-z estimation, using DSPS to syntheticize the photometry from a set of input parameters.

    :param params_arr: Model parameters as output by DSPS
    :type params_arr: Array of float
    :param z_ref: array of redshift of the galaxy used as template
    :type z_ref: JAX-array of float
    :param wls: Wavelengths on which the filters are interpolated
    :type wls: JAX-array of float
    :param filt_trans_arr: Filters transmission
    :type filt_trans_arr: JAX-array of floats of dimension (nb bands+2) * len(wls). The last two bands are for the prior computation.
    :param redz_arr: redshift grid on which to compute the templates photometry
    :type redz_arr: array
    :param av_arr: Attenuation grid on which to compute the templates photometry
    :type av_arr: array
    :param ssp_data: SSP library
    :type ssp_data: namedtuple
    :param id_imag: index of reference band (usually i). For 6-band LSST : u=0 g=1 r=2 i=3 z=4 y=5, defaults to 3
    :type id_imag: int, optional
    :return: Templates for photoZ estimation, accounting for the Star Formation History up to the redshift value, as estimated by DSPS
    :rtype: Tuple of arrays of floats
    """
    # template_mags = vmap_mags_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, anu_arr, ssp_data)
    # i_mag = template_mags[:, :, :, id_imag]
    # nuvk = template_mags[:, :, :, -2] - template_mags[:, :, :, -1]
    # colors = template_mags[:, :, :, :-2] - i_mag
    templ_tupl = [tuple(_pars) + tuple([z]) for _pars, z in zip(params_arr, zref_arr, strict=True)]
    reslist_of_tupl = tree_map(lambda partup: vmap_iclrs_zobs_legacy(jnp.array(partup[:-1]), partup[-1], wls, transm_arr, redz_arr, av_arr, ssp_data, id_imag), templ_tupl, is_leaf=istuple)
    # colors, nuvk = vmap_iclrs_pars_legacy(params_arr, zref_arr, wls, transm_arr, redz_arr, av_arr, ssp_data, id_imag)
    return reslist_of_tupl
