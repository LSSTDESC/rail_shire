"""
Age-dependant metallicity helpers from Andrew Hearin ; January 2024
"""
from dsps import calc_lgmet_weights_from_lognormal_mdf, calc_age_weights_from_sfh_table, cumulative_mstar_formed
from dsps.sed.ssp_weights import SSPWeights
from dsps.sed.stellar_sed import RestSED
from dsps.utils import _tw_sigmoid
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

LGAGE_CRIT_YR, LGAGE_CRIT_H = 8.0, 1.0


@jjit
def _age_correlated_met_weights_kern(lg_ages_gyr, lgmet_young, lgmet_old, lgmet_scatter, ssp_lgmet):
    lg_ages_yr = lg_ages_gyr + 9
    lgmet = _tw_sigmoid(lg_ages_yr, LGAGE_CRIT_YR, LGAGE_CRIT_H, lgmet_young, lgmet_old)
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(lgmet, lgmet_scatter, ssp_lgmet)
    return lgmet_weights


# _a = (0, None, None, None, None)
_age_correlated_met_weights_vmap = jjit(vmap(_age_correlated_met_weights_kern, in_axes=(0, None, None, None, None)))


@jjit
def _get_age_correlated_met_weights(lg_ages_gyr, lgmet_young, lgmet_old, lgmet_scatter, ssp_lgmet):
    lgmet_weights = _age_correlated_met_weights_vmap(lg_ages_gyr, lgmet_young, lgmet_old, lgmet_scatter, ssp_lgmet)
    return lgmet_weights.T


@jjit
def calc_ssp_weights_sfh_table_lognormal_mdf_agedep(
    gal_t_table,
    gal_sfh_table,
    lgmet_young,
    lgmet_old,
    lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
    t_obs,
):
    """Calculate SSP weights of a tabulated SFH and a lognormal MDF

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr when the galaxy SFH is tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Tabulation of the galaxy SFH in Msun/yr at the times gal_t_table

    gal_lgmet : float
        log10(Z), center of the lognormal metallicity distribution function

    gal_lgmet_scatter : float
        lognormal scatter about gal_lgmet

    ssp_lgmet : ndarray of shape (n_ages, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    SSPWeights : namedtuple with the following entries:

        weights : ndarray of shape (n_met, n_ages)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    age_weights = calc_age_weights_from_sfh_table(gal_t_table, gal_sfh_table, ssp_lg_age_gyr, t_obs)

    lgmet_weights = _get_age_correlated_met_weights(ssp_lg_age_gyr, lgmet_young, lgmet_old, lgmet_scatter, ssp_lgmet)
    weights = lgmet_weights * age_weights.reshape((1, -1))

    return SSPWeights(weights, lgmet_weights, age_weights)


@jjit
def calc_rest_sed_sfh_table_lognormal_mdf_agedep(
    gal_t_table,
    gal_sfr_table,
    gal_lgmet_young,
    gal_lgmet_old,
    gal_lgmet_scatter,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_flux,
    t_obs,
):
    """
    Calculate the SED of a galaxy defined by input tables of SFH and
    a lognormal metallicity distribution function

    Parameters
    ----------
    gal_t_table : ndarray of shape (n_t, )
        Age of the universe in Gyr at which the input galaxy SFH has been tabulated

    gal_sfr_table : ndarray of shape (n_t, )
        Star formation history in Msun/yr evaluated at the input gal_t_table

    gal_lgmet_young : float
        log10(Z) of young stars in the galaxy

    gal_lgmet_old : float
        log10(Z) of old stars in the galaxy

    gal_lgmet_scatter : float
        Lognormal scatter in metallicity

    ssp_lgmet : ndarray of shape (n_met, )
        Array of log10(Z) of the SSP templates

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        Array of log10(age/Gyr) of the SSP templates

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        SED of the SSP in units of Lsun/Hz/Msun

    t_obs : float
        Age of the universe in Gyr at the time the galaxy is observed

    Returns
    -------
    RestSED : namedtuple with the following entries:

        rest_sed : ndarray of shape (n_wave, )
            Restframe SED of the galaxy in units of Lsun/Hz

        weights : ndarray of shape (n_met, n_ages, 1)
            SSP weights of the joint distribution of stellar age and metallicity

        lgmet_weights : ndarray of shape (n_met, )
            SSP weights of the distribution of stellar metallicity

        age_weights : ndarray of shape (n_ages, )
            SSP weights of the distribution of stellar age

    """
    ssp_weights = calc_ssp_weights_sfh_table_lognormal_mdf_agedep(
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_young,
        gal_lgmet_old,
        gal_lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    weights, lgmet_weights, age_weights = ssp_weights
    n_met, n_ages = weights.shape
    sed_unit_mstar = jnp.sum(ssp_flux * weights.reshape((n_met, n_ages, 1)), axis=(0, 1))

    gal_mstar_table = cumulative_mstar_formed(gal_t_table, gal_sfr_table)
    gal_logsm_table = jnp.log10(gal_mstar_table)
    logsm_obs = jnp.interp(jnp.log10(t_obs), jnp.log10(gal_t_table), gal_logsm_table)
    mstar_obs = 10**logsm_obs
    rest_sed = sed_unit_mstar * mstar_obs
    return RestSED(rest_sed, weights, lgmet_weights, age_weights)
