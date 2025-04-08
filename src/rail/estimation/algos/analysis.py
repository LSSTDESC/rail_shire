#!/usr/bin/env python3
#
#  Analysis.py
#
#  Copyright 2023  <joseph@wl-chevalier>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import jax
from jax import numpy as jnp

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid


"""
Reminder :
Cosmo = namedtuple('Cosmo', ['h0', 'om0', 'l0', 'omt'])
sedpyFilter = namedtuple('sedpyFilter', ['name', 'wavelengths', 'transmission'])
BaseTemplate = namedtuple('BaseTemplate', ['name', 'flux', 'z_sps'])
SPS_Templates = namedtuple('SPS_Templates', ['name', 'redshift', 'z_grid', 'i_mag', 'colors', 'nuvk'])
Observation = namedtuple('Observation', ['num', 'AB_fluxes', 'AB_f_errors', 'z_spec'])
DustLaw = namedtuple('DustLaw', ['name', 'EBV', 'transmission'])
"""

# conf_json = 'EmuLP/COSMOS2020-with-FORS2-HSC_only-jax-CC-togglePriorTrue-opa.json' # attention à la localisation du fichier !


@jax.jit
def _cdf(z, pdz):
    cdf = jnp.array([trapezoid(pdz[:i], x=z[:i]) for i in range(len(z))])
    return cdf


@jax.jit
def _median(z, pdz):
    cdz = _cdf(z, pdz)
    medz = z[jnp.nonzero(cdz >= 0.5, size=1)][0]
    return medz


vmap_median = jax.vmap(_median, in_axes=(None, 1))


@jax.jit
def _mean(z, pdz):
    return trapezoid(z * pdz, x=z)


vmap_mean = jax.vmap(_mean, in_axes=(None, 1))


def extract_pdz(pdf_dict, zs, z_grid):
    """extract_pdz Computes and returns the marginilized Probability Density function of redshifts and associated statistics for all observations.
    Each item of the `pdf_dict` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_dict: Output of photo-z estimation as a dictonary of JAX arrays.
    :type pdf_dict: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and associated summarized statistics
    :rtype: dict
    """
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    _n2 = trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    pdz_arr = jnp.nansum(pdf_arr, axis=0)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "z_spec": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def extract_pdz_fromchi2(chi2_dict, zs, z_grid):
    r"""extract_pdz_fromchi2 Similar to extract_pdz except takes $\chi^2$ values as inputs (*i.e.* negative log-likelihood).
    Computes and returns the marginilized Probability Density function of redshifts

    :param chi2_dict: Output of photo-z estimation as a dictonary of JAX arrays.
    :type chi2_dict: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and elementary associated stats
    :rtype: dict
    """
    chi2_arr = jnp.array([chi2_templ for _, chi2_templ in chi2_dict.items()])
    _n1 = 100.0 / jnp.nanmax(chi2_arr)
    chi2_arr = chi2_arr * _n1
    exp_arr = jnp.power(jnp.exp(-0.5 * chi2_arr), 1 / _n1)
    _n2 = trapezoid(jnp.nansum(exp_arr, axis=0), x=z_grid)
    exp_arr = exp_arr / _n2
    pdz_arr = jnp.nansum(exp_arr, axis=0)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "z_spec": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def extract_pdz_allseds(pdf_dict, zs, z_grid):
    """extract_pdz_allseds Computes and returns the marginilized Probability Density function of redshifts for a single observation ;
    The conditional probability density is also computed for each galaxy template.
    Each item of the `pdf_dict` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_dict: Output of photo-z estimation as a dictonary of JAX arrays.
    :type pdf_dict: dict of jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and conditional PDF for each template.
    :rtype: dict
    """
    pdf_arr = jnp.array([pdf_templ for _, pdf_templ in pdf_dict.items()])
    _n2 = trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    pdz_arr = jnp.nansum(pdf_arr, axis=0)
    templ_wgts = trapezoid(pdf_arr, x=z_grid, axis=1)
    sed_evid_z = jnp.nansum(pdf_arr, axis=2)
    sed_evid_marg = jnp.nansum(templ_wgts, axis=1)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {
        "z_grid": z_grid,
        "PDZ": pdz_arr,
        "p(z, sed)": pdf_arr,
        "z_spec": zs,
        "z_ML": z_MLs,
        "z_mean": z_means,
        "z_med": z_meds,
        "SED weights / galaxy": templ_wgts,
        "SED evidence along z": sed_evid_z,
        "Marginalised SED evidence": sed_evid_marg,
    }
    return pdz_dict


def run_from_inputs(inputs):
    """run_from_inputs Run the photometric redshifts estimation with the given input settings.

    :param inputs: Input settings for the photoZ run. Can be loaded from a `JSON` file using `process_fors2.fetchData.json_to_inputs`.
    :type inputs: dict
    :return: Photo-z estimation results. These are not written to disk within this function.
    :rtype: list (tree-like)
    """
    from rail.dsps_fors2_pz import SPS_Templates, likelihood, likelihood_fluxRatio, load_data_for_run, posterior, posterior_fluxRatio

    z_grid, templates_dict, observed_imags, observed_colors, observed_noise, observed_zs = load_data_for_run(inputs)

    print("Photometric redshift estimation (please be patient, this may take a couple of hours on large datasets) :")

    def has_sps_template(cont):
        return isinstance(cont, SPS_Templates)

    def estim_zp(observs_cols, observs_errs, observs_i):
        if inputs["photoZ"]["prior"]:  # and observ.valid_filters[inputs["photoZ"]["i_band_num"]]:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: posterior(sps_templ.colors, observs_cols, observs_errs, observs_i, sps_templ.z_grid, sps_templ.nuvk), templates_dict, is_leaf=has_sps_template)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(
                    lambda sps_templ: posterior_fluxRatio(sps_templ.colors, observs_cols, observs_errs, observs_i, sps_templ.z_grid, sps_templ.nuvk), templates_dict, is_leaf=has_sps_template
                )
            )
        else:
            probz_dict = (
                jax.tree_util.tree_map(lambda sps_templ: likelihood(sps_templ.colors, observs_cols, observs_errs), templates_dict, is_leaf=has_sps_template)
                if inputs["photoZ"]["use_colors"]
                else jax.tree_util.tree_map(lambda sps_templ: likelihood_fluxRatio(sps_templ.colors, observs_cols, observs_errs), templates_dict, is_leaf=has_sps_template)
            )
        return probz_dict

    results_dict = extract_pdz(
        estim_zp(observed_colors, observed_noise, observed_imags),
        observed_zs,
        z_grid,
    )
    print("All done !")

    return results_dict
