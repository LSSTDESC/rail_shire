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
import pandas as pd
from jax import numpy as jnp

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid


from .dsps_params import SSPParametersFit

_DUMMY_PARS = SSPParametersFit()
PARS_DF = pd.DataFrame(index=_DUMMY_PARS.PARAM_NAMES_FLAT, columns=["INIT", "MIN", "MAX"], data=jnp.column_stack((_DUMMY_PARS.INIT_PARAMS, _DUMMY_PARS.PARAMS_MIN, _DUMMY_PARS.PARAMS_MAX)))
INIT_PARAMS = jnp.array(PARS_DF["INIT"])
PARAMS_MIN = jnp.array(PARS_DF["MIN"])
PARAMS_MAX = jnp.array(PARS_DF["MAX"])

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


def extract_pdz(pdf_arr, zs, z_grid):
    """extract_pdz Computes and returns the marginilized Probability Density function of redshifts and associated statistics for all observations.
    Each item of the `pdf_arr` corresponds to the posteriors for 1 galaxy template, for all input galaxies : `jax.ndarray` of shape `(n_inputs, len(z_grid))`

    :param pdf_arr: Output of photo-z estimation as a JAX array.
    :type pdf_arr: jax.ndarray
    :param zs: Spectro-z values for input galaxies (NaNs if not available)
    :type zs: jax array
    :param z_grid: Grid of redshift values on which the likelihood was computed
    :type z_grid: jax array
    :return: Marginalized Probability Density function of redshift values and associated summarized statistics
    :rtype: dict
    """
    _n2 = trapezoid(jnp.nansum(pdf_arr, axis=0), x=z_grid, axis=0)
    pdf_arr = pdf_arr / _n2
    pdz_arr = jnp.nansum(pdf_arr, axis=0)
    z_means = vmap_mean(z_grid, pdz_arr)
    z_MLs = z_grid[jnp.nanargmax(pdz_arr, axis=0)]
    z_meds = vmap_median(z_grid, pdz_arr)
    pdz_dict = {"z_grid": z_grid, "PDZ": pdz_arr, "redshift": zs, "z_ML": z_MLs, "z_mean": z_means, "z_med": z_meds}
    return pdz_dict


def run_from_inputs(inputs, bounds=None):
    """run_from_inputs Run the photometric redshifts estimation with the given input settings.

    :param inputs: Input settings for the photoZ run. Can be loaded from a `JSON` file using `process_fors2.fetchData.json_to_inputs`.
    :type inputs: dict
    :param bounds: index of first and last elements to load. If None, reads the whole catalog. Defaults to None.
    :type bounds: 2-tuple of int or None
    :return: Photo-z estimation results. These are not written to disk within this function.
    :rtype: list (tree-like)
    """

    from .template import (
        make_legacy_itemplates,
        make_legacy_templates,
        make_sps_itemplates,
        make_sps_templates,
    )
    from .galaxy import likelihood, posterior
    from .io_utils import istuple, load_data_for_run

    z_grid, wl_grid, transm_arr, templ_parsarr, templ_zref_arr, templ_classif, observed_imags, observed_colors, observed_noise, observed_zs, sspdata = load_data_for_run(inputs, bounds=bounds)

    print("Photometric redshift estimation (please be patient, this may take a some time on large datasets) :")

    av_arr = jnp.linspace(PARS_DF.loc["AV", "MIN"], PARS_DF.loc["AV", "MAX"], num=6, endpoint=True)

    if inputs["photoZ"]["i_colors"]:
        if "sps" in inputs["photoZ"]["Mode"].lower():
            templ_tuples = make_sps_itemplates(templ_parsarr, wl_grid, transm_arr, z_grid, av_arr, sspdata, id_imag=inputs["photoZ"]["i_band_num"])
        else:
            templ_tuples = make_legacy_itemplates(templ_parsarr, templ_zref_arr, wl_grid, transm_arr, z_grid, av_arr, sspdata, id_imag=inputs["photoZ"]["i_band_num"])
    else:
        if "sps" in inputs["photoZ"]["Mode"].lower():
            templ_tuples = make_sps_templates(templ_parsarr, wl_grid, transm_arr, z_grid, av_arr, sspdata)
        else:
            templ_tuples = make_legacy_templates(templ_parsarr, templ_zref_arr, wl_grid, transm_arr, z_grid, av_arr, sspdata)

    # try:
    if inputs["photoZ"]["prior"]:
        probz_arr = jax.tree_util.tree_map(
            lambda sed_tupl: posterior(sed_tupl[0], observed_colors, observed_noise, observed_imags, z_grid, sed_tupl[1]),
            templ_tuples,
            is_leaf=istuple,
        )
    else:
        probz_arr = jax.tree_util.tree_map(
            lambda sed_tupl: likelihood(sed_tupl[0], observed_colors, observed_noise),
            templ_tuples,
            is_leaf=istuple,
        )

    probz_arr = jnp.array(probz_arr)
    results_dict = extract_pdz(probz_arr, observed_zs, z_grid)

    print("All done !")

    return results_dict
