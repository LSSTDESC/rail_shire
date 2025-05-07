#!/usr/bin/env python3
"""
Module to load data for combined SPS and PhotoZ studies within RAIL, applied to LSST.

Created on Wed Oct 24 14:52 2024

@author: joseph
"""

import os
import json
import pandas as pd
from rail.dsps import load_ssp_templates
from jax import numpy as jnp
from .analysis import _DUMMY_PARS

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import h5py

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

_script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    SHIREDATALOC = os.environ["SHIREDATALOC"]
except KeyError:
    SHIREDATALOC = os.path.abspath(os.path.join(_script_dir, "../../../examples", "data"))
    os.environ["SHIREDATALOC"] = SHIREDATALOC
print(f"Default location for rail_shire data set to {SHIREDATALOC}.")

DEFAULTS_DICT = {}
FILENAME_SSP_DATA = "ssp_data_fsps_v3.2_lgmet_age.h5"
# FILENAME_SSP_DATA = "fspsData_v3_2_BASEL.h5"
# FILENAME_SSP_DATA = 'fspsData_v3_2_C3K.h5'
FULLFILENAME_SSP_DATA = os.path.abspath(os.path.join(SHIREDATALOC, "SSP", FILENAME_SSP_DATA))
DEFAULTS_DICT.update({"DSPS HDF5": FULLFILENAME_SSP_DATA})

def json_to_inputs(conf_json):
    """
    Load JSON configuration file and return inputs dictionary.

    Parameters
    ----------
    conf_json : path or str
        Path to the configuration file in JSON format.

    Returns
    -------
    dict
        Dictionary of inputs `{param_name: value}`.
    """
    conf_json = os.path.abspath(conf_json)
    with open(conf_json, "r") as inpfile:
        inputs = json.load(inpfile)
    return inputs


def load_ssp(ssp_file=None):
    """load_ssp _summary_

    :param ssp_file: _description_, defaults to None
    :type ssp_file: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if ssp_file == "" or ssp_file is None or "default" in ssp_file.lower():
        fullfilename_ssp_data = DEFAULTS_DICT["DSPS HDF5"]
    else:
        fullfilename_ssp_data = os.path.abspath(ssp_file)
    ssp_data = load_ssp_templates(fn=fullfilename_ssp_data)
    return ssp_data


def has_redshift(dic):
    """
    Utility to detect a leaf in a dictionary (tree) based on the assumption that a leaf is a dictionary that contains individual information linked to a spectrum, such as the redshift of the galaxy.

    Parameters
    ----------
    dic : dictionary
        Dictionary with data. Within the context of this function's use, this is an output of the catering of data to fit on DSPS.
        This function is applied to a global dictionary (tree) and its sub-dictionaries (leaves - as identified by this function).

    Returns
    -------
    bool
        `True` if `'redshift'` is in `dic.keys()` - making it a leaf - `False` otherwise.
    """
    return "redshift" in list(dic.keys())

def istuple(tree):
    """istuple Detects a leaf in a tree based on whether it is a tuple.

    :param tree: tree to be tested
    :type tree: pytree
    :return: whether the tree is a tuple or not
    :rtype: boolean
    """
    return isinstance(tree, tuple)

def readCatalogHDF5(h5file, group="photometry", filt_names=None, bounds=None):
    """readCatalogHDF5 Reads the magnitudes and spectroscopic redshift (if available) from a dictionary-like catalog provided as an `HDF5` file.
    Preliminary step for the `process_fors2.photoZ` calculations.

    :param h5file: Path to the HDF5 catalog file.
    :type h5file: str or path-like
    :param group: Identifier of the group to read within the `HDF5` file. This argument is passed to the `key` argument of `pandas.DataFrame.read_hdf`. Defaults to 'photometry'.
    :type group: str, optional
    :param filt_names: Names of filters to look for in the catalogs. Data recorded as `mag_[filter name]` and `mag_err_[filter name]` will be returned.
    If None, defaults to LSST filters. Defaults to None.
    :type filt_names: list of str, optional
    :param bounds: index of first and last elements to load. If None, reads the whole catalog. Defaults to None.
    :type bounds: 2-tuple of int or None
    :return: tuple containing AB magnitudes, corresponding errors and spectroscopic redshift as arrays.
    :rtype: tuple of arrays
    """
    if filt_names is None:
        filt_names = ["u_lsst", "g_lsst", "r_lsst", "i_lsst", "z_lsst", "y_lsst"]
    df_cat = pd.read_hdf(os.path.abspath(h5file), key=group)
    if bounds is not None:
        df_cat = df_cat[bounds[0] : bounds[-1]].copy()
    magnames = [f"mag_{filt}" for filt in filt_names]
    magerrs = [f"mag_err_{filt}" for filt in filt_names]
    obs_mags = jnp.array(df_cat[magnames])
    obs_mags_errs = jnp.array(df_cat[magerrs])
    try:
        z_specs = jnp.array(df_cat["redshift"])
    except IndexError:
        z_specs = jnp.full(obs_mags.shape[0], jnp.nan)
    return obs_mags, obs_mags_errs, z_specs


def read_h5_table(templ_h5_file, group="fit_dsps", classif="Classification"):
    """read_h5_table _summary_

    :param templ_h5_file: _description_
    :type templ_h5_file: _type_
    :param group: _description_, defaults to 'fit_dsps'
    :type group: str, optional
    :param classif: Name of the field holding classif. info. 'Classification', 'CAT_NII', 'CAT_SII', 'CAT_OI' or 'CAT_OIII/OIIvsOI', defaults to 'Classification'
    :type classif: str, optional
    :return: _description_
    :rtype: _type_
    """
    templ_df = pd.read_hdf(os.path.abspath(templ_h5_file), key=group)
    templ_pars_arr = jnp.array(templ_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
    zref_arr = jnp.array(templ_df["redshift"])
    classif_series = templ_df[classif]  # Default value is 'Classification' but other criteria can be used : 'CAT_NII', 'CAT_SII', etc.
    return templ_pars_arr, zref_arr, classif_series  # placeholder, finish the function later to return the proper array of parameters


def plot_zp_zs_ensemble(ens_PDFs, z_true, z_grid=None, key_estim="zmode", label='', bins=100):
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    z_grid = jnp.squeeze(jnp.array(ens_PDFs[0].dist.xvals, dtype=jnp.float64)) if z_grid is None else z_grid
    zp = jnp.squeeze(ens_PDFs.mode(z_grid)) if key_estim is None else ens_PDFs.ancil[key_estim]
    zs = jnp.squeeze(z_true) #ens_PDFs.ancil[key_truth]
    bias = zp - zs
    errz = bias/(1+zs)
    _, sigscat, medscat = jnp.mean(errz), jnp.std(errz), jnp.median(errz)
    mad = jnp.median(jnp.abs(errz)) # - medscat))
    sig_mad = 1.4826 * mad
    outliers = jnp.nonzero(jnp.abs(errz)*100.0 > 15) #3*sigscat) #
    outl_rate = zs[outliers].shape[0] / zs.shape[0]

    density = ax.hexbin(zs, zp, bins='log', gridsize=bins)
    ax.plot(z_grid, z_grid, c="k", ls=":", lw=1)
    outl, = ax.plot(z_grid, z_grid + 0.15 * (1 + z_grid), c="k", lw=2)
    ax.plot(z_grid, z_grid - 0.15 * (1 + z_grid), c="k", lw=2)

    med, = ax.plot(z_grid, z_grid + medscat*(1 + z_grid), c="orange", lw=2, ls='-.') #, label=r"$\mathrm{median}\left(\zeta_z \right)$")
    scat = ax.fill_between(z_grid, z_grid + (medscat+sigscat)*(1 + z_grid), z_grid + (medscat-sigscat)*(1 + z_grid), color="pink", alpha=0.4)
    
    ax.set_xlabel(r"$z_{spec}$")
    ax.set_ylabel(r"$z_{phot}$")
    ax.set_xlim(z_grid.min()-0.05, z_grid.max()+0.05)
    ax.set_ylim(z_grid.min()-0.05, z_grid.max()+0.05)
    f.legend(
        [density, outl, (med, scat)],
        [
            label,
            "Outliers:\n"+r"$\left| \frac{z_p-z_s}{1+z_s} \right| > 0.15$",
            r"$\mathrm{median} \left( \zeta_z \right) \pm \sigma_{\zeta_z}=$"+f"\n\t{medscat:.3f}"+r"$\pm$"+f"{sigscat:.3f}"
        ],
        loc='lower right',
        bbox_to_anchor=(1., 0.)
    )
    ax.grid()
    ax.set_title(f"{100.0*outl_rate:.3f}% outliers ;\n"+r"$\sigma_{MAD}=$"+f"{sig_mad:.3f}")

    #plt.colorbar(scalarMap, ax=ax, location='right', label="Scatter (%)")
    _ = f.colorbar(density, label='Density', location='right')
    ax.set_aspect("equal", adjustable="box", share=False)
    
    print(f"{label}: {100*outl_rate:.3f}% outliers out of {len(zp)} successful fits.\nsigma_mad: {sig_mad:.3f}.")
    return ax

def plot_zp_zs_hdf5BPZ(pdfs_hdf5, zs, key_zgrid='xvals', key_estim='zmode', label='', bins=100):
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    with h5py.File(pdfs_hdf5, 'r') as h5pdf:
        ancil = h5pdf.get('ancil')
        meta = h5pdf.get('meta')
        z_grid = jnp.squeeze(jnp.array(meta.get(key_zgrid), dtype=jnp.float64))
        zp = jnp.array(ancil.get(key_estim), dtype=jnp.float64)
    
    bias = zp - zs
    errz = bias/(1+zs)
    _, sigscat, medscat = jnp.mean(errz), jnp.std(errz), jnp.median(errz)
    mad = jnp.median(jnp.abs(errz)) # - medscat))
    sig_mad = 1.4826 * mad
    outliers = jnp.nonzero(jnp.abs(errz)*100.0 > 15) #3*sigscat) #
    outl_rate = len(zs[outliers]) / len(zs)

    density = ax.hexbin(zs, zp, bins='log', gridsize=bins)
    ax.plot(z_grid, z_grid, c="k", ls=":", lw=1)
    outl, = ax.plot(z_grid, z_grid + 0.15 * (1 + z_grid), c="k", lw=2)
    ax.plot(z_grid, z_grid - 0.15 * (1 + z_grid), c="k", lw=2)

    med, = ax.plot(z_grid, z_grid + medscat*(1 + z_grid), c="orange", lw=2, ls='-.')
    scat = ax.fill_between(z_grid, z_grid + (medscat+sigscat)*(1 + z_grid), z_grid + (medscat-sigscat)*(1 + z_grid), color="pink", alpha=0.4)
    
    ax.set_xlabel(r"$z_{spec}$")
    ax.set_ylabel(r"$z_{phot}$")
    ax.set_xlim(z_grid.min()-0.05, z_grid.max()+0.05)
    ax.set_ylim(z_grid.min()-0.05, z_grid.max()+0.05)
    f.legend(
        [density, outl, (med, scat)],
        [
            label,
            "Outliers:\n"+r"$\left| \frac{z_p-z_s}{1+z_s} \right| > 0.15$",
            r"$\mathrm{median} \left( \zeta_z \right) \pm \sigma_{\zeta_z}=$"+f"\n\t{medscat:.3f}"+r"$\pm$"+f"{sigscat:.3f}"
        ],
        loc='lower right',
        bbox_to_anchor=(1., 0.)
    )
    ax.grid()
    ax.set_title(f"{100.0*outl_rate:.3f}% outliers ;\n"+r"$\sigma_{MAD}=$"+f"{sig_mad:.3f}")

    #plt.colorbar(scalarMap, ax=ax, location='right', label="Scatter (%)")
    _ = f.colorbar(density, label='Density', location='right')
    ax.set_aspect("equal", adjustable="box", share=False)
    
    print(f"{label}: {100*outl_rate:.3f}% outliers out of {len(zp)} successful fits.\nsigma_mad: {sig_mad:.3f}. Median scatter {medscat*100:.3f}%.")
    return ax


def hist_outliers(qp_ens_1, zs, z_grid=None, key_estim='zmode', label1='', qp_ens_2=None, label2='', qp_ens_3=None, label3=''):
    f, ax = plt.subplots(1, 1, figsize=(7, 6))

    z_grid = jnp.squeeze(jnp.array(qp_ens_1[0].dist.xvals, dtype=jnp.float64)) if z_grid is None else z_grid
    zp1 = jnp.squeeze(qp_ens_1.mode(z_grid)) if key_estim is None else qp_ens_1.ancil[key_estim]

    bias1 = zp1 - zs
    errz1 = bias1/(1+zs)
    outliers1 = jnp.nonzero(jnp.abs(errz1)*100.0 > 15) #3*sigscat) #
    _n, _bins, _ = ax.hist(zs[outliers1], bins='auto', density=False, label=label1, alpha=0.7)

    if qp_ens_2 is not None:
        zp2 = jnp.squeeze(qp_ens_2.mode(z_grid)) if key_estim is None else qp_ens_2.ancil[key_estim]
        bias2 = zp2 - zs
        errz2 = bias2/(1+zs)
        outliers2 = jnp.nonzero(jnp.abs(errz2)*100.0 > 15) #3*sigscat) #
        ax.hist(zs[outliers2], bins=_bins, density=False, label=label2, alpha=0.7)

    if qp_ens_3 is not None:
        zp3 = jnp.squeeze(qp_ens_3.mode(z_grid)) if key_estim is None else qp_ens_3.ancil[key_estim]
        bias3 = zp3 - zs
        errz3 = bias3/(1+zs)
        outliers3 = jnp.nonzero(jnp.abs(errz3)*100.0 > 15) #3*sigscat) #
        ax.hist(zs[outliers3], bins=_bins, density=False, label=label3, alpha=0.7)
    
    #ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$z_{spec}$")
    ax.set_ylabel("Outliers count")
    ax.legend()
    return ax
