#!/usr/bin/env python3
"""
Module to load data for combined SPS and PhotoZ studies within RAIL, applied to LSST.

Created on Wed Oct 24 14:52 2024

@author: joseph
"""

import os
import h5py
import json
import jax
import numpy as np
import pandas as pd
from tqdm import tqdm
from rail.dsps import load_ssp_templates
from jax import numpy as jnp

_script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    PZDATALOC = os.environ["PZDATALOC"]
except KeyError:
    try:
        PZDATALOC = input("Please type in the path to FORS2 data, e.g. /home/usr/rail_dspsXfors2_pz/src/data")
        os.environ["PZDATALOC"] = PZDATALOC
    except Exception:
        PZDATALOC = os.path.join(_script_dir, "data")
        os.environ["PZDATALOC"] = PZDATALOC

DEFAULTS_DICT = {}
FILENAME_SSP_DATA = "ssp_data_fsps_v3.2_lgmet_age.h5"
# FILENAME_SSP_DATA = "test_fspsData_v3_2_BASEL.h5"
# FILENAME_SSP_DATA = 'test_fspsData_v3_2_C3K.h5'
FULLFILENAME_SSP_DATA = os.path.abspath(os.path.join(PZDATALOC, "ssp", FILENAME_SSP_DATA))
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


def templatesToHDF5(outfilename, templ_dict):
    """
    Writes the SED templates used for photo-z in an HDF5 for a quicker use in future runs.
    Mimics the structure of the class SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ.

    Parameters
    ----------
    outfilename : str or path
        Name of the `HDF5` file that will be written.
    templ_dict : dict
        Dictionary object containing the SED templates.

    Returns
    -------
    path
        Absolute path to the written file - if successful.
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        for key, templ in templ_dict.items():
            groupout = h5out.create_group(key)
            groupout.attrs["name"] = templ.name
            groupout.attrs["redshift"] = templ.redshift
            groupout.create_dataset("z_grid", data=templ.z_grid, compression="gzip", compression_opts=9)
            groupout.create_dataset("i_mag", data=templ.i_mag, compression="gzip", compression_opts=9)
            groupout.create_dataset("colors", data=templ.colors, compression="gzip", compression_opts=9)
            groupout.create_dataset("nuvk", data=templ.nuvk, compression="gzip", compression_opts=9)

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readTemplatesHDF5(h5file):
    """readTemplatesHDF5 loads the SED templates for photo-z from the specified HDF5 and returns them as a dictionary of objects
    SPS_Templates = namedtuple("SPS_Templates", ["name", "redshift", "z_grid", "i_mag", "colors", "nuvk"]) from process_fors2.photoZ

    :param h5file: Path to the HDF5 containing the SED templates data.
    :type h5file: str or path-like object
    :return: The dictionary of SPS_Templates objects.
    :rtype: dictionary
    """
    from .template import SPS_Templates

    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key in h5in:
            grp = h5in.get(key)
            out_dict.update(
                {
                    key: SPS_Templates(
                        grp.attrs.get("name"), grp.attrs.get("redshift"), jnp.array(grp.get("z_grid")), jnp.array(grp.get("i_mag")), jnp.array(grp.get("colors")), jnp.array(grp.get("nuvk"))
                    )
                }
            )
    return out_dict


def photoZtoHDF5(outfilename, pz_out_dict):
    """photoZtoHDF5 Saves the dictionary of `process_fors2.photoZ` outputs to an `HDF5` file.

    :param outfilename: Name of the `HDF5` file that will be written.
    :type outfilename: str or path-like object
    :param pz_out_dict: Dictionary of photo-z results : each item corresponds to the array of values of one result (identified by the key) for all input galaxies.
    :type pz_out_dict: dict
    :return: Absolute path to the written file - if successful.
    :rtype: str or path-like object
    """
    fileout = os.path.abspath(outfilename)

    with h5py.File(fileout, "w") as h5out:
        groupout = h5out.create_group("pz_outputs")
        for key, jarray in pz_out_dict.items():
            groupout.create_dataset(key, data=jarray, compression="gzip", compression_opts=9)

    ret = fileout if os.path.isfile(fileout) else f"Unable to write data to {outfilename}"
    return ret


def readPhotoZHDF5(h5file):
    """readPhotoZHDF5 Reads the photo-z results file and generates the corresponding pytree (dictionary) for analysis.

    :param h5file: Path to the HDF5 containing the photo-z results.
    :type h5file: str or path-like object
    :return: Dictionary of photo-z results as computed by `process_fors2.photoZ`.
    :rtype: dict
    """
    filein = os.path.abspath(h5file)
    with h5py.File(filein, "r") as h5in:
        pzout_dict = {key: jnp.array(jarray) for key, jarray in h5in.get("pz_outputs").items()}
    return pzout_dict


def readDSPSHDF5(h5file):
    """readDSPSHDF5 Reads the contents of the HDF5 that stores the results of DSPS fitting procedure.
    Useful to generate templates for photo-z estimation in `process_fors2.photoZ`.

    :param h5file: Path to the HDF5 file containing the DSPS fitting results.
    :type h5file: str or path-like
    :return: Dictionary of DSPS parameters written as attributes in the HDF5 file
    :rtype: dict
    """
    filein = os.path.abspath(h5file)
    out_dict = {}
    with h5py.File(filein, "r") as h5in:
        for key, grp in h5in.items():
            out_dict.update({key: dict(grp.attrs.items())})
    return out_dict


def _recursive_dict_to_hdf5(group, attrs):
    for key, item in attrs.items():
        if isinstance(item, dict):
            sub_group = group.create_group(key, track_order=True)
            _recursive_dict_to_hdf5(sub_group, item)
        else:
            group.attrs[key] = item

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

def load_data_for_run(inp_glob):
    """load_data_for_run Generates input data from the inputs configuration dictionary

    :param inp_glob: input configuration and settings
    :type inp_glob: dict
    :return: data for photo-z evaluation : redshift grid, templates dictionary and the arrays of processed observed data (input catalog) (i mags ; colors ; errors on colors ; spectro-z).
    :rtype: 6-tuple of jax.ndarray, dictionary, jax.ndarray, jax.ndarray, jax.ndarray, jax.ndarray
    """
    from rail.dsps_fors2_pz import NIR_filt, NUV_filt, get_2lists, load_filt, make_legacy_templates, make_sps_templates, sedpyFilter

    _ssp_file = (
        None
        if (inp_glob["photoZ"]["ssp_file"].lower() == "default" or inp_glob["photoZ"]["ssp_file"] == "" or inp_glob["photoZ"]["ssp_file"] is None)
        else os.path.abspath(inp_glob["photoZ"]["ssp_file"])
    )
    ssp_data = load_ssp(_ssp_file)

    inputs = inp_glob["photoZ"]
    z_grid = jnp.arange(inputs["Z_GRID"]["z_min"], inputs["Z_GRID"]["z_max"] + inputs["Z_GRID"]["z_step"], inputs["Z_GRID"]["z_step"])

    filters_dict = inputs["Filters"]
    for _f in filters_dict:
        filters_dict[_f]["path"] = os.path.abspath(os.path.join(PZDATALOC, filters_dict[_f]["path"]))
    print("Loading filters :")
    filters_arr = tuple(sedpyFilter(*load_filt(int(ident), filters_dict[ident]["path"], filters_dict[ident]["transmission"])) for ident in tqdm(filters_dict)) + (NUV_filt, NIR_filt)

    filters_names = [_f["name"] for _, _f in filters_dict.items()]
    # print(f"DEBUG: filters = {filters_arr}")

    print("Building templates :")
    Xfilt = get_2lists(filters_arr)
    # sps_temp_pkl = os.path.abspath(inputs["Templates"])
    # sps_par_dict = read_params(sps_temp_pkl)
    if inputs["Templates"]["overwrite"] or not os.path.isfile(os.path.abspath(inputs["Templates"]["output"])):
        sps_temp_h5 = os.path.abspath(inputs["Templates"]["input"])
        sps_par_dict = readDSPSHDF5(sps_temp_h5)
        if "sps" in inputs["Mode"].lower():
            templ_dict = jax.tree_util.tree_map(lambda dico: make_sps_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        else:
            templ_dict = jax.tree_util.tree_map(lambda dico: make_legacy_templates(dico, Xfilt, z_grid, ssp_data, id_imag=inputs["i_band_num"]), sps_par_dict, is_leaf=has_redshift)
        _ = templatesToHDF5(inputs["Templates"]["output"], templ_dict)
    else:
        templ_dict = readTemplatesHDF5(inputs["Templates"]["output"])

    print("Loading observations :")
    data_path = os.path.abspath(os.path.join(PZDATALOC, inputs["Dataset"]["path"]))
    data_ismag = inputs["Dataset"]["type"].lower() == "m"

    if inputs["Dataset"]["is_ascii"]:
        h5catpath = catalog_ASCIItoHDF5(data_path, data_ismag, filt_names=filters_names)
    else:
        h5catpath = data_path

    if inputs["Dataset"]["overwrite"] or not os.path.isfile(f"pz_inputs_{os.path.basename(h5catpath)}"):
        ab_mags, ab_mags_errs, z_specs = readCatalogHDF5(h5catpath, filt_names=filters_names)

        from .galaxy import vmap_mags_to_i_and_colors
        i_mag_ab, ab_colors, ab_cols_errs = vmap_mags_to_i_and_colors(ab_mags, ab_mags_errs, inputs["i_band_num"])

        clrh5file = f"pz_inputs_{os.path.basename(h5catpath)}"
        _colrs_h5out = pzInputsToHDF5(clrh5file, ab_colors, ab_cols_errs, z_specs, i_mag_ab, filt_names=filters_names)
    else:
        i_mag_ab, ab_colors, ab_cols_errs, z_specs = readPZinputsHDF5(f"pz_inputs_{os.path.basename(h5catpath)}", filt_names=filters_names)

    return z_grid, templ_dict, i_mag_ab, ab_colors, ab_cols_errs, z_specs


def readCatalogHDF5(h5file, group="catalog", filt_names=None):
    """readCatalogHDF5 Reads the magnitudes and spectroscopic redshift (if available) from a dictionary-like catalog provided as an `HDF5` file.
    Preliminary step for the `process_fors2.photoZ` calculations.

    :param h5file: Path to the HDF5 catalog file.
    :type h5file: str or path-like
    :param group: Identifier of the group to read within the `HDF5` file. This argument is passed to the `key` argument of `pandas.DataFrame.read_hdf`. Defaults to 'catalog'.
    :type group: str, optional
    :param filt_names: Names of filters to look for in the catalogs. Data recorded as `mag_[filter name]` and `mag_err_[filter name]` will be returned.
    If None, defaults to LSST filters. Defaults to None.
    :type filt_names: list of str, optional
    :return: tuple containing AB magnitudes, corresponding errors and spectroscopic redshift as arrays.
    :rtype: tuple of arrays
    """
    if filt_names is None:
        filt_names = ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]
    df_cat = pd.read_hdf(os.path.abspath(h5file), key=group)
    magnames = [f"mag_{filt}" for filt in filt_names]
    magerrs = [f"mag_err_{filt}" for filt in filt_names]
    obs_mags = jnp.array(df_cat[magnames])
    obs_mags_errs = jnp.array(df_cat[magerrs])
    try:
        z_specs = jnp.array(df_cat["z_spec"])
    except IndexError:
        z_specs = jnp.full(obs_mags.shape[0], jnp.nan)
    return obs_mags, obs_mags_errs, z_specs


def catalog_ASCIItoHDF5(ascii_file, data_ismag, group="catalog", filt_names=None):
    """catalog_ASCIItoHDF5 Reads a catalog provided as an ASCII file (as in LEPHARE) containing either fluxes or magnitudes and saves it in an `HDF5` file containing AB-magnitudes.

    :param ascii_file: Path to the ASCII file containing catalog cata as an array that can be read with `numpy.loadtxt(ascii_file)`.
    :type ascii_file: str or path-like
    :param data_ismag: Whether the photometry in the file is given as AB-magnitudes or flux density (in erg/s/cm²/Hz). True for AB-magnitudes.
    :type data_ismag: bool
    :param group: Name of the group to write in the `HDF5` file. This argument is passed to the `key` argument of `pandas.DataFrame.to_hdf`. Defaults to 'catalog'.
    :type group: str, optional
    :param filt_names: Names of filters to use as column names in the catalog. Data will be recored as `mag_[filter name]` and `mag_err_[filter name]`.
    If None, defaults to LSST filters. Defaults to None.
    :type filt_names: list of str, optional
    :return: Absolute path to the written file - if successful.
    :rtype: str or path-like object
    """
    if filt_names is None:
        filt_names = ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]
    magnames = [f"mag_{filt}" for filt in filt_names]
    magerrs = [f"mag_err_{filt}" for filt in filt_names]
    N_FILT = len(filt_names)
    data_file_arr = np.loadtxt(os.path.abspath(ascii_file))
    has_zspec = data_file_arr.shape[1] == 1 + 2 * N_FILT + 1
    no_zspec = data_file_arr.shape[1] == 1 + 2 * N_FILT
    assert has_zspec or no_zspec, "Number of column in data does not match one of 1 + 2*n_filts + 1 (id, photometry, z_spec) or 1 + 2*n_filts (id, photometry).\
        \nReview data or filters list."

    from .galaxy import vmap_load_magnitudes

    all_mags, all_mags_err = vmap_load_magnitudes(data_file_arr[:, 1 : 2 * N_FILT + 1], data_ismag)

    all_zs = data_file_arr[:, -1] if has_zspec else jnp.full(all_mags.shape[0], jnp.nan)

    df_mags = pd.DataFrame(columns=magnames + magerrs + ["z_spec"], data=jnp.column_stack((all_mags, all_mags_err, all_zs)))

    hdf_name = f"{os.path.splitext(os.path.basename(ascii_file))[0]}.h5"
    outfilename = os.path.abspath(hdf_name)
    df_mags.to_hdf(outfilename, key=group)
    respath = outfilename if os.path.isfile(outfilename) else f"Unable to write data to {outfilename}"
    return respath


def pzInputsToHDF5(h5file, clrs_ind, clrs_ind_errs, z_specs, i_mags, filt_names=None):
    """pzInputsToHDF5 Saves the photometry inputs as processed for the `process_fors2.photoZ` module, *i.e.* color indices, associated errors and spectro-z if available.
    Filter names must match those used for the photo-z estimation. Allows not to reprocess the catalog everytime the code is used on a similar dataset.

    :param h5file: Name for the HDF5 inputs file to be written.
    :type h5file: str or path-like
    :param clrs_ind: Array of color indices in magnitudes units
    :type clrs_ind: array
    :param clrs_ind_errs: Array of error on color indices in magnitudes units
    :type clrs_ind_errs: array
    :param z_specs: Array of spectroscopic redshifts (`jnp.nan` if unavailable in the catalog)
    :type z_specs: array
    :param i_mags: Array of magnitudes in i-band for nz prior computation.
    :type i_mags: array
    :param filt_names: Names of filters to use as column names in the catalog.
    Color indices will be recorded as `[filter name i]-[filter name i+1]` and `[filter name i]-[filter name i+1]_err`.
    If None, defaults to LSST filters. Defaults to None.
    :type filt_names: list of str, optional
    :return: Absolute path to the written file - if successful.
    :rtype: str or path-like object
    """
    if filt_names is None:
        filt_names = ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]

    color_names = [f"{n1}-{n2}" for (n1, n2) in zip(filt_names[:-1], filt_names[1:], strict=True)]

    color_err_names = [f"{n1}-{n2}_err" for (n1, n2) in zip(filt_names[:-1], filt_names[1:], strict=True)]

    df_clrs = pd.DataFrame(columns=color_names + color_err_names + ["i_mag", "z_spec"], data=jnp.column_stack((clrs_ind, clrs_ind_errs, i_mags, z_specs)))
    outfilename = os.path.abspath(h5file)
    df_clrs.to_hdf(outfilename, "pz_inputs")
    respath = outfilename if os.path.isfile(outfilename) else f"Unable to write data to {outfilename}"
    return respath


def readPZinputsHDF5(h5file, filt_names=None):
    """readPZinputsHDF5 Reads pre-existing photometry inputs for the `process_fors2.photoZ` module, *i.e.* color indices, associated errors and spectro-z if available.
    Filter names must match those used for the photo-z estimation. Allows not to reprocess the catalog everytime the code is used on a similar dataset.

    :param h5file: Path to the HDF5 inputs file.
    :type h5file: str or path-like
    :param filt_names: Names of filters to look for in the catalogs. Color indices `[filter name i]-[filter name i+1]` and `[filter name i]-[filter name i+1]_err` will be returned.
    If None, defaults to LSST filters. Defaults to None.
    :type filt_names: list of str, optional
    :return: 4-tuple of JAX arrays containing data to perform photo-z estimation (`jnp.nan` if missing) : mags in i-band ; color indices ; associated errors and spectro-z.
    :rtype: tuple(arrays)
    """
    if filt_names is None:
        filt_names = ["lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"]

    color_names = [f"{n1}-{n2}" for (n1, n2) in zip(filt_names[:-1], filt_names[1:], strict=True)]

    color_err_names = [f"{n1}-{n2}_err" for (n1, n2) in zip(filt_names[:-1], filt_names[1:], strict=True)]

    df_clrs = pd.read_hdf(os.path.abspath(h5file), key="pz_inputs")
    colrs = jnp.array(df_clrs[color_names])
    colrs_errs = jnp.array(df_clrs[color_err_names])
    i_mags = jnp.array(df_clrs["i_mag"])
    z_specs = jnp.array(df_clrs["z_spec"])
    return i_mags, colrs, colrs_errs, z_specs
