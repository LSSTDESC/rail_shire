#!/bin/env python3

import os

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from jax import jit
from sedpy import observate
from tqdm import tqdm
from interpax import interp1d

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid

lightspeed = 2.998e18  # AA/s
ab_gnu = 3.631e-20  # AB reference spctrum in erg/s/cm^2/Hz

sedpyFilter = namedtuple("sedpyFilter", ["name", "wavelength", "transmission"])

# NUV
_wlsnuv = np.arange(1000.0, 3000.0, 1.0)
nuv_transm = np.zeros_like(_wlsnuv)
nuv_transm[(_wlsnuv >= 2100.0) * (_wlsnuv <= 2500.0)] = 1.0
NUV_filt = sedpyFilter(98, _wlsnuv, nuv_transm)

# NIR
_wlsnir = np.arange(20000.0, 25000.0, 1.0)
nir_transm = np.zeros_like(_wlsnir)
nir_transm[(_wlsnir >= 21000.0) * (_wlsnir <= 23000.0)] = 1.0
NIR_filt = sedpyFilter(99, _wlsnir, nir_transm)

# D4000b
_wlsd4k = np.arange(3700.0, 4300.0, 1.0)
d4b_transm = np.zeros_like(_wlsd4k)
d4b_transm[(_wlsd4k >= 3850.0) * (_wlsd4k <= 3950.0)] = 1.0
D4000b_filt = sedpyFilter(96, _wlsd4k, d4b_transm)

# D4000r
d4r_transm = np.zeros_like(_wlsd4k)
d4b_transm[(_wlsd4k >= 4000.0) * (_wlsd4k <= 4100.0)] = 1.0
D4000r_filt = sedpyFilter(97, _wlsd4k, nir_transm)


def get_2lists(filter_list):
    """get_2lists Returns the wavelengths and transmissions of filters in a list of sedpyFilter objects as a 2-tuple :
    `X=([wavelengths], [transmissions])`

    :param filter_list: list of sedpyFilter objects
    :type filter_list: list or array-like
    :return: 2-tuple (list of wavelengths, list of transmissions)
    :rtype: tuple of lists
    """
    transms = [filt.transmission for filt in filter_list]
    waves = [filt.wavelength for filt in filter_list]
    return (waves, transms)


def get_sedpy(filter_dict, wls, data_path="."):
    filts_tup = []
    val_sedpy = observate.list_available_filters()
    print("Loading filters:")
    for _if, (fnam, fdir) in tqdm(enumerate(filter_dict.items()), total=len(filter_dict)):
        if fdir == "" or fdir is None:
            assert fnam in val_sedpy, f"Filter {_if} ({fnam}) is not available.\
                \nPlease provide path to an ASCII file with transmission table or use one of : {val_sedpy}."
            _filt = observate.Filter(fnam)
        else:
            fdir = os.path.abspath(
                os.path.join(
                    data_path,
                    "FILTER",
                    fdir
                )
            )
            _filt = observate.Filter(fnam, directory=fdir)
        filts_tup.append(_filt)

    transm_arr = jnp.array(
        [
            interp1d(
                wls,
                _f.wavelength,
                _f.transmission,
                method="akima",
                extrap=0.0
            ) for _f in filts_tup
        ]
    )
    return transm_arr


def sort(wl, trans):
    """sort Sort the filter's transmission in ascending order of wavelengths

    :param wls: Filter's wavelengths
    :type wls: array
    :param trans: Filter's transmissions
    :type trans: array
    :return: sorted wavelengths and transmission
    :rtype: tuple(array, array)
    """
    _inds = jnp.argsort(wl)
    wls = wl[_inds]
    transm = trans[_inds]
    return wls, transm


def lambda_mean(wls, trans):
    """lambda_mean Computes the mean wavelength of a filter

    :param wls: Filter's wavelengths
    :type wls: array
    :param trans: Filter's transmissions
    :type trans: array
    :return: the mean wavelength of the filter wrt transmission values
    :rtype: float
    """
    mean_wl = trapezoid(wls * trans, wls) / trapezoid(trans, wls)
    return mean_wl


def clip_filter(wls, trans):
    """clip_filter Clips a filter to its non-zero transmission range (removes the edges where transmission is zero)

    :param wls: Filter's wavelengths
    :type wls: array
    :param trans: Filter's transmissions
    :type trans: array
    :return: clipped wavelegths and transmissions and associated descriptors
    :rtype: tuple of arrays and floats
    """
    _indToKeep = np.where(trans >= 0.01 * np.max(trans))[0]
    new_wls = wls[_indToKeep]
    new_trans = trans[_indToKeep]
    _lambdas_peak = new_wls[np.where(new_trans >= 0.5 * np.max(new_trans))[0]]
    min_wl, max_wl = np.min(new_wls), np.max(new_wls)
    minWL_peak, maxWL_peak = np.min(_lambdas_peak), np.max(_lambdas_peak)
    peak_wl = new_wls[np.argmax(new_trans)]
    return new_wls, new_trans, min_wl, max_wl, minWL_peak, maxWL_peak, peak_wl


def transform(wls, trans, trans_type):
    """transform Transforms a filter according to its transmission type

    :param wls: Filter's wavelengths
    :type wls: array
    :param trans: Filter's transmission
    :type trans: array
    :param trans_type: Description of transmission type : number of photons or energy
    :type trans_type: str
    :return: Mean wavelength and transformed transmission
    :rtype: tuple(float, array)
    """
    mean_wl = lambda_mean(wls, trans)
    new_trans = jnp.array(trans * wls / mean_wl) if trans_type.lower() == "photons" and mean_wl > 0.0 else trans
    return mean_wl, new_trans


def load_filt(ident, filterfile, trans_type):
    """load_filt Loads and processes the filter data from the specified file

    :param ident: Filter's identifier
    :type ident: str or int
    :param filterfile: Text (ASCII) file from which to read the filter's data
    :type filterfile: str or path-like
    :param trans_type: Description of transmission type : number of photons or energy
    :type trans_type: str
    :return: Filter's data : identifier, wavelengths and transmission values
    :rtype: tuple(str or int, array, array)
    """
    __wls, __trans = np.loadtxt(os.path.abspath(filterfile), unpack=True)
    _wls, _trans = jnp.array(__wls), jnp.array(__trans)
    wls, transm = sort(_wls, _trans)
    mean_wl, new_trans = transform(wls, transm, trans_type)
    max_trans = jnp.max(new_trans)
    _sel = new_trans >= 0.01 * max_trans
    newwls = jnp.array(wls[_sel])  # noqa: F841
    newtrans = jnp.array(new_trans[_sel])  # noqa: F841
    return ident, wls[_sel], new_trans[_sel]
    # np.savetxt(os.path.join(save_dir, f'{ident}.par'), np.column_stack( (wls[_sel], new_trans[_sel]) ) )
    # return ident, sedpy.observate.Filter(f'{ident}', directory=save_dir)


#################################################
# THE FOLLOWING IS ADAPTED FROM sedpy.observate #
#################################################

# try:
#    from pkg_resources import resource_filename, resource_listdir
# except(ImportError):
#    pass
# from sedpy.reference_spectra import vega, solar, sedpydir


# __all__ = ["Filter", "FilterSet", "load_filters", "list_available_filters", "getSED", "air2vac", "vac2air", "Lbol"]


def noJit_get_properties(filtwave, filttransm):
    """Determine and store a number of properties of the filter and store
    them in the object.  These properties include several 'effective'
    wavelength definitions and several width definitions, as well as the
    in-band absolute AB solar magnitude, the Vega and AB reference
    zero-point detector signal, and the conversion between AB and Vega
    magnitudes.

    See Fukugita et al. (1996) AJ 111, 1748 for discussion and definition
    of many of these quantities.
    """
    # Calculate some useful integrals
    i0 = trapezoid(filttransm * jnp.log(filtwave), x=jnp.log(filtwave))
    i1 = trapezoid(filttransm, x=jnp.log(filtwave))
    i2 = trapezoid(filttransm * filtwave, x=filtwave)
    i3 = trapezoid(filttransm, x=filtwave)

    wave_effective = jnp.exp(i0 / i1)
    wave_pivot = jnp.sqrt(i2 / i1)  # noqa: F841
    wave_mean = wave_effective  # noqa: F841
    wave_average = i2 / i3  # noqa: F841
    rectangular_width = i3 / jnp.max(filttransm)  # noqa: F841

    i4 = trapezoid(filttransm * jnp.power((jnp.log(filtwave / wave_effective)), 2.0), x=jnp.log(filtwave))
    gauss_width = jnp.power((i4 / i1), 0.5)
    effective_width = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * gauss_width * wave_effective  # noqa: F841

    # Get zero points and AB to Vega conversion
    ab_zero_counts = obj_counts_hires(filtwave, filttransm, filtwave, ab_gnu * lightspeed / filtwave**2)
    return ab_zero_counts


@jit
def get_properties(filtwave, filttransm):
    """Determine and store a number of properties of the filter and store
    them in the object.  These properties include several 'effective'
    wavelength definitions and several width definitions, as well as the
    in-band absolute AB solar magnitude, the Vega and AB reference
    zero-point detector signal, and the conversion between AB and Vega
    magnitudes.

    See Fukugita et al. (1996) AJ 111, 1748 for discussion and definition
    of many of these quantities.
    """
    # Calculate some useful integrals
    i0 = trapezoid(filttransm * jnp.log(filtwave), x=jnp.log(filtwave))
    i1 = trapezoid(filttransm, x=jnp.log(filtwave))
    i2 = trapezoid(filttransm * filtwave, x=filtwave)
    i3 = trapezoid(filttransm, x=filtwave)

    wave_effective = jnp.exp(i0 / i1)
    wave_pivot = jnp.sqrt(i2 / i1)  # noqa: F841
    wave_mean = wave_effective  # noqa: F841
    wave_average = i2 / i3  # noqa: F841
    rectangular_width = i3 / jnp.max(filttransm)  # noqa: F841

    i4 = trapezoid(filttransm * jnp.power((jnp.log(filtwave / wave_effective)), 2.0), x=jnp.log(filtwave))
    gauss_width = jnp.power((i4 / i1), 0.5)
    effective_width = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * gauss_width * wave_effective  # noqa: F841

    # Get zero points and AB to Vega conversion
    ab_zero_counts = obj_counts_hires(filtwave, filttransm, filtwave, ab_gnu * lightspeed / filtwave**2)
    return ab_zero_counts

    """
    # If blue enough get AB mag of vega
    if wave_mean < 1e6:
        vega_zero_counts = obj_counts_hires(filtwave, filttransm, vega[:, 0], vega[:, 1])
        _ab_to_vega = -2.5 * jnp.log10(ab_zero_counts / vega_zero_counts)
    else:
        vega_zero_counts = float('NaN')
        _ab_to_vega = float('NaN')
    # If blue enough get absolute solar magnitude
    if wave_mean < 1e5:
        solar_ab_mag = ab_mag(filtwave, filt_trans, solar[:,0], solar[:,1])
    else:
        solar_ab_mag = float('NaN')
    """

    '''
    @property
    def ab_to_vega(self):
        """The conversion from AB to Vega systems for this filter.  It has the
        sense

        :math:`m_{Vega} = m_{AB} + Filter().ab_to_vega`
        """
        return self._ab_to_vega
    '''


def noJit_obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the detector signal.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    counts : ndarray of shape ``(N_source,)``
        Detector signal(s).
    """
    # Interpolate filter transmission to source spectrum
    newtrans = jnp.interp(sourcewave, filtwave, filt_trans, left=0.0, right=0.0, period=None)

    # Integrate lambda*f_lambda*R
    counts = trapezoid(sourcewave * newtrans * sourceflux, x=sourcewave)
    return counts


@jit
def obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the detector signal.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    counts : ndarray of shape ``(N_source,)``
        Detector signal(s).
    """
    # Interpolate filter transmission to source spectrum
    newtrans = jnp.interp(sourcewave, filtwave, filt_trans, left=0.0, right=0.0, period=None)

    # Integrate lambda*f_lambda*R
    counts = trapezoid(sourcewave * newtrans * sourceflux, x=sourcewave)
    return counts


def noJit_ab_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the AB magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        AB magnitude of the source(s).
    """
    ab_zero_counts = noJit_get_properties(filtwave, filt_trans)
    print(f"AB-counts={ab_zero_counts}")
    counts = noJit_obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    print(f"filter counts={counts}")
    return -2.5 * jnp.log10(counts / ab_zero_counts)


@jit
def ab_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the AB magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        AB magnitude of the source(s).
    """
    ab_zero_counts = get_properties(filtwave, filt_trans)
    counts = obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    return -2.5 * jnp.log10(counts / ab_zero_counts)

'''
@jit
def vega_mag(filtwave, filt_trans, sourcewave, sourceflux):
    """Project source spectrum onto filter and return the Vega magnitude.

    Parameters
    ----------
    sourcewave : ndarray of shape ``(N_pix,)``
        Spectrum wavelength (in Angstroms). Must be monotonic increasing.

    sourceflux : ndarray of shape ``(N_source, N_pix)``
        Associated flux (assumed to be in erg/s/cm^2/AA)

    Returns
    -------
    mag : float or ndarray of shape ``(N_source,)``
        Vega magnitude of the source(s).
    """
    counts = obj_counts_hires(filtwave, filt_trans, sourcewave, sourceflux)
    return -2.5 * jnp.log10(counts / self.vega_zero_counts)
'''

##########################################
# Fin de l'adaptation de sedpy.observate #
##########################################
