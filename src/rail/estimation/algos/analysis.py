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
import numpy as np
from jax import numpy as jnp
from rail.dsps import DEFAULT_COSMOLOGY
from dsps.cosmology import luminosity_distance_to_z
from astropy import constants as const
from astropy import units as u

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

C_KMS = (const.c).to("km/s").value  # km/s
C_CMS = (const.c).to("cm/s").value  # cm/s
C_AAS = (const.c).to("AA/s").value  # AA/s
C_MS = const.c  # m/s
LSUN = const.L_sun  # Watts
parsec = const.pc  # m
AB0_Lum = (3631.0 * u.Jy * (4 * np.pi * np.power(10 * parsec, 2))).to("W/Hz")

U_LSUNperHz = u.def_unit("Lsun . Hz^{-1}", LSUN * u.Hz**-1)
AB0 = AB0_Lum.to(U_LSUNperHz)  # 3631 Jansky placed at 10 pc in units of Lsun/Hz
U_LSUNperm2perHz = u.def_unit("Lsun . m^{-2} . Hz^{-1}", U_LSUNperHz * u.m**-2)
jy_to_lsun = (1 * u.Jy).to(U_LSUNperm2perHz)

U_FNU = u.def_unit("erg . cm^{-2} . s^{-1} . Hz^{-1}", u.erg / (u.cm**2 * u.s * u.Hz))
U_FL = u.def_unit("erg . cm^{-2} . s^{-1} . AA^{-1}", u.erg / (u.cm**2 * u.s * u.AA))

MPC_TO_M = (1 * u.Mpc).to(u.m).value
LSUN_TO_FNU = (1 * U_LSUNperHz / (u.m * u.m)).to(U_FNU).value


def convert_flux_torestframe(wl, fl, redshift=0.0):
    """
    Shifts the flux values to restframe wavelengths and scales them accordingly.

    Parameters
    ----------
    wl : array
        Wavelengths (unit unimportant) in the observation frame.
    fl : array
        Flux density (unit unimportant).
    redshift : int or float, optional
        Redshift of the object. The default is 0.

    Returns
    -------
    tuple(array, array)
        The spectrum blueshifted to restframe wavelengths.
    """
    factor = 1.0 + redshift
    return wl / factor, fl * factor


def convert_flux_toobsframe(wl, fl, redshift=0.0):
    """
    Shifts the flux values to observed wavelengths and scales them accordingly.

    Parameters
    ----------
    wl : array
        Wavelengths (unit unimportant) in the restframe.
    fl : array
        Flux density (unit unimportant).
    redshift : int or float, optional
        Redshift of the object. The default is 0.

    Returns
    -------
    tuple(array, array)
        The spectrum redshifted to observed wavelengths.
    """
    factor = 1.0 + redshift
    return wl * factor, fl / factor


def convertFlambdaToFnu(wl, flambda):
    """
    Convert spectra density flambda to fnu.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param flambda: flux density in erg/s/cm2 /AA or W/cm2/AA
    :type flambda: float

    :return: fnu, flux density in erg/s/cm2/Hz or W/cm2/Hz
    :rtype: float

    Compute Fnu = wl**2/c Flambda
    check the conversion units with astropy units and constants
    """
    fnu = (flambda * U_FL * (wl * u.AA) ** 2 / const.c).to(U_FNU).value  # / (1 * U_FNU)
    return fnu


def convertFnuToFlambda(wl, fnu):
    """
    Convert spectra density fnu to flambda.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param fnu: flux density in erg/s/cm2/Hz or W/cm2/Hz
    :type fnu: float

    :return: flambda, flux density in erg/s/cm2 /AA or W/cm2/AA
    :rtype: float

    Compute Flambda = Fnu / (wl**2/c)
    check the conversion units with astropy units and constants
    """
    flambda = (fnu * U_FNU * const.c / ((wl * u.AA) ** 2)).to(U_FL).value  # / (1 * U_FL)
    return flambda


def convertFlambdaToFnu_noU(wl, flambda):
    """
    Convert spectra density flambda to fnu.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param flambda: flux density in erg/s/cm2 /AA or W/cm2/AA
    :type flambda: float

    :return: fnu, flux density in erg/s/cm2/Hz or W/cm2/Hz
    :rtype: float

    Compute Fnu = wl**2/c Flambda
    check the conversion units with astropy units and constants
    """
    fnu = flambda * jnp.power(wl, 2) / C_AAS
    return fnu


def convertFnuToFlambda_noU(wl, fnu):
    """
    Convert spectra density fnu to flambda.
    parameters:

    :param wl: wavelength array
    :type wl: float in Angstrom

    :param fnu: flux density in erg/s/cm2/Hz or W/cm2/Hz
    :type fnu: float

    :return: flambda, flux density in erg/s/cm2 /AA or W/cm2/AA
    :rtype: float

    Compute Flambda = Fnu / (wl**2/c)
    check the conversion units with astropy units and constants
    """
    flambda = fnu * C_AAS / jnp.power(wl, 2)
    return flambda


def lsunPerHz_to_fnu(fsun, zob):
    """lsunPerHz_to_fnu _summary_

    :param fsun: _description_
    :type fsun: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY) * u.Mpc  # in meters
    dist_fact = 4 * jnp.pi * (dl.to(u.m) ** 2)  # * (1 + zob)
    fnu = (fsun * U_LSUNperHz / dist_fact).to(U_FNU).value  # .to(u.Jy).to(U_FNU).value
    return fnu


def lsunPerHz_to_fnu_noU(fsun, zob):
    """lsunPerHz_to_fnu _summary_

    :param fsun: _description_
    :type fsun: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY)  # in Mpc
    dist_fact = 4 * jnp.pi * jnp.power(dl * MPC_TO_M, 2)  # * (1 + zob)
    fnu = fsun * LSUN_TO_FNU / dist_fact
    return fnu


def fnu_to_lsunPerHz(fnu, zob):
    """fnu_to_lsunPerHz _summary_

    :param wl: _description_
    :type wl: _type_
    :param fnu: _description_
    :type fnu: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    dl = luminosity_distance_to_z(zob, *DEFAULT_COSMOLOGY) * u.Mpc  # in meters
    dist_fact = 4 * np.pi * (dl.to(u.m) ** 2)  # * (1 + zob)
    fsun = (fnu * U_FNU * dist_fact).to(U_LSUNperHz).value
    return fsun


def lsunPerHz_to_flam(wl, fsun, zob):
    """lsunPerHz_to_flam _summary_

    :param wl: _description_
    :type wl: _type_
    :param fsun: _description_
    :type fsun: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    fnu = lsunPerHz_to_fnu(fsun, zob)
    flam = convertFnuToFlambda(wl, fnu)
    return flam


def lsunPerHz_to_flam_noU(wl, fsun, zob):
    """lsunPerHz_to_flam _summary_

    :param wl: _description_
    :type wl: _type_
    :param fsun: _description_
    :type fsun: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    fnu = lsunPerHz_to_fnu_noU(fsun, zob)
    flam = convertFnuToFlambda_noU(wl, fnu)
    return flam


def flam_to_lsunPerHz(wl, flam, zob):
    """flam_to_lsunPerHz _summary_

    :param wl: _description_
    :type wl: _type_
    :param flam: _description_
    :type flam: _type_
    :param zob: _description_
    :type zob: _type_
    :return: _description_
    :rtype: _type_
    """
    fnu = convertFlambdaToFnu(wl, flam)
    fsun = fnu_to_lsunPerHz(fnu, zob)
    return fsun

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
