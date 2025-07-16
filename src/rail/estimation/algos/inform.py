import os
import jax
import numpy as np
from collections import namedtuple
from functools import partial
from jax import numpy as jnp
from jax import jit, vmap
from jax.tree_util import tree_map
from jax.scipy.special import gamma as jgamma
#from jax.scipy.optimize import minimize as jmini
import scipy.optimize as sciop
#from jax import random as jrn

import pandas as pd
#import qp
from tqdm import tqdm
import tables_io
from sklearn.ensemble import RandomForestClassifier
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
#from rail.utils.path_utils import RAILDIR
from rail.core.data import TableHandle, ModelHandle
from rail.core.common_params import SHARED_PARAMS

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns

from .io_utils import load_ssp, istuple, SHIREDATALOC
from .analysis import _DUMMY_PARS, lsunPerHz_to_fnu_noU, C_KMS, convert_flux_toobsframe #, PARAMS_MAX, PARAMS_MIN, INIT_PARAMS
from .template import (
    vmap_cols_zo,
    vmap_cols_zo_leg,
    colrs_bptrews_templ_zo,
    colrs_bptrews_templ_zo_leg,
    lim_HII_comp,
    lim_seyf_liner,
    Ka03_nii,
    Ke01_nii,
    Ke01_oi,
    Ke01_sii,
    Ke06_oi,
    Ke06_sii,
    vmap_mean_spectrum_nodust,
    v_d4000n,
    calc_d4000n,
    v_nuvk,
    calc_nuvk,
    mean_spectrum_nodust,
    vmap_calc_eqw
)
from .filter import get_sedpy
#from .cosmology import prior_mod

jax.config.update("jax_enable_x64", True)

PriorParams = namedtuple("PriorParams", ["mod", "type", "fo", "kt", "z0", "alpha", "km", "m0", "nuv_range"])

@jit
def nz_func(mz, z0, alpha, km, m0):  # pragma: no cover
    #z0, alpha, km = X
    mm, z = mz
    m = jnp.where(mm<m0, m0, mm)
    zm = z0 + (km * (m - m0))
    vals = jnp.power(z, alpha) * jnp.exp(- jnp.power((z / zm), alpha))
    Inorm = jnp.power(zm, alpha+1) * jgamma(1 + 1 / alpha) / alpha
    return vals / Inorm

vmap_dndz_gals = vmap(
    nz_func,
    in_axes=(0, None, None, None, None)
)

vmap_dndz = vmap(
    vmap(
        nz_func,
        in_axes=(0, None, None, None, None)
    ),
    in_axes=(None, 0, 0, 0, 0)
)


@jit
def frac_func(X, m0, m):
    fo, kt = X
    mm = jnp.where(m<m0, m0, m)
    return fo * jnp.exp(-kt * (mm - m0))

_vmap_frac = vmap(
    vmap(
        frac_func,
        in_axes=(None, None, 0)
    ),
    in_axes=(0, 0, None)
)

def vmap_frac(X, m0, m):
    _vals = _vmap_frac(X, m0, m)
    _sum = jnp.nansum(_vals, axis=0)
    return _vals/_sum

class ShireInformer(CatInformer):
    name = "ShireInformer"
    outputs = [("model", ModelHandle), ("templates", TableHandle)]
    config_options = CatInformer.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,
        err_bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        m0=Param(float, 20.0, msg="reference apparent mag, used in prior param"),
        data_path=Param(
            str,
            "None",
            msg="data_path (str): file path to the "
                "SSP, SED, FILTER, and AB directories.  If left to "
                "default `None` it will use the install "
                "directory for rail + rail/examples_data/estimation_data/data"
        ),
        spectra_file=Param(
            str,
            "dsps_valid_fits_F2_GG_DESI_SM3.h5",
            msg="name of the file specifying the set of galaxies from which to pick templates."
        ),
        templ_type=Param(
            str,
            "SPS",
            msg='Whether to use the "SPS" or "Legacy" method to derive the templates colours from the SPS parameters.'
        ),
        ssp_file=Param(
            str,
            "ssp_data_fsps_v3.2_lgmet_age.h5",
            msg="ssp_file (str): name of the h5 file that contains the SSP for use with DSPS."
        ),
        filter_dict=Param(
            dict,
            { f"{_n}_lsst": "filt_lsst" for _n in "ugrizy" },
            msg='filter_dict (dict): Dictionary `{name: directory}` of the filters to be loaded with `sedpy`.'
                'If `name` is available in `sedpy`, `directory` can be set to `None` or `""`.'
        ),
        wlmin=Param(
            float,
            100.,
            msg='wlmin (float): lower bound of wavelength grid for filters interpolation'
        ),
        wlmax=Param(
            float,
            15000.,
            msg='wlmax (float): upper bound of wavelength grid for filters interpolation'
        ),
        dwl=Param(
            float,
            100.,
            msg='dwl (float): step of wavelength grid for filters interpolation'
        ),
        randomsel=Param(
            bool,
            False,
            msg='randomsel (bool): whether to select a random sample of the galaxies in `spectra_file` instead of fitting it to training data colours.'
                'In that case, `colrsbins` specifies the size of the random sample.'
        ),
        colrsbins=Param(
            int,
            40,
            msg='colrsbins (int): number of bins for each colour index in which to select the template with the best score.'
                'If `randomsel` is `True`, this specifies the number of randomly selected galaxies to be used as templates.'
        ),
        init_kt=Param(float, 0.3, msg="initial guess for kt in training"),
        init_z0=Param(float, 0.4, msg="initial guess for z0 in training"),
        init_m0=Param(float, 20, msg="initial guess for m0 in training"),
        init_alpha=Param(float, 1.8, msg="initial guess for alpha in training"),
        init_km=Param(float, 0.1, msg="initial guess for km in training"),
        refcategs=Param(list, ["E_S0", "Sbc/Scd", "Irr"], msg="Galaxy types for prior functions.")
    )

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            self.data_path = SHIREDATALOC
        else:  # pragma: no cover
            self.data_path = datapath
            os.environ["SHIREDATALOC"] = self.data_path
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError("SHIREDATALOC " + self.data_path + " does not exist! Check value of data_path in config file!")
        
        self.m0 = None
        self.fo_arr = None
        self.kt_arr = None
        self.typmask = None
        self.mags = None
        self.refmags = None
        self.szs = None
        self.pzs = None
        self.refcategs = np.array(self.config.refcategs)
        self.ntyp = len(self.refcategs)
        self.nt_array = None
        self.besttypes = None
        self.templates_df = None
        self.filters_names = None
        self.color_names = None
        self.e0_pars = PriorParams(0, self.refcategs[0], None, None, None, None, None, None, (4.25, jnp.inf))
        self.sbcd_pars = PriorParams(1, self.refcategs[1], None, None, None, None, None, None, (1.9, 4.25))
        #self.sbc_pars = PriorParams(1, self.refcategs[1], None, None, None, None, None, None, (3.1.9, 4.25))
        #self.scd_pars = PriorParams(2, self.refcategs[2], None, None, None, None, None, None, (1.9, 3.19))
        self.irr_pars = PriorParams(2, self.refcategs[2], None, None, None, None, None, None, (-jnp.inf, 1.9))

    @partial(jit, static_argnums=0)
    def prior_mod(self, nuvk):
        """prior_z0 Determines the model (galaxy morphology) for which to compute the prior value.

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: Model Id
        :rtype: int
        """
        val = (
            self.irr_pars.mod
            + (self.sbcd_pars.mod - self.irr_pars.mod) * jnp.heaviside(nuvk - self.sbcd_pars.nuv_range[0], 0)
            + (self.e0_pars.mod - self.sbcd_pars.mod) * jnp.heaviside(nuvk - self.e0_pars.nuv_range[0], 0)
        )
        return val.astype(int)


    def open_templates(self, **kwargs):
        """Load the templates parameters and attach them to this Estimator

        Parameters
        ----------
        templates : ``object``, ``str`` or ``TableHandle``
            Either an object with the array of parameters, a path pointing to a file
            that can be read to obtain the templates, or a `TableHandle`
            providing access to the templates.

        Returns
        -------
        self.templates : ``object``
            The object encapsulating the templates.
        """
        templates = kwargs.get("templates", None)
        if templates is None or templates == "None":
            self.templates_df = None
            return self.templates_df
        if isinstance(templates, str):
            self.templates_df = self.set_data("templates", data=None, path=templates)
            self.config["templates"] = templates
            return self.templates
        if isinstance(templates, TableHandle):
            if templates.has_path:
                self.config["templates"] = templates.path
        self.templates_df = self.set_data("templates", templates)
        return self.templates_df


    def _load_training(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        if self.config.ref_band not in training_data.keys():  # pragma: no cover
            raise KeyError(f"ref_band {self.config.ref_band} not found in input data!")
        if self.config.redshift_col not in training_data.keys():  # pragma: no cover
            raise KeyError(f"redshift column {self.config.redshift_col} not found in input data!")

        self.mags = jnp.column_stack(
            [training_data[f"mag_{_n}"] for _n in self.filters_names]
        )
        self.refmags = jnp.array(
            training_data[self.config.ref_band]
        )

        self.szs = jnp.array(
            training_data[self.config["redshift_col"]]
        )

        self.pzs = np.histogram_bin_edges(self.szs, bins='auto')

    def _load_filters(self):
        wls = jnp.arange(
            self.config.wlmin,
            self.config.wlmax+self.config.dwl,
            self.config.dwl
        )

        transm_arr = get_sedpy(self.config.filter_dict, wls, self.data_path)
        self.filters_names = [_fnam for _fnam, _fdir in self.config.filter_dict.items()]
        self.color_names = [f"{n1}-{n2}" for n1,n2 in zip(self.filters_names[:-1], self.filters_names[1:])]

        return wls, transm_arr

    def _load_templates(self):
        # The redshift range we will evaluate on
        pzs = np.histogram_bin_edges(self.szs, bins='auto')

        sspdata = load_ssp(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SSP",
                    self.config.ssp_file
                )
            )
        )

        fwls, ftransm = self._load_filters()

        '''
        spectra_file = os.path.join(self.data_path, "SED", self.config.spectra_file)
        if not os.path.isfile(spectra_file):
            print(f"{spectra_file} does not exist ! Trying with local file instead :")
            spectra_file = os.path.abspath(self.config.spectra_file)
            if os.path.isfile(spectra_file):
                print(f"New file: {spectra_file}")
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), spectra_file)

        templs_df = tables_io.read(
            spectra_file,
            tables_io.types.PD_DATAFRAME,
            fmt='hf5'
        )
        '''
        templs_df = self.templates_df
        templ_pars_arr = jnp.array(templs_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
        templ_zref = jnp.array(templs_df[self.config.redshift_col])

        if "sps" in self.config.templ_type.lower():
            templ_tupl = [tuple(_pars) for _pars in templ_pars_arr]
            templ_tupl_sps = tree_map(
                lambda partup: colrs_bptrews_templ_zo(
                    jnp.array(partup),
                    fwls,
                    pzs,
                    ftransm,
                    sspdata
                ),
                templ_tupl,
                is_leaf=istuple
            )
        else:
            templ_tupl = [tuple(_pars)+tuple([_z]) for _pars, _z in zip(templ_pars_arr, templ_zref, strict=True)]
            templ_tupl_sps = tree_map(
                lambda partup: colrs_bptrews_templ_zo_leg(
                    jnp.array(partup[:-1]),
                    fwls,
                    pzs,
                    partup[-1],
                    ftransm,
                    sspdata
                ),
                templ_tupl,
                is_leaf=istuple
            )

        filters_names = [_fnam for _fnam, _fdir in self.config.filter_dict.items()]
        color_names = [f"{n1}-{n2}" for n1,n2 in zip(filters_names[:-1], filters_names[1:])]
        lines_names = [
            "SF_[OII]_3728.48_REW",
            "Balmer_HI_4862.68_REW",
            "AGN_[OIII]_5008.24_REW",
            "SF_[OI]_6302.046_REW",
            "Balmer_HI_6564.61_REW",
            "AGN_[NII]_6585.27_REW",
            "AGN_[SII]_6718.29_REW"
        ]
        templs_as_dict = {}
        for it, (tname, row) in enumerate(templs_df.iterrows()):
            _colrrews = templ_tupl_sps[it]
            _df = pd.DataFrame(columns=color_names+['NUVK', 'D4000n']+lines_names, data=_colrrews)
            _df['z_p'] = pzs
            _df['Dataset'] = np.full(pzs.shape, row['Dataset'])
            _df['name'] = np.full(pzs.shape, tname)
            _df[self.config.redshift_col] = np.full(pzs.shape, row[self.config.redshift_col])
            templs_as_dict.update({f"{tname}": _df})
        all_templs_df = pd.concat(
            [_df for _, _df in templs_as_dict.items()],
            ignore_index=True
        )

        return all_templs_df


    @partial(jit, static_argnums=0) #, 2, 3))
    def _frac_likelihood(self, frac_params, mags): #, btyp, idxbtyp):
        _foi = frac_params[:self.ntyp]
        _kti = frac_params[self.ntyp:]
        X = (_foi, _kti) #jnp.vstack((_foi, _kti)).T
        probs = vmap_frac(X, self.m0, mags)
        probsel = probs[self.besttypes, jnp.arange(len(self.besttypes))]
        likelihood = jnp.where(probsel>0, -jnp.log(probsel), 0)
        return jnp.nansum(likelihood)


    @partial(jit, static_argnums=0) #, 2, 3))
    def _frac_likelihood_combined(self, frac_params, mags): #, btyp, idxbtyp):
        _foi = frac_params[:self.ntyp]
        _kti = frac_params[self.ntyp:2*self.ntyp]
        _moi = frac_params[2*self.ntyp:]
        X = (_foi, _kti) #jnp.vstack((_foi, _kti)).T
        probs = vmap_frac(X, _moi, mags)
        probsel = probs[self.besttypes, jnp.arange(len(self.besttypes))]
        likelihood = jnp.where(probsel>0, -jnp.log(probsel), 0)
        return jnp.nansum(likelihood)


    @partial(jit, static_argnums=0)
    def _dn_dz_likelihood(self, pars, mags, zs):
        lik = vmap_dndz_gals((mags, zs), *pars)
        nllik = jnp.nansum(jnp.where(lik>0, -jnp.log(lik), 0))
        return nllik


    @partial(jit, static_argnums=0)
    def _dn_dz_likelihood_combined(self, pars, mags, zs):
        _moi, _zoi, _alphai, _kmi = pars[:self.ntyp], pars[self.ntyp:2*self.ntyp], pars[2*self.ntyp:3*self.ntyp], pars[3*self.ntyp:]
        lik = vmap_dndz((mags, zs), _zoi, _alphai, _kmi, _moi)
        liksel = lik[self.besttypes, jnp.arange(len(self.besttypes))]
        nllik = jnp.nansum(jnp.where(liksel>0, -jnp.log(lik), 0))
        return nllik

    @partial(jit, static_argnums=0)
    def _combined_nllik(self, pars, mags, zs):
        fracpars = pars[:3*self.ntyp]
        dnzpars = pars[2*self.ntyp:] # overlap for m0
        return self._frac_likelihood_combined(fracpars, mags)+self._dn_dz_likelihood_combined(dnzpars, mags, zs)


    def _fit_prior(self):
        print("Fitting prior...")
        fo_init = jnp.full(self.ntyp, 1/self.ntyp)
        kt_init = jnp.full(self.ntyp, self.config.init_kt)
        m0_init = jnp.full(self.ntyp, self.config.init_m0)
        z0_init = jnp.full(self.ntyp, self.config.init_z0)
        al_init = jnp.full(self.ntyp, self.config.init_alpha)
        km_init = jnp.full(self.ntyp, self.config.init_km)
        
        initparams = jnp.concatenate(
            (fo_init, kt_init, m0_init, z0_init, al_init, km_init)
        )

        #def funconstr(X):
        #    fo, kt, m0 = X[:self.ntyp], X[self.ntyp:2*self.ntyp], X[2*self.ntyp:3*self.ntyp]
        #    vals = _vmap_frac((fo, kt), m0, self.refmags)
        #    return jnp.nansum(vals, axis=0)

        constrmatrx = jnp.vstack(
            (
                jnp.concatenate(
                    (jnp.ones(self.ntyp), jnp.zeros(initparams.shape[0]-self.ntyp))
                ),
                jnp.concatenate(
                    (
                        jnp.identity(self.ntyp),
                        jnp.zeros((self.ntyp, initparams.shape[0]-self.ntyp))
                    ),
                    axis=1
                ),
                jnp.concatenate(
                    (
                        jnp.zeros((self.ntyp, 3*self.ntyp)),
                        jnp.identity(self.ntyp),
                        jnp.zeros((self.ntyp, initparams.shape[0]-4*self.ntyp))
                    ),
                    axis=1
                )
            )
        )
        lb = jnp.concatenate(
            (jnp.ones(1), jnp.zeros(constrmatrx.shape[0]-1))
        )
        ub = jnp.concatenate(
            (jnp.ones(1+self.ntyp), jnp.full(self.ntyp, jnp.inf))
        )
        print(f"Constraints matrices: {constrmatrx, lb, ub}")
        _results = sciop.minimize(
            lambda P: self._combined_nllik(P, self.refmags, self.szs), initparams,
            method="COBYQA",# "Nelder-Mead", #
            #bounds=[
            #    (0, 1), (0, 1), (0, 1), (0, 1),
            #    (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)
            #],
            constraints=[
                sciop.LinearConstraint(
                    constrmatrx,
                    lb,
                    ub
                ),
            #    sciop.NonlinearConstraint(funconstr, jnp.ones_like(self.refmags), jnp.ones_like(self.refmags))
            ]
        ).x
        tmpfo = _results[:self.ntyp]
        # minimizer can sometimes give fractions greater than one, if so normalize
        fracnorm = jnp.sum(tmpfo)
        self.fo_arr = tmpfo/fracnorm
        self.kt_arr = _results[self.ntyp:2*self.ntyp]
        self.m0 = _results[2*self.ntyp:3*self.ntyp]
        zo_arr = _results[3*self.ntyp:4*self.ntyp]
        alpha_arr = _results[4*self.ntyp:-self.ntyp]
        km_arr = _results[-self.ntyp:]
        
        for i, (ifo, ikt, imo, izo, ial, ikm) in enumerate(zip(
            self.fo_arr, self.kt_arr, self.m0, zo_arr, alpha_arr, km_arr )):
            print(f"best fit f0, kt, m0, z0, alpha, km for type {i}: {ifo, ikt, imo, izo, ial, ikm}")
        
        return zo_arr, alpha_arr, km_arr
        #return None


    #@partial(jit, static_argnums=0)
    def _find_fractions(self):
        # set up fo and kt arrays, choose default start values
        fo_init = jnp.full(self.ntyp, 1/self.ntyp)
        kt_init = jnp.full(self.ntyp, self.config.init_kt)
        fracparams = jnp.concatenate((fo_init, kt_init))
        print("Finding fractions...")
        # run scipy optimize to find best params
        # note that best fit vals are stored as "x" for some reason
        '''
        frac_results = jmini(
            self._frac_likelihood, fracparams,
            method="BFGS"
        ).x
        '''

        foconstrmatrx = jnp.vstack(
            (
                jnp.concatenate((jnp.ones_like(fo_init), jnp.zeros_like(kt_init))),
                jnp.concatenate(
                    (
                        jnp.identity(self.ntyp),
                        jnp.zeros((self.ntyp, self.ntyp))
                    ),
                    axis=1
                )
            )
        )

        lb = jnp.concatenate(
            (jnp.ones(1), jnp.zeros(foconstrmatrx.shape[0]-1))
        )
        ub = jnp.ones(foconstrmatrx.shape[0])
        
        #minmags = jnp.where(self.refmags<self.m0, self.m0, self.refmags)
        #def funconstr(X):
        #    fo, kt = X[:self.ntyp], X[self.ntyp:]
        #    vals = _vmap_frac((fo, kt), self.m0, self.refmags)
        #    return jnp.nansum(vals, axis=0)
        
        frac_results = sciop.minimize(
            lambda P: self._frac_likelihood(P, self.refmags), fracparams,
            method="COBYQA",# "Nelder-Mead", #
            #bounds=[
            #    (0, 1), (0, 1), (0, 1), (0, 1),
            #    (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)
            #],
            constraints=[
                sciop.LinearConstraint(
                    foconstrmatrx,
                    lb,
                    ub
                ),
            #    sciop.NonlinearConstraint(funconstr, jnp.ones_like(self.refmags), jnp.ones_like(self.refmags))
            ]
        ).x
        tmpfo = frac_results[:self.ntyp]
        # minimizer can sometimes give fractions greater than one, if so normalize
        fracnorm = jnp.sum(tmpfo)
        self.fo_arr = tmpfo/fracnorm
        self.kt_arr = frac_results[self.ntyp:]


    #@partial(jit, static_argnums=0)
    def _find_dndz_params(self):
        # initial parameters for zo, alpha, and km
        mo_arr = []
        zo_arr = []
        a_arr = []
        km_arr = []
        print("Fitting prior parameters...")
        for i in range(self.ntyp):
            print(f"minimizing for type {i}")
            typmask = jnp.array([b == i for b in self.besttypes])
            _m = self.refmags[typmask]
            zarr = self.szs[typmask]
            dndzparams = jnp.array([self.config.init_m0, self.config.init_z0, self.config.init_alpha, self.config.init_km])
            Aconstraint = jnp.array(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]
            )
            lb = jnp.array([0, 0, 0, 0])
            ub = jnp.array([0, jnp.inf, 0, 0])
            moi, zoi, alfi, kmi = sciop.minimize(
                lambda P: self._dn_dz_likelihood(P, _m, zarr),
                dndzparams,
                method="COBYQA", #"Nelder-Mead", #
                constraints=(
                    sciop.LinearConstraint(
                        Aconstraint,
                        lb,
                        ub
                    )
                )
                #bounds=[(0, jnp.inf), (0, jnp.inf), (-jnp.inf, jnp.inf), (-jnp.inf, jnp.inf)]
            ).x
            mo_arr.append(moi)
            zo_arr.append(zoi)
            a_arr.append(alfi)
            km_arr.append(kmi)
            print(f"best fit m0, z0, alpha, km for type {i}: {moi, zoi, alfi, kmi}")
        return jnp.array(mo_arr), jnp.array(zo_arr), jnp.array(km_arr), jnp.array(a_arr)


    def class_nuvk(self, test_df):
        #self._load_filters()
        all_tsels_df = self._nuvk_classif()
        classifier = RandomForestClassifier() # use defaults settings for now
        X = np.clip(all_tsels_df[self.color_names], -3.4e+38, 3.4e+38)
        y = np.array(all_tsels_df['CAT_NUVK'])
        classifier.fit(X, y)

        Xtest = np.clip(test_df[self.color_names], -3.4e+38, 3.4e+38)
        ytest = classifier.predict(Xtest)
        #yvals, ycounts = np.unique(ytest, return_counts=True)

        test_df['CAT_NUVK'] = ytest
        self.besttypes = [np.argwhere(self.refcategs==cat)[0][0] for cat in ytest]

        z0list, alflist, kmlist = self._fit_prior()

        #molist, z0list, alflist, kmlist = self._find_dndz_params()
        #self.m0 = molist
        #self._find_fractions()

        return z0list, alflist, kmlist #jnp.array(z0list), jnp.array(alflist), jnp.array(kmlist)


    def run(self):
        wls, transm_arr = self._load_filters()
        self._load_training()

        train_df = pd.DataFrame(
            data=jnp.column_stack(
                (self.mags[:, :-1]-self.mags[:, 1:], self.refmags, self.szs)
            ),
            columns=self.color_names+[self.config.ref_band, self.config.redshift_col]
        )

        templs_ref_df = pd.read_hdf(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SED",
                    self.config.spectra_file
                )
            ),
            key="fit_dsps"
        )

        if self.config.randomsel:
            print(f"Selecting {self.config.colrsbins} random templates.")
            templs_score_df = templs_ref_df.sample(n=self.config.colrsbins, replace=False)
            templs_score_df['score'] = np.full(self.config.colrsbins, -1)
            templs_score_df['name'] = templs_score_df.index
        else:
            print(f"Using training data to select the best templates from {self.config.colrsbins} bins for each colour index.")
            pars_arr = jnp.array(templs_ref_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
            templ_zref = jnp.array(templs_ref_df[self.config.redshift_col])

            ssp_data = load_ssp(
                os.path.abspath(
                    os.path.join(
                        self.data_path,
                        "SSP",
                        self.config.ssp_file
                    )
                )
            )
            if "sps" in self.config.templ_type.lower():
                templ_tupl = [tuple(_pars) for _pars in pars_arr]
                templ_tupl_sps = tree_map(
                    lambda partup: vmap_cols_zo(
                        jnp.array(partup),
                        wls,
                        self.pzs,
                        transm_arr,
                        ssp_data
                    ),
                    templ_tupl,
                    is_leaf=istuple
                )
            else:
                templ_tupl = [tuple(_pars)+tuple([_z]) for _pars, _z in zip(pars_arr, templ_zref, strict=True)]
                templ_tupl_sps = tree_map(
                    lambda partup: vmap_cols_zo_leg(
                        jnp.array(partup[:-1]),
                        wls,
                        self.pzs,
                        partup[-1],
                        transm_arr,
                        ssp_data
                    ),
                    templ_tupl,
                    is_leaf=istuple
                )

            templs_as_dict = {}
            for it, (tname, row) in enumerate(templs_ref_df.iterrows()):
                _colrs = templ_tupl_sps[it]
                _df = pd.DataFrame(columns=self.color_names, data=_colrs)
                _df['z_p'] = self.pzs
                _df['Dataset'] = np.full(self.pzs.shape, row['Dataset'])
                _df['name'] = np.full(self.pzs.shape, tname)
                templs_as_dict.update({f"{tname}": _df})
            all_templs_df = pd.concat(
                [_df for _, _df in templs_as_dict.items()],
                ignore_index=True
            )

            list_edges = []
            for idc, c in enumerate(self.color_names):
                _arr = np.array(train_df[c])
                #H_data_1D, _edges1d = np.histogram(_arr[np.isfinite(_arr)], bins=self.config.colrsbins) #, bins='auto') #
                #H_templ_1d, _edges1d = np.histogram(np.array(all_templs_df[c]), bins=_edges1d)
                #H_data_1D, _edges1d = np.histogram(_arr[np.isfinite(_arr)], bins='auto')
                #H_templ_1d, _edges1d = np.histogram(np.array(all_templs_df[c]), bins=_edges1d)
                _edges1d = np.histogram_bin_edges(_arr[np.isfinite(_arr)], bins=self.config.colrsbins)
                list_edges.append(_edges1d)

            coords = []
            for c, b in zip(self.color_names, list_edges):
                c_idxs = np.digitize(train_df[c], b)
                coords.append(c_idxs)
            coords = np.column_stack(coords)
            train_df[[f'{c}_bin' for c in self.color_names]] = coords

            templ_coords = []
            for c, b in zip(self.color_names, list_edges):
                c_idxs = np.digitize(all_templs_df[c], b)
                templ_coords.append(c_idxs)
            templ_coords = np.column_stack(templ_coords)
            all_templs_df[[f'{c}_bin' for c in self.color_names]] = templ_coords

            best_templs_names = []
            allbestscores = []
            print("Computing scores in colour bins:")
            for c in self.color_names:
                for cbin in tqdm(jnp.unique(train_df[f'{c}_bin'].values)):
                #cbin = row[f'{c}_bin']
                    sel = train_df[f'{c}_bin']==cbin
                    _sel_df = train_df[sel]
                    zs = jnp.array(_sel_df[self.config["redshift_col"]].values)
                    sel_templ = all_templs_df[f'{c}_bin']==cbin
                    _templ_df = all_templs_df[sel_templ]
                    scores = jnp.array(
                        [
                            jnp.sum(jnp.abs(zs-zp)/(1+zs)) / zs.shape[0] if zs.shape[0]>0 else jnp.nan for zp in _templ_df['z_p']
                        ]
                    )
                    if scores.shape[0]>0 and not jnp.all(jnp.isnan(scores)):
                        ix_best = int(jnp.nanargmin(scores))
                        bestscore = scores[ix_best]
                        if bestscore < 0.15:
                            best_templs_names.append(_templ_df['name'].iloc[ix_best])
                            allbestscores.append(scores[ix_best])

            best_templ_sels = np.unique(best_templs_names)
            allbestscores = jnp.array(allbestscores)

            meanscores = []
            print("Finalising templates:")
            for it, nt in tqdm(enumerate(best_templ_sels), total=len(best_templ_sels)):
                _sel = jnp.array([_t==nt for _t in best_templs_names])
                _sc = allbestscores[_sel]
                meanscores.append(jnp.nanmean(_sc))
            meanscores = jnp.array(meanscores)

            templs_score_df = templs_ref_df.loc[best_templ_sels]
            for msc, tn in zip(meanscores, best_templ_sels):
                templs_score_df.loc[tn, 'score'] = float(msc)
                templs_score_df.loc[tn, 'name'] = tn
            templs_score_df.sort_values('score', ascending=True, inplace=True)

        tables_io.write(
            templs_score_df,
            self.config.output,
            fmt='h5' #'hf5'
        )

        self.templates_df = templs_score_df[["name", "num", "score", "Dataset", self.config["redshift_col"]]+_DUMMY_PARS.PARAM_NAMES_FLAT]
        
        res_classif = self.class_nuvk(train_df)
        
        self.model = dict(
            fo_arr=self.fo_arr,
            kt_arr=self.kt_arr,
            zo_arr=np.array(res_classif[0]),
            a_arr=np.array(res_classif[1]),
            km_arr=np.array(res_classif[2]),
            mo=self.m0,
            nt_array=self.nt_array
        )
        self.e0_pars = PriorParams(
            0,
            self.refcategs[0],
            self.model["fo_arr"][0],
            self.model["kt_arr"][0],
            self.model["zo_arr"][0],
            self.model["a_arr"][0],
            self.model["km_arr"][0],
            self.model["mo"][0],
            (4.25, jnp.inf)
        )
        self.sbcd_pars = PriorParams(
            1,
            self.refcategs[1],
            self.model["fo_arr"][1],
            self.model["kt_arr"][1],
            self.model["zo_arr"][1],
            self.model["a_arr"][1],
            self.model["km_arr"][1],
            self.model["mo"][1],
            (1.9, 4.25)
        )
        # self.sbc_pars = PriorParams(
        #     1,
        #     self.refcategs[1],
        #     self.model["fo_arr"][1],
        #     self.model["kt_arr"][1],
        #     self.model["zo_arr"][1],
        #     self.model["a_arr"][1],
        #     self.model["km_arr"][1],
        #     (3.19, 4.25)
        # )
        # self.scd_pars = PriorParams(
        #     2,
        #     self.refcategs[2],
        #     self.model["fo_arr"][2],
        #     self.model["kt_arr"][2],
        #     self.model["zo_arr"][2],
        #     self.model["a_arr"][2],
        #     self.model["km_arr"][2],
        #     (1.9, 3.19)
        # )
        self.irr_pars = PriorParams(
            2,
            self.refcategs[2],
            self.model["fo_arr"][2],
            self.model["kt_arr"][2],
            self.model["zo_arr"][2],
            self.model["a_arr"][2],
            self.model["km_arr"][2],
            self.model["mo"][2],
            (-jnp.inf, 1.9)
        )
        
        self.add_data("model", self.model)
        self.add_handle("templates", data=self.templates_df, path=self.config.output)


    def inform(self, training_data):
        """The main interface method for Informers

        This will attach the input_data to this `Informer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the model that it creates to this Estimator
        by using `self.add_data('model', model)`.

        Finally, this will return a ModelHandle providing access to the trained model.

        Parameters
        ----------
        input_data : `dict` or `TableHandle`
            dictionary of all input data, or a `TableHandle` providing access to it

        Returns
        -------
        model : ModelHandle
            Handle providing access to trained model
        templates : TableHandle
            Handle providing access to selected templates as SPS parameters
        """
        self.set_data("input", training_data)
        self.run()
        self.finalize()
        return self.get_handle("model"), self.get_handle("templates")

    def plot_colrs_templates(self, hue='CAT_NUVK', style='Dataset', order=None):
        if order is None and hue=='CAT_NUVK':
            order=self.refcategs

        self._load_training()
        all_tsels_df = self._nuvk_classif()
        train_df = pd.DataFrame(
            data=jnp.column_stack(
                (self.mags[:, :-1]-self.mags[:, 1:], self.refmags, self.szs)
            ),
            columns=self.color_names+[self.config.ref_band, self.config.redshift_col]
        )

        leg1 = mlines.Line2D([], [], color='gray', label='LSST sim', marker='o', markersize=6, alpha=0.7, ls='')
        fig_list = []
        for ix, (c1, c2) in enumerate(zip(self.color_names[:-1], self.color_names[1:])):
            f,a = plt.subplots(1,1, constrained_layout=True)
            # Create a legend for the first line.
            
            sns.scatterplot(
                data=train_df,
                x=c1,
                y=c2,
                c='gray',
                size='redshift',
                sizes=(10, 100),
                ax=a,
                legend=False,
                alpha=0.2
            )
            
            sns.scatterplot(
                data=all_tsels_df,
                x=c1,
                y=c2,
                ax=a,
                size='z_p',
                sizes=(10, 100),
                alpha=0.5,
                hue=hue,
                hue_order=order,
                style=style,
                legend='brief'
            )
            a.grid()

            handles, labels = a.get_legend_handles_labels()
            a.legend(handles=[handles[0]]+[leg1]+handles, labels=['Training set']+['LSST sim']+labels)
            fig_list.append(f)
            plt.show()
        return fig_list

    def hist_colrs_templates(self, hue='Dataset'):
        self._load_training()
        all_tsels_df = self._nuvk_classif()
        train_df = pd.DataFrame(
            data=jnp.column_stack(
                (self.mags[:, :-1]-self.mags[:, 1:], self.refmags, self.szs)
            ),
            columns=self.color_names+[self.config.ref_band, self.config.redshift_col]
        )

        train_patch = mpatches.Patch(edgecolor='k', facecolor='grey', label='LSST sim', alpha=0.7)

        list_edges = []
        fig_list = []
        for idc, c in enumerate(self.color_names):
            _arr = np.array(train_df[c])
            H_data_1D, _edges1d = np.histogram(_arr[np.isfinite(_arr)], bins=self.config.colrsbins)
            H_templ_1d, _edges1d = np.histogram(np.array(all_tsels_df[c]), bins=_edges1d) 
            list_edges.append(_edges1d)
            
            f,a = plt.subplots(1,1)

            sns.histplot(
                data=train_df,
                x=c,
                bins=_edges1d,
                stat='density',
                label='Training data',
                color='grey',
                ax=a,
                legend=False
            )

            sns.histplot(
                data=all_tsels_df,
                x=c,
                bins=_edges1d,
                stat='density',
                multiple='stack',
                hue=hue,
                alpha=0.7,
                ax=a,
                legend=True
            )

            old_legend = a.get_legend()
            handles = old_legend.legend_handles
            labels = [t.get_text()+' templates' for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()
            
            a.legend(handles=[train_patch]+handles, labels=['Training data']+labels, title=title, loc='best')
            fig_list.append(f)
            
            plt.show()
        return fig_list


    def plot_sfh_templates(self):
        self._load_training()

        from .template import vmap_mean_sfr, T_ARR
        srcs = np.unique(self.templates_df['Dataset'].values)
        fcolors = plt.cm.rainbow(np.linspace(0, 1, len(srcs)))
        pars_arr = jnp.array(self.templates_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
        cdict = dict(zip(srcs, fcolors, strict=True))
        all_sfh = vmap_mean_sfr(pars_arr)
        _min, _max = all_sfh.max()/1e6, all_sfh.max()*1.1
        f, a = plt.subplots(1,1)
        for sfh, src in zip(all_sfh, self.templates_df['Dataset'], strict=True):
            a.plot(T_ARR, sfh, lw=1, ls='-', c=cdict[src])
            a.set_xlabel('Age of the Universe [Gyr]')
            a.set_ylabel('SFR '+r"$\mathrm{M_\odot.yr}^{-1}$")
            a.set_title('SFH of photo-z templates')
            a.set_ylim(_min, _max)
            a.set_yscale('log')

        legs = []
        for src, colr in cdict.items():
            _line = mlines.Line2D([], [], color=colr, label=src, lw=1)
            legs.append(_line)
        a.legend(handles=legs)
        plt.show()
        return f


    def _nuvk_classif(self):
        _mod_names = self.refcategs
        self._load_training()
        all_tsels_df = self._load_templates()
        all_tsels_df['CAT_NUVK'] = np.array( _mod_names[ _n] for _n in self.prior_mod(jnp.array(all_tsels_df['NUVK'].values)) )
        self.nt_array = np.array([ np.count_nonzero(all_tsels_df['CAT_NUVK'] == _typ) for _typ in _mod_names ])
        #self.nt_array = np.array([ np.count_nonzero(all_tsels_df['CAT_NUVK'] == _typ)//self.pzs.shape[0]  for _typ in _mod_names ])
        return all_tsels_df


    def _bpt_classif(self):
        all_tsels_df = self._nuvk_classif()
        all_tsels_df["log([OIII]/[Hb])"] = np.where(
            np.logical_and(all_tsels_df["AGN_[OIII]_5008.24_REW"] > 0.0, all_tsels_df["Balmer_HI_4862.68_REW"] > 0.0),
            np.log10(all_tsels_df["AGN_[OIII]_5008.24_REW"] / all_tsels_df["Balmer_HI_4862.68_REW"]),
            np.nan
        )

        all_tsels_df["log([NII]/[Ha])"] = np.where(
            np.logical_and(all_tsels_df["AGN_[NII]_6585.27_REW"] > 0.0, all_tsels_df["Balmer_HI_6564.61_REW"] > 0.0),
            np.log10(all_tsels_df["AGN_[NII]_6585.27_REW"] / all_tsels_df["Balmer_HI_6564.61_REW"]),
            np.nan
        )

        all_tsels_df["log([SII]/[Ha])"] = np.where(
            np.logical_and(all_tsels_df["AGN_[SII]_6718.29_REW"] > 0.0, all_tsels_df["Balmer_HI_6564.61_REW"] > 0.0),
            np.log10(all_tsels_df["AGN_[SII]_6718.29_REW"] / all_tsels_df["Balmer_HI_6564.61_REW"]),
            np.nan
        )

        all_tsels_df["log([OI]/[Ha])"] = np.where(
            np.logical_and(all_tsels_df["SF_[OI]_6302.046_REW"] > 0.0, all_tsels_df["Balmer_HI_6564.61_REW"] > 0.0),
            np.log10(all_tsels_df["SF_[OI]_6302.046_REW"] / all_tsels_df["Balmer_HI_6564.61_REW"]),
            np.nan
        )

        all_tsels_df["log([OIII]/[OII])"] = np.where(
            np.logical_and(all_tsels_df["AGN_[OIII]_5008.24_REW"] > 0.0, all_tsels_df["SF_[OII]_3728.48_REW"] > 0),
            np.log10(all_tsels_df["AGN_[OIII]_5008.24_REW"] / all_tsels_df["SF_[OII]_3728.48_REW"]),
            np.nan
        )

        cat_nii = []
        for x, y in zip(all_tsels_df["log([NII]/[Ha])"], all_tsels_df["log([OIII]/[Hb])"], strict=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                cat_nii.append("NC")
            elif y < Ka03_nii(x):
                cat_nii.append("Star-forming")
            elif y < Ke01_nii(x):
                cat_nii.append("Composite")
            else:
                cat_nii.append("AGN")

        all_tsels_df["CAT_NII"] = np.array(cat_nii)

        cat_sii = []
        for x, y in zip(all_tsels_df["log([SII]/[Ha])"], all_tsels_df["log([OIII]/[Hb])"], strict=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                cat_sii.append("NC")
            elif y < Ke01_sii(x):
                cat_sii.append("Star-forming")
            elif y < Ke06_sii(x):
                cat_sii.append("LINER")
            else:
                cat_sii.append("Seyferts")

        all_tsels_df["CAT_SII"] = np.array(cat_sii)

        cat_oi = []
        for x, y in zip(all_tsels_df["log([OI]/[Ha])"], all_tsels_df["log([OIII]/[Hb])"], strict=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                cat_oi.append("NC")
            elif y < Ke01_oi(x):
                cat_oi.append("Star-forming")
            elif y < Ke06_oi(x):
                cat_oi.append("LINER")
            else:
                cat_oi.append("Seyferts")

        all_tsels_df["CAT_OI"] = np.array(cat_oi)

        cat_oii = []
        for x, y in zip(all_tsels_df["log([OI]/[Ha])"], all_tsels_df["log([OIII]/[OII])"], strict=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                cat_oii.append("NC")
            elif y < lim_HII_comp(x):
                cat_oii.append("SF / composite")
            elif y < lim_seyf_liner(x):
                cat_oii.append("LINER")
            else:
                cat_oii.append("Seyferts")

        all_tsels_df["CAT_OIII/OIIvsOI"] = np.array(cat_oii)

        return all_tsels_df


    def plot_bpt_templates(self):
        all_tsels_df = self._bpt_classif()
        cat_x_y = [
            ("CAT_NII", "log([NII]/[Ha])", "log([OIII]/[Hb])"),
            ("CAT_SII", "log([SII]/[Ha])", "log([OIII]/[Hb])"),
            ("CAT_OI", "log([OI]/[Ha])", "log([OIII]/[Hb])"),
            ("CAT_OIII/OIIvsOI", "log([OI]/[Ha])", "log([OIII]/[OII])")
        ]
        fig_list = []
        for cat, x, y in cat_x_y:
            if np.any(np.isfinite(all_tsels_df[x])) and np.any(np.isfinite(all_tsels_df[y])):
                f, a = plt.subplots(1, 1)
                sns.scatterplot(
                    data=all_tsels_df,
                    x=x,
                    y=y,
                    hue=cat,
                    size='z_p',
                    sizes=(10, 100),
                    alpha=0.5,
                    ax=a
                )

                _x = np.linspace(np.nanmin(all_tsels_df[x]), np.nanmax(all_tsels_df[x]), 100, endpoint=True)
                if "NII" in cat:
                    a.plot(_x, Ka03_nii(_x), 'k-', lw=1)
                    a.plot(_x, Ke01_nii(_x), 'k-', lw=1)
                    a.set_xlim(np.nanmin(all_tsels_df[x]), 0.0)
                elif "SII" in cat:
                    a.plot(_x, Ke01_sii(_x), 'k-', lw=1)
                    a.plot(_x, Ke06_sii(_x), 'k-', lw=1)
                elif "OII" in cat:
                    a.plot(_x, lim_HII_comp(_x), 'k-', lw=1)
                    a.plot(_x, lim_seyf_liner(_x), 'k-', lw=1)
                else:
                    a.plot(_x, Ke01_oi(_x), 'k-', lw=1)
                    a.plot(_x, Ke06_oi(_x), 'k-', lw=1)
                
                #a.set_ylim(np.nanmin(all_tsels_df[y]), np.nanmax(all_tsels_df[y]))
                fig_list.append(f)
                plt.show()
        return fig_list

    def plot_templ_seds(self, redshifts=None):
        if redshifts is None:
            redshifts = jnp.linspace(self.config.zmin, self.config.zmax, 6, endpoint=True)
        elif isinstance(redshifts, (int, float, jnp.float32, jnp.float64, np.float32, np.float64)):
            redshifts = jnp.array([redshifts])
        elif isinstance(redshifts, (list, tuple, np.ndarray)):
            redshifts = jnp.array(redshifts)
        else:
            assert isinstance(redshifts, jnp.ndarray), "Please specify the redshift as a single value or a list, tuple, numpy array or jax array of values."
        wls, transm_arr = self._load_filters()
        templ_pars = jnp.array(self.templates_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
        templ_zref = jnp.array(self.templates_df[self.config.redshift_col])
        sspdata = load_ssp(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SSP",
                    self.config.ssp_file
                )
            )
        )
        if "sps" in self.config.templ_type.lower():
            restframe_fnus = lsunPerHz_to_fnu_noU(
                vmap_mean_spectrum_nodust(wls, templ_pars, redshifts, sspdata),
                0.001
            )
            nuvk = v_nuvk(templ_pars, wls, redshifts, sspdata)
            _selnorm = jnp.logical_and(wls>3950, wls<4000)
            norms = jnp.nanmean(restframe_fnus[:, :, _selnorm], axis=2)
            restframe_fnus = restframe_fnus/jnp.expand_dims(jnp.squeeze(norms), 2)
        else:
            _vspec = vmap(mean_spectrum_nodust, in_axes=(None, 0, 0, None))
            _vnuvk = vmap(calc_nuvk, in_axes=(0, None, 0, None))
            restframe_fnus = lsunPerHz_to_fnu_noU(
                _vspec(wls, templ_pars, templ_zref, sspdata),
                0.001
            )
            nuvk = _vnuvk(templ_pars, wls, templ_zref, sspdata)
            _selnorm = jnp.logical_and(wls>3950, wls<4000)
            norms = jnp.nanmean(restframe_fnus[:, _selnorm], axis=1)
            restframe_fnus = restframe_fnus/jnp.expand_dims(jnp.squeeze(norms), 1)
        
        colrs = mpl.colormaps['tab10'].colors
        clrdict = {_categ: colrs[icat] for icat, _categ in enumerate(self.refcategs)}

        filtcols = plt.cm.rainbow(np.linspace(0, 1, transm_arr.shape[0]))
        figlist = []
        for iz, z in enumerate(redshifts):
            f, a = plt.subplots(1,1, figsize=(7, 4), constrained_layout=True)
            if "sps" in self.config.templ_type.lower():
                for fnu, _nuvk in zip(restframe_fnus[:, iz, :], nuvk[:, iz], strict=True):
                    _n = self.prior_mod(_nuvk) 
                    classnuvk = self.refcategs[ _n]
                    a.plot(*convert_flux_toobsframe(wls, fnu, z), c=clrdict[classnuvk])
            else:
                for fnu, _nuvk in zip(restframe_fnus, nuvk, strict=True):
                    _n = self.prior_mod(_nuvk) 
                    classnuvk = self.refcategs[ _n]
                    a.plot(*convert_flux_toobsframe(wls, fnu, z), c=clrdict[classnuvk])
            a.set_xlabel(r'Observed wavelength $\mathrm{[\AA]}$')
            a.set_ylabel(r'Normalized Spectral Energy Density [-]') #$\mathrm{[erg.s^{-1}.cm^{-2}.Hz^{-1}]}$')
            aa = a.twinx()
            for trans, fcol in zip(transm_arr, filtcols, strict=True):
                aa.plot(wls, trans, c=tuple(fcol), lw=1)
                aa.fill_between(wls, trans, alpha=0.3, color=tuple(fcol), lw=1)
            aa.set_ylabel(r'Filter transmission / effective area [- / $\mathrm{m^2}$]')
            #a.set_xscale('log')
            a.set_yscale('log')
            a.set_xlim(self.config.wlmin, self.config.wlmax+self.config.dwl)
            a.grid()
            _legends = []
            for _cat, _colrs in clrdict.items():
                _line = mlines.Line2D([], [], color=_colrs, label=_cat)
                _legends.append(_line)
            a.legend(handles=_legends)
            a.set_title(r'SED templates at $z=$'+f"{z:.2f}")
            secax = a.secondary_xaxis('top', functions=(lambda wl: wl/(1+z), lambda wl: wl*(1+z)))
            secax.set_xlabel(r'Resframe wavelength $\mathrm{[\AA]}$')
            figlist.append(f)
            plt.show()
        return figlist

    def plot_templ_seds_d4000(self, redshifts=None):
        if redshifts is None:
            redshifts = jnp.linspace(self.config.zmin, self.config.zmax, 6, endpoint=True)
        elif isinstance(redshifts, (int, float, jnp.float32, jnp.float64, np.float32, np.float64)):
            redshifts = jnp.array([redshifts])
        elif isinstance(redshifts, (list, tuple, np.ndarray)):
            redshifts = jnp.array(redshifts)
        else:
            assert isinstance(redshifts, jnp.ndarray), "Please specify the redshift as a single value or a list, tuple, numpy array or jax array of values."
        wls, transm_arr = self._load_filters()
        templ_pars = jnp.array(self.templates_df[_DUMMY_PARS.PARAM_NAMES_FLAT])
        templ_zref = jnp.array(self.templates_df[self.config.redshift_col])
        sspdata = load_ssp(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SSP",
                    self.config.ssp_file
                )
            )
        )
        if "sps" in self.config.templ_type.lower():
            restframe_fnus = lsunPerHz_to_fnu_noU(
                vmap_mean_spectrum_nodust(wls, templ_pars, redshifts, sspdata),
                0.001
            )
            d4000n = v_d4000n(templ_pars, wls, redshifts, sspdata)
            _selnorm = jnp.logical_and(wls>3950, wls<4000)
            norms = jnp.nanmean(restframe_fnus[:, :, _selnorm], axis=2)
            restframe_fnus = restframe_fnus/jnp.expand_dims(jnp.squeeze(norms), 2)
        else:
            _vspec = vmap(mean_spectrum_nodust, in_axes=(None, 0, 0, None))
            _vd4k = vmap(calc_d4000n, in_axes=(0, None, 0, None))
            restframe_fnus = lsunPerHz_to_fnu_noU(
                _vspec(wls, templ_pars, templ_zref, sspdata),
                0.001
            )
            d4000n = _vd4k(templ_pars, wls, templ_zref, sspdata)
            _selnorm = jnp.logical_and(wls>3950, wls<4000)
            norms = jnp.nanmean(restframe_fnus[:, _selnorm], axis=1)
            restframe_fnus = restframe_fnus/jnp.expand_dims(jnp.squeeze(norms), 1)
        rbmap = mpl.colormaps['coolwarm']
        cNorm = mpl.colors.Normalize(vmin=d4000n.min(), vmax=d4000n.max())
        d4map = mpl.cm.ScalarMappable(norm=cNorm, cmap=rbmap)
        d4cols = d4map.to_rgba(d4000n)
        print(restframe_fnus.shape, d4000n.shape, wls.shape, d4cols.shape)
        filtcols = plt.cm.rainbow(np.linspace(0, 1, transm_arr.shape[0]))
        figlist = []
        for iz, z in enumerate(redshifts):
            f, a = plt.subplots(1,1, figsize=(7, 4), constrained_layout=True)
            if "sps" in self.config.templ_type.lower():
                for fnu, col in zip(restframe_fnus[:, iz, :], d4cols[:, iz, :], strict=True):
                    a.plot(*convert_flux_toobsframe(wls, fnu, z), c=tuple(col))
            else:
                for fnu, col in zip(restframe_fnus, d4cols, strict=True):
                    a.plot(*convert_flux_toobsframe(wls, fnu, z), c=tuple(col))
            plt.colorbar(d4map, ax=a, label='D4000')
            a.set_xlabel(r'Observed wavelength $\mathrm{[\AA]}$')
            a.set_ylabel(r'Normalized Spectral Energy Density [-]') #$\mathrm{[erg.s^{-1}.cm^{-2}.Hz^{-1}]}$')
            aa = a.twinx()
            for trans, fcol in zip(transm_arr, filtcols, strict=True):
                aa.plot(wls, trans, c=tuple(fcol), lw=1)
                aa.fill_between(wls, trans, alpha=0.3, color=tuple(fcol), lw=1)
            aa.set_ylabel(r'Filter transmission / effective area [- / $\mathrm{m^2}$]')
            #a.set_xscale('log')
            a.set_yscale('log')
            a.set_xlim(self.config.wlmin, self.config.wlmax+self.config.dwl)
            a.grid()
            a.set_title(r'SED templates at $z=$'+f"{z:.2f}")
            secax = a.secondary_xaxis('top', functions=(lambda wl: wl/(1+z), lambda wl: wl*(1+z)))
            secax.set_xlabel(r'Resframe wavelength $\mathrm{[\AA]}$')
            figlist.append(f)
            plt.show()
        return figlist

    def plot_line_sed(self, templ_id, redshift=None):
        figlist = []
        try:
            subdf = self.templates_df.loc[templ_id]
        except KeyError:
            if isinstance(templ_id, int):
                try:
                    subdf = self.templates_df.iloc[templ_id]
                except IndexError:
                    print("Specified index not found in the templates dataframe.")
            else:
                print("Specified key not found in the templates dataframe's index.")
        pars = jnp.array(subdf[_DUMMY_PARS.PARAM_NAMES_FLAT].values, dtype=jnp.float64)
        zref = subdf[self.config.redshift_col] #.iloc[0, self.config.redshift_col]
        z = zref if redshift is None else redshift
        lines = jnp.array([3728.48, 4862.68, 5008.24, 6302.046, 6564.61, 6585.27, 6718.29])
        lines_names = [
            "SF_[OII]_3728.48_REW",
            "Balmer_HI_4862.68_REW",
            "AGN_[OIII]_5008.24_REW",
            "SF_[OI]_6302.046_REW",
            "Balmer_HI_6564.61_REW",
            "AGN_[NII]_6585.27_REW",
            "AGN_[SII]_6718.29_REW"
        ]
        line_wids = lines * 400 / C_KMS / 2
        cont_wids = lines * 15000 / C_KMS / 2

        sspdata = load_ssp(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SSP",
                    self.config.ssp_file
                )
            )
        )

        wls = jnp.arange(3500., 7000., 0.1)
        sed = mean_spectrum_nodust(wls, pars, z, sspdata) if "sps" in self.config.templ_type.lower() else mean_spectrum_nodust(wls, pars, zref, sspdata)
        eqws = vmap_calc_eqw(wls, sed, lines)
        fnu = lsunPerHz_to_fnu_noU(sed, 0.001)

        for il, lin in enumerate(lines):
            f, a = plt.subplots(1,1)
            sel = jnp.logical_and(wls>=lin-1.5*cont_wids[il], wls<=lin+1.5*cont_wids[il])
            a.plot(wls[sel], fnu[sel], ls='-', color='k', label=subdf['name'])
            a.axvline(lin-cont_wids[il], ls=':', color='orange', label="Continuum bounds")
            a.axvline(lin+cont_wids[il], ls=':', color='orange')
            a.axvline(lin-line_wids[il], ls=':', color='r', label="Line bounds")
            a.axvline(lin+line_wids[il], ls=':', color='r')
            a.axvline(lin, ls='-', color='g', label=lines_names[il])
            a.fill_between(wls[sel], fnu[sel], where=np.logical_and(wls[sel]>lin-0.5*eqws[il], wls[sel]<lin+0.5*eqws[il]), color='g', alpha=0.5, label=r"REW $=$"+f"{eqws[il]:.2f}"+r"$\mathrm{\AA}$")
            a.set_xlabel(r'Restframe wavelength $\mathrm{[\AA]}$')
            a.set_ylabel(r'*Spectral Energy Density $\mathrm{[erg.s^{-1}.cm^{-2}.Hz^{-1}]}$')
            a.legend()
            a.set_title(f"Restframe Equivalent Width of {lines_names[il]} for template {subdf['name']} at "+r"$z=$"+f"{z:.2f}")
            plt.show()
            figlist.append(f)
        
        return figlist
