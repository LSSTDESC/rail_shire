import os
import jax
from functools import partial
import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map
from jax import jit, vmap
import qp
#import pandas as pd
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatEstimator
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import TableHandle, ModelHandle

from .analysis import _DUMMY_PARS, PARS_DF, vmap_mean, vmap_median
from .galaxy import vmap_mags_to_i_and_colors, likelihood, vmap_neg_log_likelihood
from .filter import get_sedpy
from .template import make_legacy_templates, make_sps_templates, istuple
from .io_utils import load_ssp, SHIREDATALOC
from .inform import PriorParams, nz_func, frac_func

try:
    from jax.numpy import trapezoid
except ImportError:
    try:
        from jax.scipy.integrate import trapezoid
    except ImportError:
        from jax.numpy import trapz as trapezoid

jax.config.update("jax_enable_x64", True)

class ShireEstimator(CatEstimator):
    """CatEstimator subclass to implement basic marginalized PDF for BPZ
    In addition to the marginalized redshift PDF, we also compute several
    ancillary quantities that will be stored in the ensemble ancil data:
    zmode: mode of the PDF
    amean: mean of the PDF
    tb: integer specifying the best-fit SED *at the redshift mode*
    todds: fraction of marginalized posterior prob. of best template,
    so lower numbers mean other templates could be better fits, likely
    at other redshifts
    """
    name = "ShireEstimator"
    inputs = [("model", ModelHandle), ("input", TableHandle), ("templates", TableHandle)]
    config_options = CatEstimator.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,
        bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        err_bands=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        chunk_size=5000,
        avbins=Param(
            int,
            6,
            msg="Number of Av values for which to compute the dust attenuation for each template."
        ),
        unobserved_val=Param(float, -99.0, msg="value to be replaced with zero flux and given large errors for non-observed filters"),
        data_path=Param(
            str,
            "None",
            msg="data_path (str): file path to the "
            "SED, FILTER, and AB directories.  If left to "
            "default `None` it will use the install "
            "directory for rail + ../examples_data/estimation_data/data"),
        templ_type=Param(
            str,
            "SPS",
            msg='Whether to use the "SPS" or "Legacy" method to derive the templates colours from the SPS parameters.'
        ),
        no_prior=Param(bool, True, msg="set to True if you want to run without prior"),
        mag_err_min=Param(
            float,
            0.005,
            msg="a minimum floor for the magnitude errors to prevent a "
            "large chi^2 for very very bright objects"
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
            25000.,
            msg='wlmax (float): upper bound of wavelength grid for filters interpolation'
        ),
        dwl=Param(
            float,
            100.,
            msg='dwl (float): step of wavelength grid for filters interpolation'
        ),
        prior_type=Param(str, "NUVK", msg="Quantity to classify galaxies in broad types; must be one of 'NUVK' or 'BPT'.")
    )

    def __init__(self, args, **kwargs):
        """Constructor, build the CatEstimator, then do BPZ specific setup
        """
        super().__init__(args, **kwargs)

        datapath = self.config["data_path"]
        if datapath is None or datapath == "None":
            self.data_path = SHIREDATALOC
        else:  # pragma: no cover
            self.data_path = datapath
            os.environ["SHIREDATALOC"] = self.data_path
        if not os.path.exists(self.data_path):  # pragma: no cover
            raise FileNotFoundError("SHIREDATALOC " + self.data_path + " does not exist! Check value of data_path in config file!")

        # check on bands, errs, and prior band
        if len(self.config.bands) != len(self.config.err_bands):  # pragma: no cover
            raise ValueError("Number of bands specified in bands must be equal to number of mag errors specified in err_bands!")
        if self.config.ref_band not in self.config.bands:  # pragma: no cover
            raise ValueError(f"reference band not found in bands specified in bands: {str(self.config.bands)}")
        if len(self.config.bands) != len(self.config.err_bands) or len(self.config.bands) != len(self.config.filter_dict):
            raise ValueError("length of bands, err_bands, and filter_list are not the same!")

        self.modeldict=None
        self.colrs_templates=None
        self.zgrid=None
        self.avgrid=None
        self.templates=None
        self.prior_type = self.config.prior_type
        self.refcategs = np.array(["E_S0", "Sbc/Scd", "Irr"]) if "nuvk" in self.prior_type.lower() else np.array(["Star-forming", "AGN", "Composite", "NC"])
        self.ntyp = len(self.refcategs)
        self.e0_pars = None
        self.sbcd_pars = None
        self.irr_pars = None
        self.sf_pars = None
        self.agn_pars = None
        self.com_pars = None
        self.nc_pars = None


    def _initialize_run(self):
        super()._initialize_run()

        # If we are not the root process then we wait for
        # the root to (potentially) create all the templates before
        # reading them ourselves.
        if self.rank > 0:  # pragma: no cover
            # The Barrier method causes all processes to stop
            # until all the others have also reached the barrier.
            # If our rank is > 0 then we must be running under MPI.
            self.comm.Barrier()
            self.colrs_templates = self._load_templates()
        # But if we are the root process then we just go
        # ahead and load them before getting to the Barrier,
        # which will allow the other processes to continue
        else:
            self.colrs_templates = self._load_templates()
            # We might only be running in serial, so check.
            # If we are running MPI, then now we have created
            # the templates we let all the other processes that
            # stopped at the Barrier above continue and read them.
            if self.is_mpi():  # pragma: no cover
                self.comm.Barrier()

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        self.modeldict = self.model
        if "nuvk" in self.prior_type.lower():
            self.e0_pars = PriorParams(
                0,
                self.refcategs[0],
                self.modeldict["fo_arr"][0],
                self.modeldict["kt_arr"][0],
                self.modeldict["zo_arr"][0],
                self.modeldict["a_arr"][0],
                self.modeldict["km_arr"][0],
                self.modeldict["mo"][0],
                self.modeldict["nt_array"][0],
                (4.25, jnp.inf)
            )
            self.sbcd_pars = PriorParams(
                1,
                self.refcategs[1],
                self.modeldict["fo_arr"][1],
                self.modeldict["kt_arr"][1],
                self.modeldict["zo_arr"][1],
                self.modeldict["a_arr"][1],
                self.modeldict["km_arr"][1],
                self.modeldict["mo"][1],
                self.modeldict["nt_array"][1],
                (1.9, 4.25)
            )
            # self.sbc_pars = PriorParams(
            #     1,
            #     "Sbc",
            #     self.modeldict["fo_arr"][1],
            #     self.modeldict["kt_arr"][1],
            #     self.modeldict["zo_arr"][1],
            #     self.modeldict["a_arr"][1],
            #     self.modeldict["km_arr"][1],
            #     (3.19, 4.25)
            # )
            # self.scd_pars = PriorParams(
            #     2,
            #     "Scd",
            #     self.modeldict["fo_arr"][2],
            #     self.modeldict["kt_arr"][2],
            #     self.modeldict["zo_arr"][2],
            #     self.modeldict["a_arr"][2],
            #     self.modeldict["km_arr"][2],
            #     (1.9, 3.19)
            # )
            self.irr_pars = PriorParams(
                2,
                self.refcategs[2],
                1-(self.modeldict["fo_arr"][0]+self.modeldict["fo_arr"][1]),
                -99,
                self.modeldict["zo_arr"][2],
                self.modeldict["a_arr"][2],
                self.modeldict["km_arr"][2],
                self.modeldict["mo"][2],
                self.modeldict["nt_array"][2],
                (-jnp.inf, 1.9)
            )
        else:
            self.sf_pars = PriorParams(
                0,
                self.refcategs[0],
                self.modeldict["fo_arr"][0],
                self.modeldict["kt_arr"][0],
                self.modeldict["zo_arr"][0],
                self.modeldict["a_arr"][0],
                self.modeldict["km_arr"][0],
                self.modeldict["mo"][0],
                self.modeldict["nt_array"][0],
                None
            )
            self.agn_pars = PriorParams(
                1,
                self.refcategs[1],
                self.modeldict["fo_arr"][1],
                self.modeldict["kt_arr"][1],
                self.modeldict["zo_arr"][1],
                self.modeldict["a_arr"][1],
                self.modeldict["km_arr"][1],
                self.modeldict["mo"][1],
                self.modeldict["nt_array"][1],
                None
            )
            self.com_pars = PriorParams(
                2,
                self.refcategs[2],
                self.modeldict["fo_arr"][2],
                self.modeldict["kt_arr"][2],
                self.modeldict["zo_arr"][2],
                self.modeldict["a_arr"][2],
                self.modeldict["km_arr"][2],
                self.modeldict["mo"][2],
                self.modeldict["nt_array"][2],
                None
            )
            self.nc_pars = PriorParams(
                3,
                self.refcategs[3],
                1-(self.modeldict["fo_arr"][0]+self.modeldict["fo_arr"][1]+self.modeldict["fo_arr"][2]),
                -99,
                self.modeldict["zo_arr"][3],
                self.modeldict["a_arr"][3],
                self.modeldict["km_arr"][3],
                self.modeldict["mo"][3],
                self.modeldict["nt_array"][3],
                None
            )


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
            self.templates = None
            return self.templates
        if isinstance(templates, str):
            self.templates = self.set_data("templates", data=None, path=templates)
            self.config["templates"] = templates
            return self.templates
        if isinstance(templates, TableHandle):
            if templates.has_path:
                self.config["templates"] = templates.path
        self.templates = self.set_data("templates", templates)
        return self.templates


    def _load_filters(self):
        wls = jnp.arange(
            self.config.wlmin,
            self.config.wlmax+self.config.dwl,
            self.config.dwl
        )

        transm_arr = get_sedpy(self.config.filter_dict, wls, self.data_path)

        return wls, transm_arr

    def _load_templates(self):

        # The redshift range we will evaluate on
        self.zgrid = jnp.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        z = self.zgrid
        
        self.avgrid = jnp.linspace(
            PARS_DF.loc["AV", "MIN"],
            PARS_DF.loc["AV", "MAX"],
            num=self.config.avbins,
            endpoint=True
        )
        av_arr = self.avgrid

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
        templs_df = self.open_templates(**self.config)
        templ_pars_arr = jnp.array(templs_df[_DUMMY_PARS.PARAM_NAMES_FLAT])

        zref_arr = jnp.array(templs_df[self.config.redshift_col])
        if "sps" in self.config.templ_type.lower():
            print(f"Building templates with Stellar Population Synthesis... [templ_type={self.config.templ_type}]")
            tcolors = make_sps_templates(
                templ_pars_arr,
                fwls,
                ftransm,
                z,
                av_arr,
                sspdata
            )
        else:
            print(f"Building templates from a rest-frame SED... [templ_type={self.config.templ_type}]")
            tcolors = make_legacy_templates(
                templ_pars_arr,
                zref_arr,
                fwls,
                ftransm,
                z,
                av_arr,
                sspdata
            )

        return tcolors

    def _preprocess_magnitudes(self, data):

        # replace non-detects with NaN and mag_err with lim_mag for consistency
        # with typical BPZ performance
        for bandname, errname in zip(self.config.bands, self.config.err_bands, strict=True):
            _dat, _err = jnp.array(data[bandname]), jnp.array(data[errname])
            if jnp.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = jnp.isnan(_dat)
            else:
                detmask = jnp.isclose(_dat, self.config.nondetect_val)
            # if isinstance(data, pd.DataFrame):
            #     data.loc[detmask, bandname] = jnp.nan
            #     data.loc[detmask, errname] = self.config.mag_limits[bandname]
            # else:
            #     data[bandname][detmask] = jnp.nan
            #     data[errname][detmask] = self.config.mag_limits[bandname]
            data[bandname] = jnp.where(detmask, jnp.nan, _dat)
            data[errname] = jnp.where(detmask, jnp.nan, _err)

        # replace non-observations with NaN, again to match BPZ standard
        # below the fluxes for these will be set to zero but with enormous
        # flux errors
        for bandname, errname in zip(self.config.bands, self.config.err_bands, strict=True):
            _dat, _err = jnp.array(data[bandname]), jnp.array(data[errname])
            if jnp.isnan(self.config.unobserved_val):  # pragma: no cover
                obsmask = jnp.isnan(_dat)
            else:
                obsmask = jnp.isclose(_dat, self.config.unobserved_val)
            # if isinstance(data, pd.DataFrame):
            #     data.loc[obsmask, bandname] = jnp.nan
            #     data.loc[obsmask, errname] = 20.0
            # else:
            #     data[bandname][obsmask] = jnp.nan
            #     data[errname][obsmask] = 20.0
            data[bandname] = jnp.where(obsmask, jnp.nan, _dat)
            data[errname] = jnp.where(obsmask, jnp.nan, _err)

        obs_mags = jnp.column_stack([data[_b] for _b in self.config.bands])
        obs_mags_errs = jnp.column_stack([data[_b] for _b in self.config.err_bands])
        
        # Clip to min mag errors.
        # JZ: Changed the max value here to 20 as values in the lensfit
        # catalog of ~ 200 were causing underflows below that turned into
        # zero errors on the fluxes and then nans in the output
        
        obs_mags_errs = jnp.clip(obs_mags_errs, self.config.mag_err_min, 20)
        
        id_i_band = self.config.bands.index(self.config.ref_band)
        i_mag_ab, ab_colors, ab_cols_errs = (
            vmap_mags_to_i_and_colors(obs_mags, obs_mags_errs, id_i_band)
        )
        return ab_colors, ab_cols_errs, i_mag_ab

    @partial(jit, static_argnums=0)
    def prior_z0(self, nuvk):
        """prior_z0 Determines the z0 value of the prior function

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: z0
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.z0,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.z0,
                self.sbcd_pars.z0
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_m0(self, nuvk):
        """prior_m0 Determines the m0 value of the prior function

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: m0
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.m0,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.m0,
                self.sbcd_pars.m0
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_alpha(self, nuvk):
        """prior_alpha Determines the alpha0 value in the prior function (power law)

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: alpha0
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.alpha,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.alpha,
                self.sbcd_pars.alpha
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_km(self, nuvk):
        """prior_km Determines the k value in the prior function

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: k
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.km,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.km,
                self.sbcd_pars.km
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_fo(self, nuvk):
        """prior_fo Determines the f0 value in the prior (fractions) function

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: fo
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.fo,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.fo,
                self.sbcd_pars.fo
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_kt(self, nuvk):
        """prior_kt Determines the kt value in the prior (fractions) function

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: kt
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.kt,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.kt,
                self.sbcd_pars.kt
            )
        )
        return val


    @partial(jit, static_argnums=0)
    def prior_mod(self, nuvk):
        """prior_mod Determines the model (galaxy morphology) for which to compute the prior value.

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: Model Id
        :rtype: int
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.mod,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.mod,
                self.sbcd_pars.mod
            )
        )
        return val.astype(int)


    @partial(jit, static_argnums=0)
    def prior_nt(self, nuvk):
        """prior_nt Determines the number of templates for which to compute the prior value.

        :param nuvk: Emitted UV-IR color index of the galaxy
        :type nuvk: float
        :return: Number of templates of a given broad type
        :rtype: float
        """
        val = jnp.where(
            nuvk >= self.e0_pars.nuv_range[0],
            self.e0_pars.nt,
            jnp.where(
                nuvk < self.irr_pars.nuv_range[1],
                self.irr_pars.nt,
                self.sbcd_pars.nt
            )
        )
        return val.astype(int)


    @partial(jit, static_argnums=0)
    def _val_nz_prior(self, oimag, z, nuvk):
        alpha, z0, km, m0 = self.prior_alpha(nuvk), self.prior_z0(nuvk), self.prior_km(nuvk), self.prior_m0(nuvk)
        val_prior = nz_func( (oimag, z), z0, alpha, km, m0 )
        return val_prior

    vmap_nz_gals = vmap(_val_nz_prior, in_axes=(None, 0, None, None))
    vmap_nz_nuvk = vmap(vmap_nz_gals, in_axes=(None, None, None, 0))
    vmap_nz_z = vmap(vmap_nz_nuvk, in_axes=(None, None, 0, 0))

    @partial(jit, static_argnums=0)
    def _val_frac_prior(self, oimag, nuvk):
        fo, kt, m0, nt = self.prior_fo(nuvk), self.prior_kt(nuvk), self.prior_m0(nuvk), self.prior_nt(nuvk)
        val_prior = jnp.where(nt>0, frac_func((fo, kt), m0, oimag)/nt, 0.0) # cover the case where one type is missing after training !
        return val_prior


    @partial(jit, static_argnums=0)
    def _val_frac_prior_alaBPZ(self, oimag, nuvk):
        fo, kt, m0, nt, mod = self.prior_fo(nuvk), self.prior_kt(nuvk), self.prior_m0(nuvk), self.prior_nt(nuvk), self.prior_mod(nuvk)
        _val_frac = frac_func((fo, kt), m0, oimag)
        _sum_to_one = 1-\
            (
                frac_func((self.e0_pars.fo, self.e0_pars.kt), self.e0_pars.m0, oimag) + \
                frac_func((self.sbcd_pars.fo, self.sbcd_pars.kt), self.sbcd_pars.m0, oimag)
            )
        _val_alaBPZ = jnp.where(mod<self.ntyp-1, _val_frac, _sum_to_one)
        val_frac = jnp.where(nt>0, _val_alaBPZ/nt, 0.0) # cover the case where one type is missing after training !
        return val_frac

    vmap_frac_gals = vmap(_val_frac_prior, in_axes=(None, 0, None))
    vmap_frac_nuvk = vmap(vmap_frac_gals, in_axes=(None, None, 0))


    @partial(jit, static_argnums=0)
    def _val_prior(self, oimag, z, nuvk):
        nzval = self._val_nz_prior(oimag, z, nuvk)
        fracval = self._val_frac_prior_alaBPZ(oimag, nuvk)
        return nzval*fracval


    vmap_prior_gals = vmap(_val_prior, in_axes=(None, 0, None, None))
    vmap_prior_nuvk = vmap(vmap_prior_gals, in_axes=(None, None, None, 0))
    _vmap_for_prior_norm = vmap(vmap_prior_nuvk, in_axes=(None, None, 0, None))
    
    
    vmap_prior_z = vmap(vmap_prior_nuvk, in_axes=(None, None, 0, 0))

    @partial(jit, static_argnums=0)
    def _prior(self, oimags, redz, nuvk):
        #corrmags = jnp.where(oimags<self.modeldict['mo'], self.modeldict['mo'], oimags)
        vals = self.vmap_prior_z(oimags, redz, nuvk)
        _vals_for_norm = self._vmap_for_prior_norm(oimags, redz, jnp.array([1.0, 3.0, 5.0]))
        _sums = jnp.nansum(_vals_for_norm, axis=1)
        #valmax = jnp.nanmax(vals, axis=1)
        norm = trapezoid(_sums, x=redz, axis=0)
        return vals/norm


    def _estimate_pdf(self, templ_tuples, observed_colors, observed_noise, observed_imags):

        if self.config.no_prior:
            probz_arr = tree_map(
                lambda sed_tupl: likelihood(sed_tupl[0], observed_colors, observed_noise),
                templ_tuples,
                is_leaf=istuple,
            )
        else:
            def _posterior(sedcols, ocols, onoise, oimags, redz, nuvks):
                _nllik = vmap_neg_log_likelihood(sedcols, ocols, onoise)
                _pz = jnp.exp(-0.5 * _nllik)
                #_n = trapezoid(_pz, x=self.zgrid, axis=0)
                _prior = self._prior(oimags, redz, nuvks)
                _vals = _pz*_prior #/_n
                return jnp.nanmax(_vals, axis=1)
                #return likelihood(sedcols, ocols, onoise) * self._prior(oimags, redz, nuvks[0][0]) # prior is computed for the template without dust
            
            probz_arr = tree_map(
                lambda sed_tupl: _posterior(sed_tupl[0], observed_colors, observed_noise, observed_imags, self.zgrid, sed_tupl[1]),
                templ_tuples,
                is_leaf=istuple,
            )

        probz_arr = jnp.array(probz_arr)
        _n2 = trapezoid(jnp.nansum(probz_arr, axis=0), x=self.zgrid, axis=0)
        probz_arr = probz_arr / _n2

        pdz_arr = jnp.nansum(probz_arr, axis=0)
        
        zpos = jnp.nanargmax(pdz_arr, axis=0)
        zmodes = self.zgrid[zpos]

        # Find T_B, the highest probability template *at zmode*
        #tmode = probz_arr[zpos, :, :]
        #t_b = jnp.nanargmax(tmode, axis=0)

        # compute TODDS, the fraction of probability of the "best" template
        # relative to the other templates
        #tmarg = probz_arr.sum(axis=0)
        #todds = tmarg[t_b] / jnp.sum(tmarg)

        return pdz_arr, zmodes


    def _process_chunk(self, start, end, data, first):
        test_colrs, test_colr_errs, test_m_0 = self._preprocess_magnitudes(data)
        
        pdfs, zmodes = self._estimate_pdf(
            self.colrs_templates,
            test_colrs,
            test_colr_errs,
            test_m_0
        )
        
        qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs.T))
        
        zmeans = vmap_mean(self.zgrid, pdfs)
        zmedians = vmap_median(self.zgrid, pdfs)
        
        qp_dstn.set_ancil(dict(zmode=zmodes, zmean=zmeans, zmedian=zmedians)) #, tb=tb, todds=todds))
        self._do_chunk_output(qp_dstn, start, end, first)
