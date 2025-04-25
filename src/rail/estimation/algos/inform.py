import os
import jax
import numpy as np
from jax import numpy as jnp
#from jax import vmap, jit
from jax.tree_util import tree_map
#from jax import random as jrn
import pandas as pd
#import qp
from tqdm import tqdm
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
#from rail.utils.path_utils import RAILDIR
from rail.core.data import TableHandle, ModelHandle
from rail.core.common_params import SHARED_PARAMS
from .io_utils import load_ssp, istuple, SHIREDATALOC
from .analysis import _DUMMY_PARS #, PARS_DF, PARAMS_MAX, PARAMS_MIN, INIT_PARAMS
from .template import vmap_cols_zo
from .filter import get_sedpy

jax.config.update("jax_enable_x64", True)

def nzfunc(z, z0, alpha, km, m, m0):  # pragma: no cover
    zm = z0 + (km * (m - m0))
    return np.power(z, alpha) * np.exp(-1. * np.power((z / zm), alpha))

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
        colrsbins=Param(
            int,
            40,
            msg='colrsbins (int): number of bins for each colour index in which to select the template with the best score.'
        )
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
        
        self.fo_arr = None
        self.kt_arr = None
        self.typmask = None
        self.ntyp = None
        self.mags = None
        self.szs = None
        self.besttypes = None
        self.m0 = None
        self.templates_df = None


    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        if self.config.ref_band not in training_data.keys():  # pragma: no cover
            raise KeyError(f"ref_band {self.config.ref_band} not found in input data!")
        if self.config.redshift_col not in training_data.keys():  # pragma: no cover
            raise KeyError(f"redshift column {self.config.redshift_col} not found in input data!")

        filters_names = [_fnam for _fnam, _fdir in self.config.filter_dict.items()]

        color_names = [f"{n1}-{n2}" for n1,n2 in zip(filters_names[:-1], filters_names[1:])]
        #color_err_names = [f"{n1}-{n2}_err" for n1,n2 in zip(filters_names[:-1], filters_names[1:])]

        #mobs_df = tables_io.read(
        #    self.datafile,
        #    tables_io.types.PD_DATAFRAME,
        #    fmt='hf5'
        #)

        self.mags = jnp.column_stack(
            [training_data[f"mag_{_n}"] for _n in filters_names]
        )

        self.szs = jnp.array(
            training_data[self.config["redshift_col"]]
        )

        pzs = np.histogram_bin_edges(self.szs, bins='auto')

        #key = jrn.key(717)
        #key, subkey = jrn.split(key)
        #train_sel = jrn.choice(
        #    subkey,
        #    ngal,
        #    shape=(min(20*ngal//100, 20000),),
        #    replace=False
        #) # 20% of data is selected
        #del subkey

        #train_sel = jnp.sort(train_sel, axis=0)
        #train_df = mobs_df.iloc[train_sel]
        train_df = pd.DataFrame(
            data=jnp.column_stack(
                (self.mags[:, :-1]-self.mags[:, 1:], self.szs)
            ),
            columns=color_names+[self.config["redshift_col"]]
        )

        '''
        templs_ref_df = tables_io.read(
            os.path.abspath(
                os.path.join(
                    self.config.data_path,
                    "SED",
                    self.config.spectra_file
                )
            ),
            tables_io.types.PD_DATAFRAME,
            fmt='hf5'
        )
        '''
        
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
        
        pars_arr = jnp.array(templs_ref_df[_DUMMY_PARS.PARAM_NAMES_FLAT])

        wls = jnp.arange(
            self.config.wlmin,
            self.config.wlmax+self.config.dwl,
            self.config.dwl
        )

        transm_arr = get_sedpy(self.config.filter_dict, wls, self.config.data_path)

        templ_tupl = [tuple(_pars) for _pars in pars_arr]

        ssp_data = load_ssp(
            os.path.abspath(
                os.path.join(
                    self.data_path,
                    "SSP",
                    self.config.ssp_file
                )
            )
        )

        templ_tupl_sps = tree_map(
            lambda partup: vmap_cols_zo(
                jnp.array(partup),
                wls,
                pzs,
                transm_arr,
                ssp_data
            ),
            templ_tupl,
            is_leaf=istuple
        )

        templs_as_dict = {}
        for it, (tname, row) in enumerate(templs_ref_df.iterrows()):
            _colrs = templ_tupl_sps[it]
            _df = pd.DataFrame(columns=color_names, data=_colrs)
            _df['z_p'] = pzs
            _df['Dataset'] = np.full(pzs.shape, row['Dataset'])
            _df['name'] = np.full(pzs.shape, tname)
            templs_as_dict.update({f"{tname}": _df})
        all_templs_df = pd.concat(
            [_df for _, _df in templs_as_dict.items()],
            ignore_index=True
        )

        list_edges = []
        for idc, c in enumerate(color_names):
            _arr = np.array(train_df[c])
            #H_data_1D, _edges1d = np.histogram(_arr[np.isfinite(_arr)], bins=self.config.colrsbins) #, bins='auto') #
            #H_templ_1d, _edges1d = np.histogram(np.array(all_templs_df[c]), bins=_edges1d)
            #H_data_1D, _edges1d = np.histogram(_arr[np.isfinite(_arr)], bins='auto')
            #H_templ_1d, _edges1d = np.histogram(np.array(all_templs_df[c]), bins=_edges1d)
            _edges1d = np.histogram_bin_edges(_arr[np.isfinite(_arr)], bins=self.config.colrsbins)
            list_edges.append(_edges1d)

        coords = []
        for c, b in zip(color_names, list_edges):
            c_idxs = np.digitize(train_df[c], b)
            coords.append(c_idxs)
        coords = np.column_stack(coords)
        train_df[[f'{c}_bin' for c in color_names]] = coords

        templ_coords = []
        for c, b in zip(color_names, list_edges):
            c_idxs = np.digitize(all_templs_df[c], b)
            templ_coords.append(c_idxs)
        templ_coords = np.column_stack(templ_coords)
        all_templs_df[[f'{c}_bin' for c in color_names]] = templ_coords

        best_templs_names = []
        allbestscores = []
        print("Computing scores in colour bins:")
        for c in color_names:
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

        '''
        tables_io.write(
            templs_score_df,
            'trained_templ.hf5',
            fmt='hf5'
        )
        '''
        self.templates_df = templs_score_df[["name", "num", "score", "Dataset", self.config["redshift_col"]]+_DUMMY_PARS.PARAM_NAMES_FLAT]
        self.model = dict(
            fo_arr=self.fo_arr,
            kt_arr=self.kt_arr,
            zo_arr=None,
            km_arr=None,
            a_arr=None,
            mo=self.m0,
            nt_array=None)
        self.add_data("model", self.model)
        self.add_data("templates", self.templates_df)


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
