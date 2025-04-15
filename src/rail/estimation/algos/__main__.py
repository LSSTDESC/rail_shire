#!/usr/bin/env python3
#
#  __main__.py
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

import sys

def main(args):
    """
    Main function to start an external call to the photoZ module. Arguments must be the JSON configuration file.
    """
    from jaxlib.xla_extension import XlaRuntimeError

    from .io_utils import json_to_inputs
    from .analysis import run_from_inputs

    conf_json = args[1] if len(args) > 1 else "./defaults.json"  # le premier argument de args est toujours `__main__.py` ; attention à la localisation du fichier !
    inputs = json_to_inputs(conf_json)
    try:
        tree_of_results_dict = run_from_inputs(inputs)
    except XlaRuntimeError:
        print("Dataset is too large for a single run (OOM error) - running on chunks instead :")
        import os

        import pandas as pd
        from jax import numpy as jnp

        filters_dict = inputs["photoZ"]["Filters"]
        filters_names = [_f["name"] for _, _f in filters_dict.items()]
        data_path = os.path.abspath(inputs["photoZ"]["Dataset"]["path"])
        data_ismag = inputs["photoZ"]["Dataset"]["type"].lower() == "m"
        if inputs["photoZ"]["Dataset"]["is_ascii"]:
            from .io_utils import catalog_ASCIItoHDF5

            h5catpath = catalog_ASCIItoHDF5(data_path, data_ismag, filt_names=filters_names)
        else:
            h5catpath = data_path
        cat_df = pd.read_hdf(h5catpath)

        n_chunks = (cat_df.shape[0] // 5000) + 1
        l_last_chunk = cat_df.shape[0] % 5000

        pz_dicts = []
        for ichunk in range(n_chunks):
            _bnds = (ichunk * 5000, (ichunk + 1) * 5000) if ichunk < n_chunks - 1 else (ichunk * 5000, ichunk * 5000 + l_last_chunk)
            print(f"Running on chunk n. {ichunk+1}/{n_chunks} ; objects {_bnds[0]+1} to {_bnds[1]}...")
            pz_dicts.append(run_from_inputs(inputs, bounds=_bnds))

        # tree_of_results_dict = {_key: jnp.concatenate([_dict[_key] for _dict in pz_dicts], axis=0) for _key in pz_dicts[0]}
        tree_of_results_dict = {"z_grid": pz_dicts[0]["z_grid"], "PDZ": jnp.concatenate([_dict["PDZ"] for _dict in pz_dicts], axis=1)}
        for _key in ["redshift", "z_ML", "z_mean", "z_med"]:
            tree_of_results_dict.update({_key: jnp.concatenate([_dict[_key] for _dict in pz_dicts], axis=0)})

    if inputs["photoZ"]["save results"]:
        from .io_utils import photoZtoHDF5

        # df_gal.to_pickle(f"{inputs['run name']}_results_summary.pkl")
        # with open(f"{inputs['photoZ']['run name']}_posteriors_dict.pkl", "wb") as handle:
        #    pickle.dump(tree_of_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        resfile = photoZtoHDF5(f"{inputs['photoZ']['run name']}_posteriors_dict.h5", tree_of_results_dict)
    else:
        resfile = "Run terminated correctly but results were not saved, please check your input configuration."
    print(resfile)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
