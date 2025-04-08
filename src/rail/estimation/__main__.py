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
    from rail.dsps_fors2_pz import json_to_inputs, run_from_inputs

    conf_json = args[1] if len(args) > 1 else "./defaults.json"  # le premier argument de args est toujours `__main__.py` ; attention à la localisation du fichier !
    inputs = json_to_inputs(conf_json)

    tree_of_results_dict = run_from_inputs(inputs)

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
    sys.exit(main(sys.argv))
