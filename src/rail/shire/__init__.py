from ._version import __version__

#from rail.estimation.algos.analysis import run_from_inputs
from rail.estimation.algos.io_utils import json_to_inputs #, load_data_for_run
from rail.estimation.algos import example_module
from rail.estimation.algos.inform import ShireInformer
from rail.estimation.algos.estimate import ShireEstimator

__all__ = [
    "__version__",
    "json_to_inputs",
    "example_module",
    "ShireInformer",
    "ShireEstimator"
]
