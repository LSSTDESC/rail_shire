from ._version import __version__

from rail.estimation.algos.analysis import run_from_inputs
from rail.estimation.algos.io_utils import load_data_for_run, json_to_inputs
from rail.estimation.algos import example_module

__all__ = [
    "__version__",
    "json_to_inputs",
    "load_data_for_run",
    "run_from_inputs",
    "example_module"
]
