
**TEMPORARY FIX: to avoid incompatibilities that were noticed on platforms other than NERSC, the `pyproject.toml` file now imposes `numpy<2` and corresponding `qp` and `tables_io` versions. This may cause clashes with existing `RAIL` installation, please be careful and install `RAIL_SHIRE` in a separate environment.**

# `rail_shire`
`SHIRE` stands for **S**tar-formation **HI**story **R**edshift **E**stimator. It is a photometric redshift estimation code, that works on the principle of *Template Fitting* (TF).
Unlike most other TF codes, `SHIRE` does not use SED templates in `ASCII` files; rather, it synthetises reference photometry along a specified redshift grid using *Stellar Population Synthesis*. To do so, *templates* are given as a set of parameters compatible with the SPS tool [DSPS](https://dsps.readthedocs.io/en/latest/) [(Hearin *et al.*, 2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1741H/abstract).

`SHIRE` can be run in two modes:
- The 'SPS' mode, that computes the star-formation rate and the corresponding SED and photometry at any given redshift when building the reference $\mathrm{photometry} = f(z)$ grid;
- The 'Legacy' mode, that computes the SED once at the native redshift of a template, and then shifts it IAW $\lambda_\mathrm{obs} = (1+z)\lambda_\mathrm{em}$ to compute the reference $\mathrm{photometry} = f(z)$ grid at any $z$. This is similar to existing TF codes.

_Caution: `SHIRE` was designed with extensive LSST-like datasets in mind, and therefore uses `JAX` to be compatible with GPU architectures. This improves its speed greatly on appropriate machines but may cause crashes due to high memory requirements on other platforms. Therefore, `SHIRE` is likely not the best suited code to be run on a personal laptop or limited-resources shared installation... sorry about that!_

## Getting Started

### Virtual environment
Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```bash
conda create -n <env_name> python=3.11
conda activate <env_name>
```

### Installing `RAIL`
This software is meant to be part of a broader software suite: [RAIL](https://github.com/LSSTDESC/rail).
Once you have created a new environment, you can install `RAIL` by following the [production installation steps](https://rail-hub.readthedocs.io/en/latest/source/installation.html#production-installation) as described in `RAIL`'s [documentation](https://rail-hub.readthedocs.io/en/latest/index.html).
This should be enough, unless of course you are an active `RAIL` developer, in which case another installation process may be better suited, see the doc for info.

### Installing `RAIL_SHIRE`
Finally, you can `clone` this project and install it in your environment `<env_name>`:

```bash
conda activate <env_name>
cd <dir_where_to_clone_the_repo>
git clone git@github.com:lsstdesc/rail_shire.git
cd rail_shire
pip install --no-cache-dir .
```

Alternatively, if you wish to contribute to `RAIL_SHIRE`, you can install this project for local
development using the following commands:

```bash
./.setup_dev.sh
conda install pandoc
```

Notes:
1. `./.setup_dev.sh` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)

## Running examples
Examples of how to use `rail_shire` are provided as `jupyter` notebooks in the `examples/` directory, for several datasets.
_Note: it will most likely be necessary to update paths in these notebooks so that they work properly!_
- For simulated LSST-like data, start with [LSSTsim minimal example](examples/SHIRE_demo_LSSTsim_mini.ipynb) then perhaps move on to [LSSTsim](examples/SHIRE_demo_LSSTsim.ipynb) and [LSSTxROMANsim](examples/SHIRE_demo_LSSTxROMANsim.ipynb) are the best places to start. These notebooks also showcase some plotting utilities available in `rail_shire`, however these can be time-consuming so it is recommended to skip the corresponding cells in a first run.
- For a small sample of COSMOS data, it is necessary to [generate training samples](examples/TrainingSample_COSMOS2020.ipynb) first (pay attention to the paths and datasets used...), then try to adapt the minimal example to the new datasets!
- If you are in an appropriate environment (_e.g._ at NERSC), you may be tempted to see how to generate your own dataset with `RAIL`'s functionalities: [CosmoDC2 data](examples/get_CosmoDC2_gold_phot.ipynb) and [RomanXRubin data](examples/get_RomanRubin_gold_phot.ipynb); then adapt the LSSTsim examples above. You can also try running `SHIRE` [on actual Data Preview 1 data](examples/SHIRE_run_DESC-DP1.ipynb), and even [cross-matched with data from Euclid](examples/SHIRE_run_DESC-DP1-Euclid.ipynb)!
- Finally, checkout the quick examples of post-processing: [comparison of prior distributions](examples/compare_priors.ipynb) and [PDZ evaluation](examples/Evaluation_demo_LSSTsim_v2.ipynb).

__* Please remember that all examples shall be adapted to your environment before being able to run properly! That includes changing some paths and variable names, commenting/uncommenting cells, etc. *__

## Building your own run
The structure of photometric redshifts estimation with `SHIRE` is imposed by that of `RAIL`. First, you must have two datasets: one for training and one for estimation of "test". Then, the necessary steps are summarized below (for detailed examples, syntax and imports, please refer to the examples notebooks):
1. Load training and test data into the `DataStore` with `DS.add_data(*args, handle_class=TableHandle)` or `DS.read_file(*args, handle_class=TableHandle)`. This will make the `training_data` and `test_data` avilable as `TableHandle` objects.
2. Set-up the `ShireInformer` object with `informer = ShireInformer(**config_dict)`
3. Train the informer (_i.e._ select appropriate templates and fit the prior): `informer.inform(training_data)`. The templates and prior are then available as `TableHandle` and `ModelHandle` objects respectively.
4. Set-up the `ShireEstimator` object with `estimator = ShireEstimator(model=informer.get_handle('model'), templates=informer.get_handle('templates'), **other_config_dict)` where you can see how we used the outputs of the training as inputs for the estimation.
5. Run the estimation `estimator.estimate(test_data)`. The outputs are written in an `HDF5` file that can be opened with [qp](https://qp.readthedocs.io/en/main/) or loaded into the `DataStore` with `DS.read_file(*args, handle_class=QPHandle)` for analysis.

## Project template
[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/rail_shire?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/rail_shire/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/LSSTDESC/rail_shire/smoke-test.yml)](https://github.com/LSSTDESC/rail_shire/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/LSSTDESC/rail_shire/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/rail_shire)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).
