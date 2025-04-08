"""Module to handle parameters
"""
# pylint: disable=trailing-newlines
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long
# pylint: disable=W1309
# pylint: disable=trailing-whitespace

import itertools
from collections import OrderedDict

import jax.numpy as jnp
import numpy as np
from diffstar.defaults import DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS

# DSPS parameters
MAH_PARAMNAMES = ["MAH_lgmO", "MAH_logtc", "MAH_early_index", "MAH_late_index"]
MS_PARAMNAMES = ["MS_lgmcrit", "MS_lgy_at_mcrit", "MS_indx_lo", "MS_indx_hi", "MS_tau_dep"]
Q_PARAMNAMES = ["Q_lg_qt", "Q_qlglgdt", "Q_lg_drop", "Q_lg_rejuv"]

DEFAULT_MAH_PARAMS_MIN = DEFAULT_MAH_PARAMS + np.array([-3.0, -0.01, -1.5, -0.5])
DEFAULT_MAH_PARAMS_MAX = DEFAULT_MAH_PARAMS + np.array([2.0, +0.01, +1.5, +0.5])

DEFAULT_MS_PARAMS_MIN = DEFAULT_MS_PARAMS - 0.25 * np.ones(5)
DEFAULT_MS_PARAMS_MAX = DEFAULT_MS_PARAMS + 0.25 * np.ones(5)

DEFAULT_Q_PARAMS_MIN = DEFAULT_Q_PARAMS - 0.1 * np.ones((4,))
DEFAULT_Q_PARAMS_MAX = DEFAULT_Q_PARAMS + 0.1 * np.ones((4,))

# Dust parameters
AV = 1
UV_BUMP_AMPL = 2.0
PLAW_SLOPE = -0.25
DEFAULT_DUST_PARAMS = [AV, UV_BUMP_AMPL, PLAW_SLOPE]
DUST_PARAMNAMES = ["AV", "UV_BUMP", "PLAW_SLOPE"]
DEFAULT_DUST_PARAMS_MIN = DEFAULT_DUST_PARAMS + np.array([-1.0, -1.0, -0.1])
DEFAULT_DUST_PARAMS_MAX = DEFAULT_DUST_PARAMS + np.array([2.0, 1.0, 0.25])


# Metalicity parameters
LGMET = -2.0
LGMETSCATTER = 0.2
DEFAULT_LGMET_PARAMS = np.array([LGMET, LGMETSCATTER])
LGMET_PARAMNAMES = ["LGMET", "LGMETSCATTER"]
DEFAULT_LGMET_PARAMS_MIN = np.array([-4.34, 0.1])
DEFAULT_LGMET_PARAMS_MAX = np.array([-3.5, 0.3])

# Age-dependant metallicity parameters
LGMET_YOUNG = 1.0
LGMET_OLD = -2.0
DEFAULT_AGEDEPLGMET_PARAMS = np.array([LGMET_YOUNG, LGMET_OLD])
AGEDEPLGMET_PARAMNAMES = ["LGMET_YOUNG", "LGMET_OLD"]
DEFAULT_AGEDEPLGMET_PARAMS_MIN = np.array([-2.0, -4.34])
DEFAULT_AGEDEPLGMET_PARAMS_MAX = np.array([3.0, 0.0])

# Scaling free-parameter
SCALE = 1.0
DEFAULT_SCALE_PARAMS = np.array([SCALE])
SCALE_PARAMNAMES = ["SCALE"]
DEFAULT_SCALE_PARAMS_MIN = np.array([1.0e-12])
DEFAULT_SCALE_PARAMS_MAX = np.array([1.0e12])

# bound parameters together
DEFAULT_PARAMS = [DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS, DEFAULT_DUST_PARAMS, DEFAULT_LGMET_PARAMS, DEFAULT_AGEDEPLGMET_PARAMS, DEFAULT_SCALE_PARAMS]

PARAMS_MIN = np.concatenate(
    [DEFAULT_MAH_PARAMS_MIN, DEFAULT_MS_PARAMS_MIN, DEFAULT_Q_PARAMS_MIN, DEFAULT_DUST_PARAMS_MIN, DEFAULT_LGMET_PARAMS_MIN, DEFAULT_AGEDEPLGMET_PARAMS_MIN, DEFAULT_SCALE_PARAMS_MIN]
)

PARAMS_MAX = np.concatenate(
    [DEFAULT_MAH_PARAMS_MAX, DEFAULT_MS_PARAMS_MAX, DEFAULT_Q_PARAMS_MAX, DEFAULT_DUST_PARAMS_MAX, DEFAULT_LGMET_PARAMS_MAX, DEFAULT_LGMET_PARAMS_MAX, DEFAULT_SCALE_PARAMS_MAX]
)

# PARAMS_MIN= jnp.array(PARAMS_MIN)
# PARAMS_MAX= jnp.array(PARAMS_MAX)

INIT_PARAMS = np.concatenate(DEFAULT_PARAMS)
INIT_PARAMS = jnp.array(INIT_PARAMS)

PARAM_NAMES = [MAH_PARAMNAMES, MS_PARAMNAMES, Q_PARAMNAMES, DUST_PARAMNAMES, LGMET_PARAMNAMES, AGEDEPLGMET_PARAMNAMES, SCALE_PARAMNAMES]
PARAM_NAMES_FLAT = list(itertools.chain(*PARAM_NAMES))

DICT_PARAM_MAH_true = OrderedDict(
    [(MAH_PARAMNAMES[0], DEFAULT_MAH_PARAMS[0]), (MAH_PARAMNAMES[1], DEFAULT_MAH_PARAMS[1]), (MAH_PARAMNAMES[2], DEFAULT_MAH_PARAMS[2]), (MAH_PARAMNAMES[3], DEFAULT_MAH_PARAMS[3])]
)

DICT_PARAM_MAH_true_selected = OrderedDict([(MAH_PARAMNAMES[0], DEFAULT_MAH_PARAMS[0])])

DICT_PARAM_MS_true = OrderedDict(
    [
        (MS_PARAMNAMES[0], DEFAULT_MS_PARAMS[0]),
        (MS_PARAMNAMES[1], DEFAULT_MS_PARAMS[1]),
        (MS_PARAMNAMES[2], DEFAULT_MS_PARAMS[2]),
        (MS_PARAMNAMES[3], DEFAULT_MS_PARAMS[3]),
        (MS_PARAMNAMES[4], DEFAULT_MS_PARAMS[4]),
    ]
)

DICT_PARAM_Q_true = OrderedDict([(Q_PARAMNAMES[0], DEFAULT_Q_PARAMS[0]), (Q_PARAMNAMES[1], DEFAULT_Q_PARAMS[1]), (Q_PARAMNAMES[2], DEFAULT_Q_PARAMS[2]), (Q_PARAMNAMES[3], DEFAULT_Q_PARAMS[3])])

DICT_PARAM_DUST_true = OrderedDict([(DUST_PARAMNAMES[0], DEFAULT_DUST_PARAMS[0]), (DUST_PARAMNAMES[1], DEFAULT_DUST_PARAMS[1]), (DUST_PARAMNAMES[2], DEFAULT_DUST_PARAMS[2])])

DICT_PARAM_DUST_true_selected = OrderedDict([(DUST_PARAMNAMES[0], DEFAULT_DUST_PARAMS[0])])

DICT_PARAM_LGMET_true = OrderedDict([(LGMET_PARAMNAMES[0], DEFAULT_LGMET_PARAMS[0]), (LGMET_PARAMNAMES[1], DEFAULT_LGMET_PARAMS[1])])

DICT_PARAM_AGEDEPLGMET_true = OrderedDict([(AGEDEPLGMET_PARAMNAMES[0], DEFAULT_AGEDEPLGMET_PARAMS[0]), (AGEDEPLGMET_PARAMNAMES[1], DEFAULT_AGEDEPLGMET_PARAMS[1])])

DICT_PARAM_SCALE_true = OrderedDict([(SCALE_PARAMNAMES[0], DEFAULT_SCALE_PARAMS[0])])

DICT_PARAMS_true = DICT_PARAM_MAH_true
DICT_PARAMS_true.update(DICT_PARAM_MS_true)
DICT_PARAMS_true.update(DICT_PARAM_Q_true)
DICT_PARAMS_true.update(DICT_PARAM_DUST_true)
DICT_PARAMS_true.update(DICT_PARAM_LGMET_true)
DICT_PARAMS_true.update(DICT_PARAM_AGEDEPLGMET_true)
DICT_PARAMS_true.update(DICT_PARAM_SCALE_true)

# increase range of parameters
# check in article

FLAG_INCREASE_RANGE_MAH = False
if FLAG_INCREASE_RANGE_MAH:
    # MAH_logtc
    DEFAULT_MAH_PARAMS_MIN[1] = 0.01
    DEFAULT_MAH_PARAMS_MAX[1] = 0.15
    # MAH_early_index
    DEFAULT_MAH_PARAMS_MIN[2] = 0.1
    DEFAULT_MAH_PARAMS_MAX[2] = 10.0  # 0.10 -- erreur de frappe? ça bloquait cette valeur à 0.1
    # MAH_late_index
    DEFAULT_MAH_PARAMS_MIN[3] = 0.1
    DEFAULT_MAH_PARAMS_MAX[3] = 10.0

FLAG_INCREASE_RANGE_MS = False
if FLAG_INCREASE_RANGE_MS:
    # MS_lgmcrit  12
    DEFAULT_MS_PARAMS_MIN[0] = 6.0  # 9.0
    DEFAULT_MS_PARAMS_MAX[0] = 15.0  # 13.0
    # MS_lgy_at_mcrit : -1
    DEFAULT_MS_PARAMS_MIN[1] = -2.0
    DEFAULT_MS_PARAMS_MAX[1] = -0.7
    # MS_indx_lo : 1
    DEFAULT_MS_PARAMS_MIN[2] = 0.7
    DEFAULT_MS_PARAMS_MAX[2] = 2.0
    # MS_indx_hi : -1
    DEFAULT_MS_PARAMS_MIN[3] = -2.0
    DEFAULT_MS_PARAMS_MAX[3] = -0.7
    # MS_tau_dep : 2
    DEFAULT_MS_PARAMS_MIN[4] = 0.7
    DEFAULT_MS_PARAMS_MAX[4] = 3.0

FLAG_INCREASE_RANGE_Q = False
if FLAG_INCREASE_RANGE_Q:
    #'Q_lg_qt', 1.0)
    DEFAULT_Q_PARAMS_MIN[0] = 0.5
    DEFAULT_Q_PARAMS_MAX[0] = 2.0
    # ('Q_qlglgdt', -0.50725),
    DEFAULT_Q_PARAMS_MIN[1] = -2.0
    DEFAULT_Q_PARAMS_MAX[1] = -0.2
    # ('Q_lg_drop', -1.01773),
    DEFAULT_Q_PARAMS_MIN[2] = -2.0
    DEFAULT_Q_PARAMS_MAX[2] = -0.5
    # ('Q_lg_rejuv', -0.212307),
    DEFAULT_Q_PARAMS_MIN[3] = -2.0
    DEFAULT_Q_PARAMS_MAX[3] = -0.1

PARAMS_MIN = np.concatenate(
    [DEFAULT_MAH_PARAMS_MIN, DEFAULT_MS_PARAMS_MIN, DEFAULT_Q_PARAMS_MIN, DEFAULT_DUST_PARAMS_MIN, DEFAULT_LGMET_PARAMS_MIN, DEFAULT_AGEDEPLGMET_PARAMS_MIN, DEFAULT_SCALE_PARAMS_MIN]
)

PARAMS_MAX = np.concatenate(
    [DEFAULT_MAH_PARAMS_MAX, DEFAULT_MS_PARAMS_MAX, DEFAULT_Q_PARAMS_MAX, DEFAULT_DUST_PARAMS_MAX, DEFAULT_LGMET_PARAMS_MAX, DEFAULT_AGEDEPLGMET_PARAMS_MAX, DEFAULT_SCALE_PARAMS_MAX]
)

PARAMS_MIN = jnp.array(PARAMS_MIN)
PARAMS_MAX = jnp.array(PARAMS_MAX)


def paramslist_to_dict(params_list, param_names):
    """
    Convert the list of parameters into a dictionnary
    :param params_list: list of params values
    :type params_list: float in an array

    :param param_names: list of parameter names
    :type params_names: strings in an array

    :return: dictionnary of parameters
    :rtype: dictionnary
    """
    list_of_tuples = list(zip(param_names, params_list, strict=False))
    dict_params = OrderedDict(list_of_tuples)
    return dict_params


class SSPParametersFit:
    """Contain all necessary parameters to fit SSP
    Note this class must use the values calculated outside this class
    """

    def __init__(self):
        """Book space for initialisation constants"""
        # DSPS (diffstar) parameters
        self.DEFAULT_MAH_PARAMS = DEFAULT_MAH_PARAMS  # [DEFAULT_MAH_PARAMS[0], DEFAULT_MAH_PARAMS[-1]]
        self.MAH_PARAMNAMES = MAH_PARAMNAMES  # [MAH_PARAMNAMES[0], MAH_PARAMNAMES[-1]]
        self.DEFAULT_MAH_PARAMS_MIN = DEFAULT_MAH_PARAMS_MIN  # [DEFAULT_MAH_PARAMS_MIN[0], DEFAULT_MAH_PARAMS_MIN[-1]]
        self.DEFAULT_MAH_PARAMS_MAX = DEFAULT_MAH_PARAMS_MAX  # [DEFAULT_MAH_PARAMS_MAX[0], DEFAULT_MAH_PARAMS_MAX[-1]]

        self.DEFAULT_MS_PARAMS = DEFAULT_MS_PARAMS  # [DEFAULT_MS_PARAMS[0], DEFAULT_MS_PARAMS[2], DEFAULT_MS_PARAMS[3]]
        self.MS_PARAMNAMES = MS_PARAMNAMES  # [MS_PARAMNAMES[0], MS_PARAMNAMES[2], MS_PARAMNAMES[3]]
        self.DEFAULT_MS_PARAMS_MIN = DEFAULT_MS_PARAMS_MIN  # [DEFAULT_MS_PARAMS_MIN[0], DEFAULT_MS_PARAMS_MIN[2], DEFAULT_MS_PARAMS_MIN[3]]
        self.DEFAULT_MS_PARAMS_MAX = DEFAULT_MS_PARAMS_MAX  # [DEFAULT_MS_PARAMS_MAX[0], DEFAULT_MS_PARAMS_MAX[2], DEFAULT_MS_PARAMS_MAX[3]]

        # We only keep Q drop parameters
        self.DEFAULT_Q_PARAMS = DEFAULT_Q_PARAMS
        self.Q_PARAMNAMES = Q_PARAMNAMES

        self.DEFAULT_Q_PARAMS_MIN = DEFAULT_Q_PARAMS_MIN
        self.DEFAULT_Q_PARAMS_MAX = DEFAULT_Q_PARAMS_MAX

        # Dust parameters
        self.AV = AV
        self.UV_BUMP = UV_BUMP_AMPL
        self.PLAW_SLOPE = PLAW_SLOPE
        self.DEFAULT_DUST_PARAMS = [self.AV, self.UV_BUMP, self.PLAW_SLOPE]
        self.DUST_PARAMNAMES = DUST_PARAMNAMES
        self.DEFAULT_DUST_PARAMS_MIN = DEFAULT_DUST_PARAMS_MIN
        self.DEFAULT_DUST_PARAMS_MAX = DEFAULT_DUST_PARAMS_MAX

        """
        # Age-dependant metallicity parameters - We only keep the young metallicity
        self.LGMET_YOUNG = LGMET_YOUNG
        self.DEFAULT_AGEDEPLGMET_PARAMS = np.array([self.LGMET_YOUNG])
        self.AGEDEPLGMET_PARAMNAMES = [AGEDEPLGMET_PARAMNAMES[0]]
        self.DEFAULT_AGEDEPLGMET_PARAMS_MIN = [DEFAULT_AGEDEPLGMET_PARAMS_MIN[0]]
        self.DEFAULT_AGEDEPLGMET_PARAMS_MAX = [DEFAULT_AGEDEPLGMET_PARAMS_MAX[0]]
        """
        # Scaling free parameter
        # self.DEFAULT_SCALE_PARAMS = DEFAULT_SCALE_PARAMS
        # self.SCALE_PARAMNAMES = SCALE_PARAMNAMES
        # self.DEFAULT_SCALE_PARAMS_MIN = DEFAULT_SCALE_PARAMS_MIN
        # self.DEFAULT_SCALE_PARAMS_MAX = DEFAULT_SCALE_PARAMS_MAX

        # bound parameters together
        self.DEFAULT_PARAMS = [self.DEFAULT_MAH_PARAMS, self.DEFAULT_MS_PARAMS, self.DEFAULT_Q_PARAMS, self.DEFAULT_DUST_PARAMS]  # , self.DEFAULT_SCALE_PARAMS en avant-dernier

        self.PARAMS_MIN = np.concatenate(
            [self.DEFAULT_MAH_PARAMS_MIN, self.DEFAULT_MS_PARAMS_MIN, self.DEFAULT_Q_PARAMS_MIN, self.DEFAULT_DUST_PARAMS_MIN]
        )  # , self.DEFAULT_SCALE_PARAMS_MIN en avant-dernier
        self.PARAMS_MAX = np.concatenate(
            [self.DEFAULT_MAH_PARAMS_MAX, self.DEFAULT_MS_PARAMS_MAX, self.DEFAULT_Q_PARAMS_MAX, self.DEFAULT_DUST_PARAMS_MAX]
        )  # , self.DEFAULT_SCALE_PARAMS_MAX en avant-dernier

        self.PARAMS_MIN = jnp.array(self.PARAMS_MIN)
        self.PARAMS_MAX = jnp.array(self.PARAMS_MAX)

        self.INIT_PARAMS = np.concatenate(self.DEFAULT_PARAMS)
        self.INIT_PARAMS = jnp.array(self.INIT_PARAMS)

        self.PARAM_NAMES = [self.MAH_PARAMNAMES, self.MS_PARAMNAMES, self.Q_PARAMNAMES, self.DUST_PARAMNAMES]  # , self.SCALE_PARAMNAMES en avant-dernier
        self.PARAM_NAMES_FLAT = list(itertools.chain(*self.PARAM_NAMES))

        self.DICT_PARAM_MAH_true = OrderedDict([(name, self.DEFAULT_MAH_PARAMS[k]) for k, name in enumerate(self.MAH_PARAMNAMES)])

        self.DICT_PARAM_MS_true = OrderedDict([(name, self.DEFAULT_MS_PARAMS[k]) for k, name in enumerate(self.MS_PARAMNAMES)])

        self.DICT_PARAM_Q_true = OrderedDict([(name, self.DEFAULT_Q_PARAMS[k]) for k, name in enumerate(self.Q_PARAMNAMES)])

        # self.DICT_PARAM_SCALE_true = OrderedDict([(name, self.DEFAULT_SCALE_PARAMS[k]) for k, name in enumerate(self.SCALE_PARAMNAMES)])

        self.DICT_PARAM_DUST_true = OrderedDict([(name, self.DEFAULT_DUST_PARAMS[k]) for k, name in enumerate(self.DUST_PARAMNAMES)])

        self.DICT_PARAMS_true = self.DICT_PARAM_MAH_true
        self.DICT_PARAMS_true.update(self.DICT_PARAM_MS_true)
        self.DICT_PARAMS_true.update(self.DICT_PARAM_Q_true)
        # self.DICT_PARAMS_true.update(self.DICT_PARAM_SCALE_true)
        self.DICT_PARAMS_true.update(self.DICT_PARAM_DUST_true)

    def __repr__(self) -> str:
        all_str = []
        all_str.append("Class SSPParametersFit : ")
        all_str.append(" - Name     : " + str(self.PARAM_NAMES_FLAT))
        all_str.append(" - Mean Val : " + str(self.INIT_PARAMS))
        all_str.append(" - Min Val  : " + str(self.PARAMS_MIN))
        all_str.append(" - Max Val  : " + str(self.PARAMS_MAX))

        return "\n".join(all_str)
