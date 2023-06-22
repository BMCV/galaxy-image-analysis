from .pipeline import Stage

import numpy as np


DSM_CONFIG_DEFAULTS = {
    'cachesize': 1,
    'cachetest': None,
    'sparsity_tol': 0,
    'init': 'elliptical',
    'smooth_amount': 10,
    'epsilon': 1.0,
    'alpha': 0.5,
    'scale': 1000,
    'smooth_subsample': 20,
    'gaussian_shape_multiplier': 2,
    'smooth_mat_dtype': 'float32',
    'smooth_mat_max_allocations': np.inf,
    'background_margin': 20,
    'cp_timeout': 300,
}


class DSM_Config(Stage):
    """Fetches the hyperparameters from the ``dsm`` namespace and provides them as an output.

    The purpose of this stage is to provide the hyperparameters from the ``dsm`` namespace as the output ``dsm_cfg``, which is a dictionary of the hyperparameters without the leading ``dsm/`` namespace prefix. This enables any stage to access the DSM-related hyperparameters, like the :py:class:`~.c2freganal.C2F_RegionAnalysis` and :py:class:`~.globalenergymin.GlobalEnergyMinimization` stages, without having to access the ``dsm`` hyperparameter namespace. Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    Hyperparameters
    ---------------

    The following hyperparameters are fetched:

    ``dsm/cachesize``
        The maximum number of entries used for caching during convex programming. This concerns invocations of the callback function ``F`` used by the `cvxopt solver for nonlinear problems <https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives>`_. Defaults to 1.

    ``dsm/cachetest``
        The test function to be used for cache testing. If ``None``, then ``numpy.array_equal`` will be used. Using other functions like ``numpy.allclose`` has shown to introduce numerical instabilities. Defaults to ``None``.

    ``dsm/sparsity_tol``
        Absolute values below this threshold will be treated as zeros during optimization. Defaults to 0.

    ``dsm/init``
        Either a function or a string. If this is function, then it will be called to determine the initialization, and the dimension of the vector :math:`\\xi` will be passed as a parameter. If this is a string, then the initialization corresponds to the result of convex programming using elliptical models (if set to ``elliptical``, see Supplemental Material 6 of the :ref:`paper <references>`) or a zeros vector of is used (otherwise). Defaults to ``elliptical``.

    ``dsm/smooth_amount``
        Corresponds to :math:`\\sigma_G` described in :ref:`pipeline_theory_dsm`. Defaults to 10, or to ``AF_smooth_amount × scale`` if computed automatically (forced to :math:`\\geq 4` and ``AF_smooth_amount`` defaults to 0.2).

    ``dsm/smooth_subsample``
        Corresponds to the amount of sub-sampling used to obtain the matrix :math:`\\tilde G_\\omega` in the :ref:`paper <references>` (Section 3.3). Defaults to 20, or to ``AF_smooth_subsample × scale`` if computed automatically (forced to :math:`\\geq 8` and ``AF_smooth_subsample`` defaults to 0.4).

    ``dsm/epsilon``
        Corresponds to the constant :math:`\\epsilon` which is used for the smooth approximation of the regularization term :math:`\\|\\xi\\|_1 \\approx \\mathbb 1^\\top_\\Omega \\sqrt{\\xi^2 + \\epsilon} - \\sqrt{\\epsilon} \\cdot \\#\\Omega` (see Supplemental Material 2 of the :ref:`paper <references>`). Defaults to 1.

    ``dsm/alpha``
        Governs the regularization of the deformations and corresponds to :math:`\\alpha` described in :ref:`pipeline_theory_cvxprog`. Increasing this value leads to a smoother segmentation result. Defaults to 0.5, or to ``AF_alpha × scale^2`` if computed automatically (where ``AF_alpha`` corresponds to :math:`\\alpha_\\text{factor}` in the :ref:`paper <references>` and defaults to 5e-4).

    ``dsm/scale``
        Fixed factor used during convex programming to slightly improve numerical stabilities. Defaults to 1000.

    ``dsm/gaussian_shape_multiplier``
        The Gaussian function with standard deviation :math:`\\sigma_G` used to construct the block Toeplitz matrix :math:`G_\\omega` is cut off after :math:`4 \\sigma_G` multiplied by this value (see :ref:`pipeline_theory_dsm`). Defaults to 2.

    ``dsm/smooth_mat_dtype``
        A string indicating the data type used for the matrix :math:`G_\\omega`. Defaults to ``float32``.

    ``dsm/smooth_mat_max_allocations``
        Maximum number of simultaneous allocation of the matrix :math:`\\tilde G_\\omega` during parallel processing (see Section 3.3 of the :ref:`paper <references>`, each allocation might require a considerable amount of system memory).

    ``dsm/background_margin``
        Governs the amount of image background included in the obtained image region. This is the width of the "stripe" of background retained around each connected foreground region (in pixels). See Supplemental Material 6 of the :ref:`paper <references>` for details, however, due to a transmission error, the threshold :math:`\\sigma_G` in Eq. (S11) was misstated by a factor of 2 (the correct threshold is :math:`2\\sigma_G`). Defaults to 20, or to ``AF_background_margin × scale`` if computed automatically (forced to :math:`\\geq 8` and ``AF_background_margin`` defaults to 0.4).

    ``dsm/cp_timeout``
        The maximum run time of convex programming for each object (in seconds). The convex optimization will be interrupted if it takes longer than that (the :py:meth:`~superdsm.objects.cvxprog` function will report the status ``fallback`` in this case). If this is set to ``None``, the run time is not limited. Defaults to 300 (i.e. 5 minutes).
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(DSM_Config, self).__init__('dsm', inputs=[], outputs=['dsm_cfg'])

    def process(self, input_data, cfg, out, log_root_dir):
        dsm_cfg = {
            key: cfg.get(key, DSM_CONFIG_DEFAULTS[key]) for key in DSM_CONFIG_DEFAULTS.keys()
        }
        
        return {
            'dsm_cfg': dsm_cfg
        }

    def configure_ex(self, scale, radius, diameter):
        return {
            'alpha': (scale ** 2, 0.0005),
            'smooth_amount':     (scale, 0.2, dict(type=int, min=4)),
            'smooth_subsample':  (scale, 0.4, dict(type=int, min=8)),
            'background_margin': (scale, 0.4, dict(type=int, min=8)),
        }

