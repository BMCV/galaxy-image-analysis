from ._aux import copy_dict, uplift_smooth_matrix, join_path, SystemSemaphore, get_ray_1by1
from .output import get_output
from .dsm import DeformableShapeModel, CP, SmoothMatrixFactory, Energy

import ray
import sys, io, contextlib, traceback, time
import scipy.ndimage as ndi
import numpy as np
import skimage.morphology as morph


class BaseObject:
    """Each object of this class represents a segmentation mask, consisting of a *foreground fragment* and an *offset*.

    The attributes :py:attr:`~.fg_offset` and :py:attr:`~.fg_fragment` are initialized with ``None``, indicating that they have not been computed yet.
    """

    def __init__(self):
        self.fg_offset   = None
        self.fg_fragment = None
    
    def fill_foreground(self, out, value=True):
        """Reproduces the segmentation mask of this object.
        
        The foreground fragment is written into the image ``out``, which must be an object of ``numpy.ndarray`` type. Image points corresponding to the segmentation mask will be set to ``value``.

        The method requires that :py:attr:`~.fg_offset` and :py:attr:`~.fg_fragment` have been computed before (see the :py:meth:`~compute_objects` function).

        :return: The slice corresponding to the altered region of ``out``.

        .. runblock:: pycon

           >>> import superdsm.objects
           >>> import numpy as np
           >>> obj = superdsm.objects.BaseObject()
           >>> obj.fg_fragment = np.array([[False,  True],
           ...                             [ True,  True],
           ...                             [ True, False]])
           >>> obj.fg_offset = (1, 2)
           >>> mask = np.zeros((4, 5), bool)
           >>> obj.fill_foreground(mask)
           >>> mask
        
        This method is the counterpart of the :py:meth:`~extract_foreground_fragment` function.
        """
        assert self.fg_offset is not None
        assert self.fg_fragment is not None
        sel = np.s_[self.fg_offset[0] : self.fg_offset[0] + self.fg_fragment.shape[0], self.fg_offset[1] : self.fg_offset[1] + self.fg_fragment.shape[1]]
        out[sel] = value * self.fg_fragment
        return sel


class Object(BaseObject):
    """Each object of this class represents a set of atomic image regions.

    Each object corresponds to a realization of the set :math:`X` in the paper (see :ref:`Section 3 <references>`). It also represents a segmented object after it has been passed to the :py:meth:`compute_objects` function.

    :ivar footprint: Set of integer labels that identify the atomic image regions, which the object represents.
    :ivar energy: The value of the set energy function :math:`\\nu(X)` (see :ref:`pipeline_theory_jointsegandclustersplit`).
    :ivar on_boundary: ``True`` if this object intersects the image boundary.
    :ivar is_optimal: ``True`` if optimization of :py:attr:`~.energy` was successful.
    :ivar processing_time: How long the computation of the attributes took (in seconds).

    The attributes :py:attr:`~.energy`, :py:attr:`~.on_boundary`, :py:attr:`~.is_optimal`, :py:attr:`~.processing_time` are initialized with ``nan``, which indicates that the values have not been computed yet, i.e. the object was not passed to the :py:meth:`~compute_objects` function yet.

    Possible reasons for :py:attr:`~.is_optimal` being ``False`` include the rare cases of numerical issues during optimization as well as regions of the size of a single pixel.
    """

    def __init__(self):
        self.footprint       = set()
        self.energy          = np.nan
        self.on_boundary     = np.nan
        self.is_optimal      = np.nan
        self.processing_time = np.nan
    
    def get_mask(self, atoms):
        """Returns binary image corresponding to the union of the represented set of atomic image regions.

        :param atoms: Integer-valued image representing the universe of atomic image regions (each atomic image region has a unique label, which is the integer value).
        :return: Binary image corresponding to :math:`\\tilde\\omega(X) = \\bigcup X` in the paper, where each object of this class is a realization of the set :math:`X` (see :ref:`Section 3 <references>`).

        .. runblock:: pycon

           >>> import superdsm.objects
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2],
           ...                   [1, 3, 2],
           ...                   [3, 3, 3]])
           >>> obj = superdsm.objects.Object()
           >>> obj.footprint = set([2, 3])
           >>> obj.get_mask(atoms)
        """
        return np.in1d(atoms, list(self.footprint)).reshape(atoms.shape)

    def get_cvxprog_region(self, y, atoms, background_margin):
        """Returns the image region used for convex programming.

        :param y: Object of :py:class:`~.image.Image` class, corresponding to the offset image intensities.
        :param atoms: Integer-valued image representing the universe of atomic image regions (each atomic image region has a unique label, which is the integer value).
        :param background_margin: Governs the amount of image background included in the obtained image region. This is the width of the "stripe" of background retained around each connected foreground region (in pixels, intersected with the image region determined by the :py:meth:`~.get_mask` method).
        :return: Image region corresponds to :math:`\\tilde\\omega'(X)` in the paper (see :ref:`Supplemental Material 6 <references>`), where each object of this class is a realization of the set :math:`X` (see :ref:`Section 3 <references>`). The image region is represented by an object of :py:class:`~.image.Image` type.

        .. runblock:: pycon

           >>> import superdsm.objects
           >>> import superdsm.image
           >>> import numpy as np
           >>> y_data = np.array([[-1, -1, -1, -1, -1],
           ...                    [-1, -1, -1, -1, -1],
           ...                    [-1, -1, -1, -1, -1],
           ...                    [-1, +1, -1, -1, -1],
           ...                    [-1, +1, -1, -1, +1],
           ...                    [-1, +1, -1, -1, +1]])
           >>> atoms  = np.array([[ 1,  1,  1,  1,  1],
           ...                    [ 1,  1,  1,  1,  1],
           ...                    [ 1,  1,  1,  1,  2],
           ...                    [ 1,  1,  1,  2,  2],
           ...                    [ 1,  1,  1,  2,  2],
           ...                    [ 1,  1,  1,  2,  2]])
           >>> obj = superdsm.objects.Object()
           >>> obj.footprint = set([1])
           >>> y = superdsm.image.Image(y_data)
           >>> region = obj.get_cvxprog_region(y, atoms, background_margin=2)
           >>> region.mask
        """
        region = y.get_region(self.get_mask(atoms))
        region.mask = np.logical_and(region.mask, ndi.distance_transform_edt(y.model <= 0) <= background_margin)
        return region

    def set(self, state):
        """Adopts the state of another object.
        """
        self.fg_fragment     = state.fg_fragment.copy() if state.fg_fragment is not None else None
        self.fg_offset       = state.fg_offset.copy() if state.fg_offset is not None else None
        self.footprint       = set(state.footprint)
        self.energy          = state.energy
        self.on_boundary     = state.on_boundary
        self.is_optimal      = state.is_optimal
        self.processing_time = state.processing_time
        return self

    def copy(self):
        """Returns a deep copy of this object.
        """
        return Object().set(self)


def extract_foreground_fragment(fg_mask):
    """Returns the minimal-size rectangle region of image foreground and the corresponding offset.

    .. runblock:: pycon

       >>> import superdsm.objects
       >>> import numpy as np
       >>> mask = np.array([[False, False, False, False, False],
       ...                  [False, False, False,  True, False],
       ...                  [False, False,  True,  True, False],
       ...                  [False, False,  True, False, False]])
       >>> offset, fragment = superdsm.objects.extract_foreground_fragment(mask)
       >>> offset
       >>> fragment
    
    This function is the counterpart of the :py:meth:`~.BaseObject.fill_foreground` method.
    """
    if fg_mask.any():
        rows = fg_mask.any(axis=1)
        cols = fg_mask.any(axis=0)
        rmin, rmax  = np.where(rows)[0][[0, -1]]
        cmin, cmax  = np.where(cols)[0][[0, -1]]
        fg_offset   = np.array([rmin, cmin])
        fg_fragment = fg_mask[rmin : rmax + 1, cmin : cmax + 1]
        return fg_offset, fg_fragment
    else:
        return np.zeros(2, int), np.zeros((1, 1), bool)


def _compute_object(y, atoms, x_map, object, dsm_cfg, smooth_mat_allocation_lock):
    cp_kwargs = copy_dict(dsm_cfg)
    region = object.get_cvxprog_region(y, atoms, cp_kwargs.pop('background_margin'))
    for infoline in ('y.mask.sum()', 'region.mask.sum()', 'np.logical_and(region.model > 0, region.mask).sum()', 'cp_kwargs'):
        print(f'{infoline}: {eval(infoline)}')

    # Skip regions whose foreground is only a single pixel (this is just noise)
    if (region.model[region.mask] > 0).sum() == 1:
        object.fg_offset   = np.zeros(2, int)
        object.fg_fragment = np.zeros((1, 1), bool)
        object.energy      = 0.
        object.on_boundary = False
        object.is_optimal  = False
        object.processing_time = 0
        return object, False

    # Otherwise, perform model fitting
    else:
        t0 = time.time()
        J, result, status = cvxprog(region, smooth_mat_allocation_lock=smooth_mat_allocation_lock, **cp_kwargs)
        dt = time.time() - t0
        padded_mask = np.pad(region.mask, 1)
        smooth_mat  = uplift_smooth_matrix(J.smooth_mat, padded_mask)
        padded_foreground = (result.map_to_image_pixels(y, region, pad=1).s(x_map, smooth_mat) > 0)
        foreground = padded_foreground[1:-1, 1:-1]
        if foreground.any():
            foreground = np.logical_and(region.mask, foreground)
            object.fg_offset, object.fg_fragment = extract_foreground_fragment(foreground)
        else:
            object.fg_offset   = np.zeros(2, int)
            object.fg_fragment = np.zeros((1, 1), bool)
        object.energy      = J(result)
        object.on_boundary = padded_foreground[0].any() or padded_foreground[-1].any() or padded_foreground[:, 0].any() or padded_foreground[:, -1].any()
        object.is_optimal  = (status == 'optimal')
        object.processing_time = dt
        return object, (status == 'fallback')


@ray.remote
def _ray_compute_object_logged(*args, **kwargs):
    return _compute_object_logged(*args, **kwargs)


def _compute_object_logged(log_root_dir, cidx, *args, **kwargs):
    try:
        if log_root_dir is not None:
            log_filename = join_path(log_root_dir, f'{cidx}.txt')
            with io.TextIOWrapper(open(log_filename, 'wb', 0), write_through=True) as log_file:
                with contextlib.redirect_stdout(log_file):
                    try:
                        result = _compute_object(*args, **kwargs)
                    except:
                        traceback.print_exc(file=log_file)
                        raise
        else:
            with contextlib.redirect_stdout(None):
                result = _compute_object(*args, **kwargs)
    except CvxprogError as error:
        error.cidx = cidx
        raise
    return (cidx, *result)


DEFAULT_COMPUTING_STATUS_LINE = ('Computing objects', 'Computed objects')


def compute_objects(objects, y, atoms, dsm_cfg, log_root_dir, status_line=DEFAULT_COMPUTING_STATUS_LINE, out=None):
    """Computes the attributes of a list of objects.

    The computation concerns the attributes :py:attr:`~Object.energy`, :py:attr:`~Object.on_boundary`, :py:attr:`~Object.is_optimal`, :py:attr:`~Object.processing_time`, :py:attr:`~BaseObject.fg_fragment`, :py:attr:`~BaseObject.fg_offset` of the objects.

    :param objects: List of objects for which the above mentioned attributes are to be computed.
    :param y: An :py:class:`~.image.Image` object corresponding to the offset image intensities.
    :param atoms: Integer-valued image representing the universe of atomic image regions (each atomic image region has a unique label, which is the integer value).
    :param dsm_cfg: Dictionary of hyperparameters defined in the :py:class:`~superdsm.dsmcfg.DSM_Config` stage (without the leading ``dsm/`` namespace prefix).
    :param log_root_dir: Path of directory where log files will be written, or ``None`` if no log files should be written.
    :param status_line: Tuple ``(s1, s2)``, where ``s1`` is the line of text to be written while objects are being computed, and ``s2`` is the line of text to be written when finished.
    :param out: An instance of an :py:class:`~superdsm.output.Output` sub-class, ``'muted'`` if no output should be produced, or ``None`` if the default output should be used.
    """
    out = get_output(out)
    dsm_cfg = copy_dict(dsm_cfg)
    smooth_mat_max_allocations = dsm_cfg.pop('smooth_mat_max_allocations', np.inf)
    with SystemSemaphore('smooth-matrix-allocation', smooth_mat_max_allocations) as smooth_mat_allocation_lock:
        objects = list(objects)
        fallbacks  = 0
        x_map      = y.get_map(normalized=False, pad=1)
        for ret_idx, ret in enumerate(_compute_objects(objects, y, atoms, x_map, smooth_mat_allocation_lock, dsm_cfg, log_root_dir)):
            objects[ret[0]].set(ret[1])
            out.intermediate(f'{status_line[0]}... {ret_idx + 1} / {len(objects)} ({fallbacks}x fallback)')
            if ret[2]: fallbacks += 1
    out.write(f'{status_line[1]}: {len(objects)} ({fallbacks}x fallback)')


def _compute_objects(objects, y, atoms, x_map, lock, dsm_cfg, log_root_dir):
    if _compute_objects._DEBUG: ## run serially
        for cidx, c in enumerate(objects):
            yield _compute_object_logged(log_root_dir, cidx, y, atoms, x_map, c, dsm_cfg, lock)
    else: ## run in parallel
        y_id       = ray.put(y)
        atoms_id   = ray.put(atoms)
        x_map_id   = ray.put(x_map)
        dsm_cfg_id = ray.put(dsm_cfg)
        lock_id    = ray.put(lock)
        futures    = [_ray_compute_object_logged.remote(log_root_dir, obj_idx, y_id, atoms_id, x_map_id, obj, dsm_cfg_id, lock_id) for obj_idx, obj in enumerate(objects)]
        for ret in get_ray_1by1(futures): yield ret


_compute_objects._DEBUG = False


def _estimate_initialization(region):
    fg = region.model.copy()
    fg[~region.mask] = 0
    fg = (fg > 0)
    roi_xmap = region.get_map()
    fg_center = np.round(ndi.center_of_mass(fg)).astype(int)
    fg_center = roi_xmap[:, fg_center[0], fg_center[1]]
    halfaxes_lengths = (roi_xmap[:, fg] - fg_center[:, None]).std(axis=1)
    halfaxes_lengths = np.max([halfaxes_lengths, np.full(halfaxes_lengths.shape, 1e-8)], axis=0)
    return DeformableShapeModel.create_ellipse(np.empty(0), fg_center, *halfaxes_lengths, np.eye(2))


def _print_cvxopt_solution(solution):
    print({key: solution[key] for key in ('status', 'gap', 'relative gap', 'primal objective', 'dual objective', 'primal slack', 'dual slack', 'primal infeasibility', 'dual infeasibility')})


def _fmt_timestamp(): return time.strftime('%X')


def _print_heading(line): print(f'-- {_fmt_timestamp()} -- {line} --')


class CvxprogError(Exception):
    def __init__(self, *args, cidx=None, cause=None):
        super().__init__(*args)
        self.cidx = cidx

    def __str__(self):
        messages = [str(arg) for arg in self.args]
        if self.cidx is not None:
            messages.append(f'cidx: {self.cidx}')
        return ', '.join(messages)


def _compute_elliptical_solution(J_elliptical, CP_params):
    solution_info  = None
    solution_array = None
    solution_value = np.inf

    # Pass 1: Try zeros initialization
    try:
        solution_info  = CP(J_elliptical, np.zeros(6), **CP_params).solve()
        solution_array = DeformableShapeModel(np.array(solution_info['x'])).array
        solution_value = J_elliptical(solution_array)
        print(f'solution: {solution_value}')
    except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc()
        pass ## continue with Pass 2 (retry)

    # Pass 2: Try data-specific initialization
    if solution_info is None or solution_info['status'] != 'optimal':
        print(f'-- retry --')
        initialization = _estimate_initialization(J_elliptical.roi)
        initialization_value = J_elliptical(initialization)
        print(f'initialization: {initialization_value}')
        if initialization_value > solution_value:
            print('initialization worse than previous solution - skipping retry')
        else:
            try:
                solution_info  = CP(J_elliptical, initialization.array, **CP_params).solve()
                solution_array = DeformableShapeModel(np.array(solution_info['x'])).array
                solution_value = J_elliptical(solution_array)
                print(f'solution: {solution_value}')
            except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
                if solution_info is None:
                    cause = sys.exc_info()[1]
                    raise CvxprogError(cause)
                else:
                    pass ## continue with previous solution (Pass 1)

    assert solution_array is not None
    return solution_array


def cvxprog(region, scale, epsilon, alpha, smooth_amount, smooth_subsample, gaussian_shape_multiplier, smooth_mat_allocation_lock, smooth_mat_dtype, sparsity_tol=0, hessian_sparsity_tol=0, init=None, cachesize=0, cachetest=None, cp_timeout=None):
    """Fits a deformable shape model to the intensities of an image region.
    
    Performs convex programming in an image region :math:`X` to determine the value of the set energy function :math:`\\nu(X)` and the optimal parameters :math:`\\theta` and :math:`\\xi` (see :ref:`pipeline_theory_cvxprog` and :ref:`pipeline_theory_jointsegandclustersplit`).

    :param region: An :py:class:`~image.Image` object corresponding to the image region :math:`X`.
    :param smooth_mat_allocation_lock: A critical section lock used for allocation of the matrix :math:`\\tilde G_\\omega`.
    
    The other parameters correspond to the hyperparameters defined in the :py:class:`~superdsm.dsmcfg.DSM_Config` stage (without the leading ``dsm/`` namespace prefix).

    :return: A tuple with the following components:

        * The value of the set energy function :math:`\\nu(X)`.
        * An instance of the :py:class:`~dsm.DeformableShapeModel` class which represents the optimal parameters :math:`\\theta` and :math:`\\xi`.
        * A status indicator string, where ``optimal`` indicats that convex programming was successful and ``fallback`` indicates that convex programming failed for the deformable shape model and the initialization is used instead.
    """
    _print_heading('initializing')
    smooth_matrix_factory = SmoothMatrixFactory(smooth_amount, gaussian_shape_multiplier, smooth_subsample, smooth_mat_allocation_lock, smooth_mat_dtype)
    J = Energy(region, epsilon, alpha, smooth_matrix_factory, sparsity_tol, hessian_sparsity_tol)
    CP_params = {'cachesize': cachesize, 'cachetest': cachetest, 'scale': scale / J.smooth_mat.shape[0], 'timeout': cp_timeout}
    print(f'scale: {CP_params["scale"]:g}')
    print(f'region: {str(region.model.shape)}, offset: {str(region.offset)}')
    status = None
    if callable(init):
        params = init(J.smooth_mat.shape[1])
    else:
        if init == 'elliptical':
            _print_heading('convex programming starting: using elliptical models')
            J_elliptical = Energy(region, epsilon, alpha, SmoothMatrixFactory.NULL_FACTORY)
            params = _compute_elliptical_solution(J_elliptical, CP_params)
        else:
            params = np.zeros(6)
        params = np.concatenate([params, np.zeros(J.smooth_mat.shape[1])])
    try:
        _print_heading('convex programming starting: using deformable shape models (DSM)')
        solution_info = CP(J, params, **CP_params).solve()
        solution = np.array(solution_info['x'])
        _print_cvxopt_solution(solution_info)
        if solution_info['status'] == 'unknown' and J(solution) > J(params):
            status = 'fallback' ## numerical difficulties lead to a very bad solution, thus fall back to the elliptical solution
        else:
            print(f'solution: {J(solution)}')
            status = 'optimal'
    except: ## e.g., fetch `Rank(A) < p or Rank([H(x); A; Df(x); G]) < n` error which happens rarely
        traceback.print_exc(file=sys.stdout)
        status = 'fallback'  ## at least something we can continue the work with
    assert status is not None
    if status == 'fallback':
        _print_heading(f'DSM failed: falling back to {"elliptical result" if init == "elliptical" else "initialization"}')
        solution = params
    _print_heading('finished')
    return J, DeformableShapeModel(solution), status

