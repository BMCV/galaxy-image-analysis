from ._aux import SystemSemaphore, uplift_smooth_matrix
from ._mkl import dot as mkl_dot, gram as mkl_gram

import numpy as np
import cvxopt
import skimage.util
import signal

from math         import sqrt
from scipy.linalg import orth
from scipy        import ndimage
from scipy.sparse import csr_matrix, coo_matrix, bmat as sparse_block, diags as sparse_diag, issparse


def _fast_dot(A, B):
    """Performs fast multiplication of two sparse or dense matrices (sparsity is exploited).
    """
    if A.shape[1] == B.shape[0] == 1: return A @ B
    return mkl_dot(A, B)


class DeformableShapeModel:
    """Represents a deformable shape model, defined by a set of *fixed* parameters.

    Each deformable shape model is defined by the polynomial parameters :math:`\\theta` and the deformation parameters :math:`\\xi` (see :ref:`details <pipeline_theory_dsm>`). The polynomial parameters define the polynomial surface :math:`f_x^\\top \\theta`, where :math:`f_x` is a second-order polynomial basis function expansion of the image point :math:`x`,

    .. math:: f_x = (x_1^2, x_2^2, 2 x_1 x_2, x_1, x_2, 1).
    
    Using :math:`\\theta = (a_1, a_2, a_3, b_1, b_2, c)` the polynomial surface can be written
    
    .. math:: f_x^\\top \\theta = x_1^2 a_1 + x_2^2 a_2 + 2 x_1 x_2 a_3 + x_1 b_1 + x_2 b_2 + c,

    or equivalently,
    
    .. math:: f_x^\\top \\theta = x^\\top A x + b^\\top x + c,

    where
    
    .. math:: A = \\begin{bmatrix} a_1 & a_3 \\\\ a_3 & a_2 \\end{bmatrix}, \\qquad b = \\begin{bmatrix} b_1 \\\\ b_2 \\end{bmatrix}.

    :ivar array: The vector of polynomial and deformation parameters of this model.
    :ivar a: The vector :math:`\\theta_{1:3} = (a_1, a_2, a_3)` corresponding to this model.
    :ivar b: The vector :math:`\\theta_{4:5} = b = (b_1, b_2)` corresponding to this model.
    :ivar c: The parameter :math:`\\theta_{6} = c` of this model.
    :ivar ξ: The deformation parameters :math:`\\xi` of this model.
    """
    
    def __init__(self, *args):
        if len(args) == 1 and len(args[0]) >= 6:
            self.array = args[0].astype(float).reshape(-1)
            self.a = args[0].flat[:3    ]
            self.b = args[0].flat[ 3:5  ]
            self.c = args[0].flat[   5  ]
            self.ξ = args[0].flat[    6:]
        elif len(args) >= 1:
            assert isinstance(args[0], (int, np.ndarray))
            self.ξ = np.zeros(args[0])   if isinstance(args[0], int) else args[0].reshape(-1)
            self.a = np.array([1, 1, 0]) if len(args) < 2 else args[1].flat[np.array([0, 3, 1])]
            self.b =         np.zeros(2) if len(args) < 3 else args[2].astype(float)
            self.c =                   0 if len(args) < 4 else float(args[3])
            self.array = np.concatenate([self.a, self.b, np.array([self.c]), self.ξ])
        else:
            raise ValueError('Initialization failed')
    
    @staticmethod
    def get_model(params):
        """Returns a :py:class:`DeformableShapeModel` object.
        
        If ``params`` is a :py:class:`DeformableShapeModel` object, then ``params`` is returned. Otherwise, the a new :py:class:`DeformableShapeModel` object is instantiated using the given parameters.
        """
        model = params if isinstance(params, DeformableShapeModel) else DeformableShapeModel(params)
        assert not np.isnan(model.array).any()
        return model

    def copy(self):
        """Returns a deep copy.
        """
        return DeformableShapeModel(self.array.copy())
    
    @property
    def A(self):
        """Returns the matrix :math:`A` corresponding to this model.
        """
        return np.array([self.a[0], self.a[2], self.a[2], self.a[1]]).reshape((2, 2))
    
    def s(self, x, smooth_mat):
        """Computes the deformable surface :math:`S_\omega(\\theta, \\xi)` as described in :ref:`pipeline_theory_dsm`.

        :param x: Either a list of coordinates of those image points for which the values of the deformable surface are to be computed, or a stack of two 2D arrays corresponding to the pixel coordinates.
        """
        xdim = x.ndim - 1 if isinstance(x, np.ndarray) else 0
        xvec = np.array(x).reshape((2, -1))
        svec = _diagquad(self.A, xvec) + 2 * np.inner(xvec.T, self.b) + self.c + _fast_dot(smooth_mat, self.ξ)
        return svec.reshape(x.shape[-xdim:]) if isinstance(x, np.ndarray) else svec
    
    @staticmethod
    def create_ellipse(ξ, center, halfaxis1_len, halfaxis2_len, U=None):
        """Creates a deformable shape model corresponding to an ellipse, possibly deformbed.

        :param ξ: The deformation parameters.
        :param center: The coordinates of the center of the ellipse.
        :param halfaxis1_len: The length of the first half axis.
        :param halfaxis2_len: The length of the second half axis.
        :param U: An orthonormal matrix whose eigenvectors define the rotation of the ellipse, or ``None`` if a random rotation should be used.
        """
        ev = lambda half_length: (1. / np.square(half_length))
        if U is None: U = orth(np.random.randn(2, 2)) # random rotation matrix
        A  = U.dot(np.diag((ev(halfaxis1_len), ev(halfaxis2_len)))).dot(U.T)
        b  = A.dot(center)
        c  = np.inner(center, b) - 1
        return DeformableShapeModel(ξ, -A, b, -c)
    
    def map_to_image_pixels(self, g, roi, pad=0):
        """Transforms the model from the coordinate system of an image region into the coordinate system of the whole image.

        :param g: An :py:class:`~superdsm.image.Image` object corresponding to the whole image.
        :param roi: An :py:class:`~superdsm.image.Image` object corresponding to the image region.
        :param pad: The number of pixels by which the coordinate system of the whole image is padded (once in each direction, i.e. twice along each axis). Using a value larger than 0 yields the same result as using an image ``g`` padded by the same value.
        :return: The transformed deformable shape model.
        """
        assert pad >= 0 and isinstance(pad, int)
        g_max_coord, roi_max_coord = 2 * pad + np.array(g.model.shape) - 1., np.array(roi.model.shape) - 1.
        G = np.diag(1. / roi_max_coord)
        v = -G.dot(np.add(roi.offset, pad))
        A = G.dot(self.A).dot(G)
        b = G.dot(self.A.dot(v) + self.b)
        c = np.inner(v, self.A.dot(v)) + 2 * np.inner(self.b, v) + self.c
        return DeformableShapeModel(self.ξ, A, b, c)


def _diagquad(A, X):
    """Computes the diagonal entries of :math:`X^\\top A X` quickly.
    """
    return np.einsum('ij,ij->i', np.dot(X.T, A), X.T)


def _create_gaussian_kernel(sigma, shape=None, shape_multiplier=1):
    if abs(shape_multiplier - 1) > 0 and shape is not None: raise ValueError()
    if shape is None: shape = [round(1 + sigma * 4 * shape_multiplier)] * 2
    inp = np.zeros(shape)
    inp[shape[0] // 2, shape[1] // 2] = 1
    return ndimage.gaussian_filter(inp, sigma)


def _convmat(filter_mask, img_shape, row_mask=None, col_mask=None, lock=None):
    assert filter_mask.ndim == 2 and filter_mask.shape[0] == filter_mask.shape[1]
    assert filter_mask.shape[0] % 2 == 1, filter_mask.shape[0]
    if row_mask is None: row_mask = np.ones(img_shape, bool)
    if col_mask is None: col_mask = np.ones(img_shape, bool)
    print('.', end='')
    p = np.subtract(img_shape, filter_mask.shape[0] // 2 + 1)
    assert (p >= 0).all(), f'filter_mask {filter_mask.shape} too large for img_shape {img_shape}'
    print('.', end='')
    z = np.pad(filter_mask, np.vstack([p, p]).T)
    print('.', end='')
    z = skimage.util.view_as_windows(z, img_shape)[::-1, ::-1]
    print('.', end='\n')
    with SystemSemaphore.get_lock(lock):
        col_mask_where = np.nonzero(col_mask)
        row_mask_where = np.nonzero(row_mask)
        return z[row_mask_where[0][:,None], row_mask_where[1][:,None], col_mask_where[0], col_mask_where[1]]


def _create_subsample_grid(mask, subsample, mask_offset=(0,0)):
    grid_offset = np.asarray(mask_offset) % subsample
    subsample_grid = np.zeros_like(mask)
    subsample_grid[grid_offset[0]::subsample, grid_offset[1]::subsample] = True
    subsample_grid = np.logical_and(mask, subsample_grid)
    distances = mask * ndimage.distance_transform_bf(~subsample_grid, metric='chessboard')
    tmp1 = np.ones_like(subsample_grid, bool)
    while True:
        outside = (distances >= subsample)
        if not outside.any(): break
        min_outside_distance = distances[outside].min()
        min_outside_pixel = tuple(np.asarray(np.where(distances == min_outside_distance)).T[0])
        subsample_grid[min_outside_pixel] = True
        tmp1[min_outside_pixel] = False
        tmp2 = ndimage.distance_transform_bf(tmp1, metric='chessboard')
        distances = np.min((distances, tmp2), axis=0)
        tmp1[min_outside_pixel] = True
    return subsample_grid


def _create_masked_smooth_matrix(kernel, mask, subsample=1, lock=None):
    mask = mask[np.where(mask.any(axis=1))[0], :]
    mask = mask[:, np.where(mask.any(axis=0))[0]]
    if (mask.shape <= np.asarray(kernel.shape) // 2).any(): return None
    subsample_grid = _create_subsample_grid(mask, subsample)
    col_mask = np.logical_and(mask, subsample_grid)
    print(f'{mask.sum()} rows, {col_mask.sum()} columns')
    M = _convmat(kernel, mask.shape, row_mask=mask, col_mask=col_mask, lock=lock)
    M_sums = M.sum(axis=1)
    M /= M_sums[:, None]
    assert (M.sum(axis=0) > 0).all() and (M.sum(axis=1) > 0).all()
    return M


class SmoothMatrixFactory:
    """Instantiates the matrix :math:`\\tilde G_\\omega` for any image region :math:`\\omega`.

    The matrix :math:`\\tilde G_\\omega` is the sub-sampled variant of the :math:`G_\\omega` matrix.

    :param smooth_amount: This is :math:`\\sigma_G` described in :ref:`pipeline_theory_dsm`.
    :param shape_multiplier: The Gaussian function with standard deviation :math:`\\sigma_G` used to construct the block Toeplitz matrix :math:`G_\\omega` is cut off after :math:`4 \\sigma_G` multiplied by this value (see :ref:`pipeline_theory_dsm`).
    :param smooth_subsample: Corresponds to the amount of sub-sampling used (see Section 3.3 in the :ref:`paper <references>`).
    :param lock: A critical section lock used for allocation of the matrix.
    :param dtype: The data type used for the matrix.
    """

    def __init__(self, smooth_amount, shape_multiplier, smooth_subsample, lock=None, dtype='float32'):
        self.smooth_amount    = smooth_amount
        self.shape_multiplier = shape_multiplier
        self.smooth_subsample = smooth_subsample
        self.lock             = lock
        self.dtype            = dtype

    def get(self, mask, uplift=False):
        """Yields the matrix :math:`\\tilde G_\\omega` for an image region :math:`\\omega`.
        
        :param mask: The image region :math:`\\omega` represented as a binary mask.
        :param uplift: Currently not used.
        """
        print('-- smooth matrix computation starting --')
        mat = None
        if self.smooth_amount < np.inf:
            psf = _create_gaussian_kernel(self.smooth_amount, shape_multiplier=self.shape_multiplier).astype(self.dtype)
            mat = _create_masked_smooth_matrix(psf, mask, self.smooth_subsample, self.lock)
            # NOTE: `mat` will be `None` if `psf` is too large for `mask`
        if mat is None:
            print('using null-matrix')
            mat = np.empty((mask.sum(), 0))
        mat = csr_matrix(mat).astype(np.float64, copy=False)
        if uplift: mat = uplift_smooth_matrix(mat, mask)
        print('-- smooth matrix finished --')
        return mat
    
SmoothMatrixFactory.NULL_FACTORY = SmoothMatrixFactory(np.inf, np.nan, np.nan)
"""Instantiates the matrix :math:`\\tilde G_\\sigma` as a matrix with zero columns (i.e. deformations are not permitted).
"""
    

def _compute_polynomial_derivatives(x_map):
    derivatives = [None] * 6
    derivatives[0] = np.square(x_map[0])
    derivatives[1] = np.square(x_map[1])
    derivatives[2] = 2 * np.prod([x_map[i] for i in range(2)], axis=0)
    derivatives[3] = 2 * x_map[0]
    derivatives[4] = 2 * x_map[1]
    derivatives[5] = 1
    return derivatives


class Energy:
    """Represents the convex energy function :math:`\\psi` for deformable shape models.

    Instances of this class can be used as functions (e.g., ``energy(params)`` computes the value :math:`\\psi_\\omega(\\theta, \\xi)` of the convex energy function :math:`\\psi` so that ``params[:6]`` corresponds to the polynomial parameters :math:`\\theta` and ``params[6:]`` corresponds to the deformation parameters :math:`\\xi`).

    :param roi: An image region represented by an instance of the :py:class:`~superdsm.image.Image` class.
    :param epsilon: Corresponds to the constant :math:`\\epsilon` which is used for the smooth approximation of the regularization term :math:`\\|\\xi\\|_1 \\approx \\mathbb 1^\\top_\\Omega \\sqrt{\\xi^2 + \\epsilon} - \\sqrt{\\epsilon} \\cdot \\#\\Omega` (see Supplemental Material 2 of the :ref:`paper <references>`).
    :param alpha: Governs the regularization of the deformations and corresponds to :math:`\\alpha` described in :ref:`pipeline_theory_cvxprog`. Increasing this value leads to a smoother segmentation result.
    :param smooth_matrix_factory: An object with a ``get`` method which yields the matrix :math:`\\tilde G_\\omega` for any image region :math:`\\omega` (represented as a binary mask and passed as a parameter).
    :param sparsity_tol: Absolute values below this threshold will be treated as zeros for computation of the gradient.
    :param hessian_sparsity_tol: Absolute values below this threshold will be treated as zeros for computation of the Hessian.
    """

    def __init__(self, roi, epsilon, alpha, smooth_matrix_factory, sparsity_tol=0, hessian_sparsity_tol=0):
        self.roi = roi
        self.p   = None

        self.smooth_mat = smooth_matrix_factory.get(roi.mask)

        self.x = self.roi.get_map()[:, roi.mask]
        self.w = np.ones(roi.mask.sum(), 'uint8')
        self.y = roi.model[roi.mask]

        assert epsilon > 0, 'epsilon must be strictly positive'
        self.epsilon = epsilon

        assert alpha >= 0, 'alpha must be positive'
        self.alpha = alpha

        assert sparsity_tol >= 0, 'sparsity_tol must be positive'
        self.sparsity_tol = sparsity_tol

        assert hessian_sparsity_tol >= 0, 'hessian_sparsity_tol must be positive'
        self.hessian_sparsity_tol = hessian_sparsity_tol

        # pre-compute common terms occuring in the computation of the derivatives
        self.q = _compute_polynomial_derivatives(self.x)
    
    def _update_maps(self, params):
        if self.p is not None and all(self.p.array == params.array): return
        s = params.s(self.x, self.smooth_mat)
        self.p     = params
        self.t     = self.y * s
        self.theta = None # invalidate
        
        valid_t_mask = (self.t >= -np.log(np.finfo(self.t.dtype).max))
        self.h = np.full(self.t.shape, np.nan)
        self.h[valid_t_mask] = np.exp(-self.t[valid_t_mask])

        if self.smooth_mat.shape[1] > 0:
            self.term3 = np.square(params.ξ)
            self.term2 = np.sqrt(self.term3 + self.epsilon)
    
    def _update_theta(self):
        if self.theta is None:
            valid_h_mask = ~np.isnan(self.h)
            self.theta = np.ones_like(self.t)
            self.theta[valid_h_mask] = self.h[valid_h_mask] / (1 + self.h[valid_h_mask])
    
    def __call__(self, params):
        """Computes the value :math:`\\psi_\\omega(\\theta, \\xi)` of the convex energy function.

        The parameters are represented by ``params`` so that ``params[:6]`` corresponds to the polynomial parameters :math:`\\theta` and ``params[6:]`` corresponds to the deformation parameters :math:`\\xi`.
        """
        params = DeformableShapeModel.get_model(params)
        self._update_maps(params)
        valid_h_mask = ~np.isnan(self.h)
        phi = np.zeros_like(self.t)
        phi[ valid_h_mask] = np.log(1 + self.h[valid_h_mask])
        phi[~valid_h_mask] = -self.t[~valid_h_mask]
        objective1 = np.inner(self.w.flat, phi.flat)
        if self.smooth_mat.shape[1] > 0:
            objective2  = self.alpha * self.term2.sum()
            objective2 -= self.alpha * sqrt(self.epsilon) * len(self.term2)
            if objective2 < 0:
                assert np.allclose(0, objective2)
                objective2 = 0
            else:
                assert objective2 >= 0
        else:
            objective2 = 0
        return objective1 + objective2
    
    def grad(self, params):
        """Computes the gradient vector :math:`\\nabla \\psi_\\omega(\\theta, \\xi)`.

        The parameters are represented by ``params`` so that ``params[:6]`` corresponds to the polynomial parameters :math:`\\theta` and ``params[6:]`` corresponds to the deformation parameters :math:`\\xi`.
        """
        params = DeformableShapeModel.get_model(params)
        self._update_maps(params)
        self._update_theta()
        term1 = -self.y * self.theta
        grad = np.asarray([term1 * q for q in self.q]) @ self.w
        term1[abs(term1) < self.sparsity_tol] = 0
        term1_sparse = coo_matrix(term1).transpose(copy=False)
        if self.smooth_mat.shape[1] > 0:
            grad2  = (self.w.reshape(-1)[None, :] @ self.smooth_mat.multiply(term1_sparse)).reshape(-1)
            grad2 += self.alpha * (params.ξ / self.term2)
            grad   = np.concatenate([grad, grad2])
        return grad
    
    def hessian(self, params):
        """Computes the Hessian matrix :math:`\\nabla^2 \\psi_\\omega(\\theta, \\xi)`.

        The parameters are represented by ``params`` so that ``params[:6]`` corresponds to the polynomial parameters :math:`\\theta` and ``params[6:]`` corresponds to the deformation parameters :math:`\\xi`. The Hessian matrix is returned as a sparse block matrix.
        """
        params = DeformableShapeModel.get_model(params)
        self._update_maps(params)
        self._update_theta()
        kappa = self.theta - np.square(self.theta)
        kappa[kappa < self.sparsity_tol] = 0
        pixelmask = (kappa != 0)
        term4 = np.sqrt(kappa[pixelmask] * self.w[pixelmask])[None, :]
        D1 = np.asarray([-self.y * qi for qi in self.q])[:, pixelmask] * term4
        D2 = self.smooth_mat[pixelmask].multiply(-self.y[pixelmask, None]).T.multiply(term4).tocsr()
        if self.smooth_mat.shape[1] > 0:
            H = sparse_block([
                [D1 @ D1.T, csr_matrix((D1.shape[0], D2.shape[0]))],
                [_fast_dot(D2, D1.T), mkl_gram(D2).T if D2.shape[1] > 0 else csr_matrix((D2.shape[0], D2.shape[0]))]])
            g = self.alpha * (1 / self.term2 - self.term3 / np.power(self.term2, 3))
            assert np.allclose(0, g[g < 0])
            g[g < 0] = 0
            H += sparse_diag(np.concatenate([np.zeros(6), g]))
            if self.hessian_sparsity_tol > 0:
                H = H.tocoo()
                H_mask = (np.abs(H.data) >= self.hessian_sparsity_tol)
                H_mask = np.logical_or(H_mask, H.row == H.col)
                H.data = H.data[H_mask]
                H.row  = H.row [H_mask]
                H.col  = H.col [H_mask]
        else:
            H = D1 @ D1.T
        return H


class _Cache:

    def __init__(self, size, getter, equality=None):
        if equality is None: equality = np.array_equal
        elif isinstance(equality, str): equality = eval(equality)
        assert callable(equality)
        self.size     = size
        self.inputs   = []
        self.outputs  = []
        self.getter   = getter
        self.equality = equality

    def __call__(self, input):
        pos = -1
        for i in range(len(self.inputs))[::-1]:
            input2 = self.inputs[i]
            if self.equality(input, input2):
                pos = i
                input = input2
                break
        if pos > -1:
            output = self.outputs[pos]
            del self.inputs[pos], self.outputs[pos]
        else:
            output = self.getter(input)
        self.inputs .append(input)
        self.outputs.append(output)
        assert len(self.inputs) == len(self.outputs)
        if len(self.inputs) > self.size:
            del self.inputs[0], self.outputs[0]
        return output


class TimeoutError(Exception):
    """Indicates that an operation has timed out.
    """
    pass


def _cp_timeout_handler(*args):
    raise TimeoutError()


class CP:
    """Represents the convex problem of minimizing a convex energy function :math:`\\psi`.

    :param energy: The convex energy function to be minimized, so that ``energy(p)`` corresponds to the value of :math:`\\psi(p)`, ``energy.grad(p)`` corresponds to the gradient vector :math:`\\nabla\\psi(p)`, and ``energy.hessian(p)`` corresponds to the Hessian matrix :math:`\\nabla^2 \\psi(p)`.
    :param params0: Parameters vector used for initialization.
    :param scale: Fixed factor used to slightly improve numerical stabilities.
    :param cachesize: The maximum number of entries used for caching the values of the ``energy`` function, the gradient, and the Hessian.
    :param cachetest: The test function to be used for cache testing. If ``None``, then ``numpy.array_equal`` will be used.
    :param timeout: The maximum run time of the :py:meth:`~.solve` method (in seconds). Convex programming will be interrupted by raising a :py:class:`~TimeoutError` if it takes longer than that. If this is set to ``None``, the run time is not limited.
    """

    CHECK_NUMBERS = False
    """Performs additional assertions when set to ``True`` which are useful for debugging but might increase the overall run time (sparse matrices need to be converted to dense).
    """

    def __init__(self, energy, params0, scale=1, cachesize=0, cachetest=None, timeout=None):
        self.params0  = params0
        self.gradient = _Cache(cachesize, lambda p: (scale * energy(p), cvxopt.matrix(scale * energy.grad(p)).T), equality=cachetest)
        self.hessian  = _Cache(cachesize, lambda p:  scale * energy.hessian(p), equality=cachetest)
        self.timeout  = timeout
    
    def __call__(self, params=None, w=None):
        if params is None:
            return 0, cvxopt.matrix(self.params0)
        else:
            p = np.array(params).reshape(-1)
            l, Dl = self.gradient(p)
            if CP.CHECK_NUMBERS:
                Dl_array = np.array(Dl)
                assert not np.isnan(p).any() and not np.isinf(p).any()
                assert not np.isnan(Dl_array).any() and not np.isinf(Dl_array).any()
            if w is None:
                return l, Dl
            else:
                H = self.hessian(p)
                if issparse(H):
                    if CP.CHECK_NUMBERS:
                        H_array = H.toarray()
                        assert not np.isnan(H_array).any() and not np.isinf(H_array).any()
                    H = H.tocoo()
                    H = cvxopt.spmatrix(w[0] * H.data, H.row, H.col, size=H.shape)
                else:
                    if CP.CHECK_NUMBERS:
                        assert not np.isnan(H).any() and not np.isinf(H).any()
                    H = cvxopt.matrix(w[0] * H)
                return l, Dl, H
    
    def solve(self, **options):
        """Performs convex programming.

        :param options: Currently not used.
        """
        alarm_set = False
        if self.timeout is not None and self.timeout > 0:
            signal.signal(signal.SIGALRM, _cp_timeout_handler)
            signal.alarm(self.timeout)
            alarm_set = True
        ret = cvxopt.solvers.cp(self)
        if alarm_set: signal.alarm(0)
        return ret

