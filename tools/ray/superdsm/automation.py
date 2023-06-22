from .render import normalize_image

import skimage
import math
import scipy.ndimage as ndi
import numpy as np


_max = max
_min = min


def _blob_doh(image, sigma_list, threshold=0.01, overlap=.5, mask=None):
    """Finds blobs in the given grayscale image.

    This implementation is widely based on:
    https://github.com/scikit-image/scikit-image/blob/fca9f16da4bd7420245d05fa82ee51bb9677b039/skimage/feature/blob.py#L538-L646
    """
    skimage.feature.blob.check_nD(image, 2)
    if mask is None: mask = np.ones(image.shape, bool)
    if not isinstance(mask, dict): mask = {sigma: mask for sigma in sigma_list}

    image = skimage.feature.blob.img_as_float(image)
    image = skimage.feature.blob.integral_image(image)

    hessian_images = [mask[s] * skimage.feature.blob._hessian_matrix_det(image, s) for s in sigma_list]
    image_cube = np.dstack(hessian_images)

    local_maxima = skimage.feature.blob.peak_local_max(image_cube, threshold_abs=threshold,
                                                       footprint=np.ones((3,) * image_cube.ndim),
                                                       threshold_rel=0.0,
                                                       exclude_border=False)

    if local_maxima.size == 0:
        return np.empty((0, 3))
    lm = local_maxima.astype(np.float64)
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return skimage.feature.blob._prune_blobs(lm, overlap)


def _estimate_scale(im, min_radius=20, max_radius=200, num_radii=10, thresholds=[0.01], inlier_tol=np.inf):
    """Estimates the scale of the image.
    """

    sigma_list = np.linspace(min_radius, max_radius, num_radii) / math.sqrt(2)
    sigma_list = np.concatenate([[sigma_list.min() / 2], sigma_list])
    
    im_norm  = normalize_image(im)
    im_norm /= im_norm.max()

    blobs_mask  = {sigma: ndi.gaussian_laplace(im_norm, sigma) < 0 for sigma in sigma_list}
    mean_radius = None
    for threshold in sorted(thresholds, reverse=True):
        blobs_doh = _blob_doh(im_norm, sigma_list, threshold=threshold, mask=blobs_mask)
        blobs_doh = blobs_doh[~np.isclose(blobs_doh[:,2], sigma_list.min())]
        if len(blobs_doh) == 0: continue

        radii = blobs_doh[:,2] * math.sqrt(2)
        radii_median  = np.median(radii)
        radii_mad     = np.mean(np.abs(radii - np.median(radii)))
        radii_bound   = np.inf if np.isinf(inlier_tol) else radii_mad * inlier_tol
        radii_inliers = np.logical_and(radii >= radii_median - radii_mad, radii <= radii_median + radii_mad)
        mean_radius   = np.mean(radii[radii_inliers])
        break
    
    if mean_radius is None:
        raise ValueError('scale estimation failed')
    return mean_radius / math.sqrt(2), blobs_doh, radii_inliers


def _create_config_entry(cfg, key, factor, default_user_factor, type=None, min=None, max=None):
    keys = key.split('/')
    af_key = f'{"/".join(keys[:-1])}/AF_{keys[-1]}'
    cfg.set_default(key, factor * cfg.get(af_key, default_user_factor), True)
    if type is not None: cfg.update(key, func=type)
    if  min is not None: cfg.update(key, func=lambda value: _max((value, min)))
    if  max is not None: cfg.update(key, func=lambda value: _min((value, max)))


def create_config(pipeline, base_cfg, img):
    """Automatically configures hyperparameters based on the scale of objects in an image. 

    The scale of the objects is estimated automatically as described in Section 3.1 of the paper (:ref:`Kostrykin and Rohr, 2023 <references>`). The current implementation determines values corresponding to object radii between 20 and 200 pixels. If, however, the hyperparameter ``AF_sigma`` is set, then the scale :math:`\sigma` is forced to its value and the automatic scale detection is not used. The hyperparameter ``AF_sigma`` is not set by default.

    .. runblock:: pycon

       >>> import superdsm, superdsm.automation, superdsm.config
       >>> base_cfg = superdsm.config.Config(dict(AF_scale=40))
       >>> pipeline = superdsm.pipeline.create_default_pipeline()
       >>> cfg, _ = superdsm.automation.create_config(pipeline, base_cfg, None)
       >>> print(cfg)
    """
    cfg   = base_cfg.copy()
    scale = cfg.get('AF_scale', None)
    if scale is None: scale = _estimate_scale(img, num_radii=10, thresholds=[0.01])[0]
    for stage in pipeline.stages:
        specs = stage.configure(scale)
        for key, spec in specs.items():
            assert len(spec) in (2,3), f'{type(stage).__name__}.configure returned tuple of unknown length ({len(spec)})'
            kwargs = dict() if len(spec) == 2 else spec[-1]
            _create_config_entry(cfg, f'{stage.cfgns}/{key}', *spec[:2], **kwargs)
    return cfg, scale


def process_image(pipeline, base_cfg, g_raw, **kwargs):
    """Performs the segmentation of an image using hyperparameters automatically configured based on the scale of objects.

    See the :py:meth:`~.create_config` function and the :py:meth:`~superdsm.pipeline.Pipeline.process_image` pipeline method for details.

    :param pipeline: The :py:class:`~superdsm.pipeline.Pipeline` object to be used for image segmentation.
    :param base_cfg: :py:class:`~superdsm.config.Config` object corresponding to custom hyperparameters.
    :param g_raw: A ``numpy.ndarray`` object corresponding to the image which is to be processed.
    :param kwargs: Additional keyword arguments passed to the :py:meth:`~superdsm.pipeline.Pipeline.process_image` pipeline method.
    :return: The same tuple that is returned by the :py:meth:`~superdsm.pipeline.Pipeline.process_image` pipeline method.
    """
    cfg, _ = create_config(pipeline, base_cfg, g_raw)
    return pipeline.process_image(g_raw, cfg=cfg, **kwargs)

