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
    
    im_norm = im

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
