import skimage
import math
import scipy.ndimage as ndi
import numpy as np


_max = max
_min = min


def _estimate_scale(im, min_radius=20, max_radius=200, num_radii=10, thresholds=[0.01], inlier_tol=np.inf):
    """Estimates the scale of the image.
    """

    sigma_list = np.linspace(min_radius, max_radius, num_radii) / math.sqrt(2)
    sigma_list = np.concatenate([[sigma_list.min() / 2], sigma_list])
    
    blobs_mask = {sigma: ndi.gaussian_laplace(im, sigma) < 0 for sigma in sigma_list}
