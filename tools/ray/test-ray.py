import sys

import skimage.io
import ray

import math
import scipy.ndimage as ndi
import numpy as np


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    img = skimage.io.imread(sys.argv[1])

    sigma_list = np.linspace(20, 200, 10) / math.sqrt(2)
    sigma_list = np.concatenate([[sigma_list.min() / 2], sigma_list])
   
    for sigma in sigma_list:
        ndi.gaussian_laplace(img, sigma)

    ray.put(img)
