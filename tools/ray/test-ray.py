import ray

import scipy.ndimage as ndi
import numpy as np


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    img = np.zeros((1344, 1024))

    for sigma in np.linspace(100, 200, 5) / np.sqrt(2):
        ndi.gaussian_laplace(img, sigma)

    ray.put(img)
