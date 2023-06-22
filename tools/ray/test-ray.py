import ray

import numpy as np
import scipy.ndimage as ndi


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    img = np.zeros((1344, 1024))

    for _ in range(3):
        ndi.gaussian_laplace(img, 200)

    ray.put(img)
