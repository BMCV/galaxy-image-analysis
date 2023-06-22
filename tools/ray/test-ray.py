import sys

import skimage.io
import ray
import superdsm.automation


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    img = skimage.io.imread(sys.argv[1])

    superdsm.automation._estimate_scale(img, num_radii=10, thresholds=[0.01])
    ray.put(img)
