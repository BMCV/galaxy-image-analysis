import sys

import skimage.io
import ray
import superdsm.automation
import superdsm.io
import superdsm.render


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    pipeline = superdsm.pipeline.create_default_pipeline()
    cfg = superdsm.config.Config()
    img = skimage.io.imread(sys.argv[1])
    data, cfg, _ = superdsm.automation.process_image(pipeline, cfg, img)
