import sys

import skimage.io
import ray
import superdsm.automation


if __name__ == "__main__":
    ray.init(num_cpus=1, log_to_driver=True)
    pipeline = superdsm.pipeline.create_default_pipeline()
    cfg = superdsm.config.Config()
    img = skimage.io.imread(sys.argv[1])

    superdsm.automation._estimate_scale(img, num_radii=10, thresholds=[0.01])
    cfg['AF_scale'] = None
    cfg['c2f-region-analysis/AF_min_atom_radius'] = 0.33
    cfg['c2f-region-analysismin_atom_radius/'] = 30

    from superdsm.c2freganal import C2F_RegionAnalysis
    stage = C2F_RegionAnalysis()
    stage.process(dict(g_raw = img), cfg, out = superdsm.output.get_output(), log_root_dir = None)
