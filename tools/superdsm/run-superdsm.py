"""
Copyright 2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import imghdr
import pathlib
import shutil
import tempfile

import ray
import superdsm.automation
import superdsm.io
import superdsm.render


# It was not possible to get ray to work in conjunction with planemo, though the reason remains
# unclear. Most of the time, the attempts failed due to out-of-memory issues, although ray
# shouldn't consume any significant amount of memory. Thus, below, ray is effectively "patched
# out" of SuperDSM. This is neither efficient nor clean, but at least it works.


class _Future:
    def __init__(self, getter):
        self.getter = getter


class _ray_remote:
    def __init__(self, func):
        self.func = func

    def remote(self, *args, **kwargs):
        args = [(arg.getter() if isinstance(arg, _Future) else arg) for arg in args]
        kwargs = {key: (arg.getter() if isinstance(arg, _Future) else arg) for key, arg in kwargs.items()}
        return _Future(lambda: self.func(*args, **kwargs))


def _ray_wait(futures, num_returns):
    return futures[:num_returns], futures[num_returns:]


def _ray_put(obj):
    return _Future(lambda: obj)


def _ray_get(future):
    return future.getter()


ray.remote = _ray_remote
ray.wait = _ray_wait
ray.put = _ray_put
ray.get = _ray_get


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation of cell nuclei in 2-D fluorescence microscopy images')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('cfg', help='Path to the file containing the configuration')
    parser.add_argument('masks', help='Path to the file containing the segmentation masks')
    parser.add_argument('overlay', help='Path to the file containing the overlay of the segmentation results')
    parser.add_argument('seg_border', type=int)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        img_ext = imghdr.what(args.image)
        img_filepath = tmpdir / f'input.{img_ext}'
        shutil.copy(str(args.image), img_filepath)

        ray.init(local_mode=True, log_to_driver=True)
        pipeline = superdsm.pipeline.create_default_pipeline()
        cfg = superdsm.config.Config()
        img = superdsm.io.imread(img_filepath)
        data, cfg, _ = superdsm.automation.process_image(pipeline, cfg, img)

        with open(args.cfg, 'w') as fp:
            cfg.dump_json(fp)

        overlay = superdsm.render.render_result_over_image(data, border_width=args.seg_border, normalize_img=False)
        superdsm.io.imwrite(args.overlay, overlay)

        masks = superdsm.render.rasterize_labels(data)
        superdsm.io.imwrite(args.masks, masks)
