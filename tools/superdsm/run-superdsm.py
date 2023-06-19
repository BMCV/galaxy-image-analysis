"""
Copyright 2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile


def create_task(image_filepath, num_cpus, num_threads):
    return {
        'runnable': True,
        'num_cpus': int(num_cpus),
        'environ': {
            'MKL_NUM_THREADS': int(num_threads),
            'OPENBLAS_NUM_THREADS': int(num_threads),
        },
        'img_pathpattern': str(image_filepath),
        'seg_pathpattern': 'masks.png',
        'cfg_pathpattern': 'cfg.json',
        'overlay_pathpattern': 'overlay.png',
        'file_ids': [[]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation of cell nuclei in 2-D fluorescence microscopy images')
    parser.add_argument('image', help='Path to the input image')
    parser.add_argument('cfg', help='Path to the file containing the configuration')
    parser.add_argument('masks', help='Path to the file containing the segmentation masks')
    parser.add_argument('overlay', help='Path to the file containing the overlay of the segmentation results')
    parser.add_argument('num_cpus', type=int)
    parser.add_argument('num_threads', type=int)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        task = create_task(num_cpus=args.num_cpus, num_threads=args.num_threads, image_filepath=args.image)
        with open(str(tmpdir / 'task.json'), 'w') as fp:
            json.dump(task, fp)
        subprocess.run(['python', '-m', 'superdsm.batch', tmpdirname, '--run'], check=True)
        shutil.copy(str(tmpdir / 'masks.png'), args.masks)
        shutil.copy(str(tmpdir / 'cfg.json'), args.cfg)
        shutil.copy(str(tmpdir / 'overlay.png'), args.overlay)
