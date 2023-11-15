"""
Copyright 2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import csv
import imghdr
import os
import pathlib
import shutil
import tempfile


hyperparameters = [
    ('AF_scale', float),
    ('c2f_region_analysis/min_atom_radius', float),
    ('c2f_region_analysis_min_norm_energy_improvement', float),
    ('c2f_region_analysis_max_atom_norm_energy', float),
    ('c2f_region_analysis_max_cluster_marker_irregularity', float),
    ('dsm_alpha', float),
    ('dsm_AF_alpha', float),
    ('global_energy_minimization_betai', float),
    ('global_energy_minimization_AF_beta', float),
    ('postprocess_mask_max_distance', int),
    ('postprocess_mask_stdamp', float),
    ('postprocess_max_norm_energy', float),
    ('postprocess_min_contrast', float),
    ('postprocess_min_object_radius', float),
]


def get_param_name(key):
    return key.replace('/', '_')


def create_config(args):
    cfg = superdsm.config.Config()
    for key, _ in hyperparameters:
        value = getattr(args, get_param_name(key))
        if value is not None:
            cfg[key] = value
    return cfg


def flatten_dict(d, sep='/'):
    result = {}
    for key, val in d.items():
        if isinstance(val, dict):
            for sub_key, sub_val in flatten_dict(val, sep=sep).items():
                result[f'{key}{sep}{sub_key}'] = sub_val
        else:
            result[key] = val
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation of cell nuclei in 2-D fluorescence microscopy images')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('slots', type=int)
    parser.add_argument('--do-masks', type=str, default=None, help='Path to the file containing the segmentation masks')
    parser.add_argument('--do-cfg', type=str, default=None, help='Path to the file containing the configuration')
    parser.add_argument('--do-overlay', type=str, default=None, help='Path to the file containing the overlay of the segmentation results')
    parser.add_argument('--do-overlay-border', type=int)
    for key, ptype in hyperparameters:
        parser.add_argument('--' + get_param_name(key), type=ptype, default=None)
    args = parser.parse_args()

    if args.slots >= 2:
        num_threads_per_process = 2
        num_processes = args.slots // num_threads_per_process
    else:
        num_threads_per_process = 1
        num_processes = 1

    os.environ['MKL_NUM_THREADS'] = str(num_threads_per_process)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads_per_process)

    import ray
    import superdsm.automation
    import superdsm.io
    import superdsm.render

    ray.init(num_cpus=num_processes, log_to_driver=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        img_ext = imghdr.what(args.image)
        img_filepath = tmpdir / f'input.{img_ext}'
        shutil.copy(str(args.image), img_filepath)

        pipeline = superdsm.pipeline.create_default_pipeline()
        cfg = create_config(args)
        img = superdsm.io.imread(img_filepath)

        if args.do_cfg:
            print(f'Writing config to: {args.do_cfg}')
            cfg, _ = superdsm.automation.create_config(pipeline, cfg, img)
            with open(args.do_cfg, 'w') as fp:
                tsv_out = csv.writer(fp, delimiter='\t')
                tsv_out.writerow(['Hyperparameter', 'Value'])
                for key, value in flatten_dict(cfg.entries).items():
                    tsv_out.writerow([key, value])

        if args.do_overlay or args.do_masks:
            print(f'Performing segmentation')
            data, cfg, _ = pipeline.process_image(img, cfg)

        if args.do_overlay:
            print(f'Writing overlay to: {args.do_overlay}')
            overlay = superdsm.render.render_result_over_image(data, border_width=args.do_overlay_border, normalize_img=False)
            superdsm.io.imwrite(args.do_overlay, overlay)

        if args.do_masks:
            print(f'Writing masks to: {args.do_masks}')
            masks = superdsm.render.rasterize_labels(data)
            superdsm.io.imwrite(args.do_masks, masks)
