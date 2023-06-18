"""
Copyright 2022-2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import pathlib
import subprocess
import tempfile
import zipfile

import pandas as pd


def process_batch(seg_dir, seg_file, gt_file, tsv_output_file, recursive, gt_unique, seg_unique, measures):
    with tempfile.NamedTemporaryFile() as csv_output_file:
        cmd = ['python', '-m', 'segmetrics.cli', str(seg_dir), str(seg_file), str(gt_file), str(csv_output_file.name), '--semicolon']
        if recursive:
            cmd.append('--recursive')
        if gt_unique:
            cmd.append('--gt-unique')
        if seg_unique:
            cmd.append('--seg-unique')
        cmd += measures
        print(cmd)
        subprocess.run(cmd, check=True)
        df = pd.read_csv(csv_output_file.name, sep=';')
        df.to_csv(str(tsv_output_file), sep='\t', index=False)
        import shutil
        shutil.copy(str(tsv_output_file), '/tmp/segmetrics-results.tsv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image segmentation and object detection performance measures for 2-D image data')
    parser.add_argument('input_seg', help='Path to the segmented image or image archive (ZIP)')
    parser.add_argument('input_gt', help='Path to the ground truth image or image archive (ZIP)')
    parser.add_argument('results', help='Path to the results file (TSV)')
    parser.add_argument('-unzip', action='store_true')
    parser.add_argument('-seg_unique', action='store_true')
    parser.add_argument('-gt_unique', action='store_true')
    parser.add_argument('measures', nargs='+', type=str, help='list of performance measures')
    args = parser.parse_args()

    if args.unzip:
        zipfile_seg = zipfile.ZipFile(args.input_seg)
        zipfile_gt = zipfile.ZipFile(args.input_gt)
        with tempfile.TemporaryDirectory() as tmpdir:
            basepath = pathlib.Path(tmpdir)
            gt_path, seg_path = basepath / 'gt', basepath / 'seg'
            zipfile_seg.extractall(str(seg_path))
            zipfile_gt.extractall(str(gt_path))
            process_batch(seg_dir=seg_path, seg_file=rf'^{seg_path}/(.+)$', gt_file=gt_path / r'\1', tsv_output_file=args.results, recursive=True, gt_unique=args.gt_unique, seg_unique=args.seg_unique, measures=args.measures)

    else:
        seg_path = pathlib.Path(args.input_seg)
        process_batch(seg_dir=seg_path.parent, seg_file=seg_path, gt_file=args.input_gt, tsv_output_file=args.results, recursive=False, gt_unique=args.gt_unique, seg_unique=args.seg_unique, measures=args.measures)
