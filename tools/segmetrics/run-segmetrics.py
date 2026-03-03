"""
Copyright 2022-2023 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import json
import os
import subprocess
import tempfile

import pandas as pd


def process_batch(seg_dir, seg_file, gt_file, tsv_output_file, recursive, gt_unique, seg_unique, measures, sample_ids):
    with tempfile.NamedTemporaryFile() as csv_output_file:
        cmd = ['python', '-m', 'segmetrics.cli', str(seg_dir), str(seg_file), str(gt_file), str(csv_output_file.name), '--semicolon']
        if recursive:
            cmd.append('--recursive')
        if gt_unique:
            cmd.append('--gt-unique')
        if seg_unique:
            cmd.append('--seg-unique')
        cmd += measures
        subprocess.run(cmd, check=True)
        df = pd.read_csv(csv_output_file.name, sep=';')
        df['Sample'] = df['Sample'].map(sample_ids)
        df.to_csv(str(tsv_output_file), sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image segmentation and object detection performance measures for 2-D image data')
    parser.add_argument('input_seg', type=str, help='Path to a directroy with segmented images')
    parser.add_argument('input_gt', type=str, help='Path to a directory with ground truth images')
    parser.add_argument('sample_ids', type=str, help='Path to JSON file with one sample ID per image')
    parser.add_argument('results', help='Path to the results file (TSV)')
    parser.add_argument('--seg_unique', action='store_true', default=False)
    parser.add_argument('--gt_unique', action='store_true', default=False)
    parser.add_argument('measures', nargs='+', type=str, help='list of performance measures')
    args = parser.parse_args()

    with open(args.sample_ids) as fp:
        sample_ids = json.load(fp)

    process_batch(
        seg_dir=args.input_seg,
        seg_file=rf'^{args.input_seg}/(p.+)$',
        gt_file=os.path.join(args.input_gt, r'\1'),
        tsv_output_file=args.results,
        recursive=True,
        gt_unique=args.gt_unique,
        seg_unique=args.seg_unique,
        measures=args.measures,
        sample_ids=sample_ids,
    )
