"""
Copyright 2022 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import csv
import itertools
import pathlib
import tempfile
import zipfile

import numpy as np
import segmetrics as sm
import skimage.io


measures = [
    ('dice', 'Dice', sm.regional.Dice()),
    ('seg', 'SEG', sm.regional.ISBIScore()),
    ('jc', 'Jaccard coefficient', sm.regional.JaccardSimilarityIndex()),
    ('ji', 'Jaccard index', sm.regional.JaccardIndex()),
    ('ri', 'Rand index', sm.regional.RandIndex()),
    ('ari', 'Adjusted Rand index', sm.regional.AdjustedRandIndex()),
    ('hsd_sym', 'HSD (sym)', sm.boundary.Hausdorff('sym')),
    ('hsd_e2a', 'HSD (e2a)', sm.boundary.Hausdorff('e2a')),
    ('hsd_a2e', 'HSD (a2e)', sm.boundary.Hausdorff('a2e')),
    ('nsd', 'NSD', sm.boundary.NSD()),
    ('o_hsd_sym', 'Ob. HSD (sym)', sm.boundary.ObjectBasedDistance(sm.boundary.Hausdorff('sym'))),
    ('o_hsd_e2a', 'Ob. HSD (e2a)', sm.boundary.ObjectBasedDistance(sm.boundary.Hausdorff('e2a'))),
    ('o_hsd_a2e', 'Ob. HSD (a2e)', sm.boundary.ObjectBasedDistance(sm.boundary.Hausdorff('a2e'))),
    ('o_nsd', 'Ob. NSD', sm.boundary.ObjectBasedDistance(sm.boundary.NSD())),
    ('fs', 'Split', sm.detection.FalseSplit()),
    ('fm', 'Merge', sm.detection.FalseMerge()),
    ('fp', 'Spurious', sm.detection.FalsePositive()),
    ('fn', 'Missing', sm.detection.FalseNegative()),
]


def process_batch(study, gt_filelist, seg_filelist, namelist, gt_is_unique, seg_is_unique):
    for gt_filename, seg_filename, name in zip(gt_filelist, seg_filelist, namelist):
        img_ref = skimage.io.imread(gt_filename)
        img_seg = skimage.io.imread(seg_filename)
        study.set_expected(img_ref, unique=gt_is_unique)
        study.process(img_seg, unique=seg_is_unique, chunk_id=name)


def aggregate(measure, values):
    fnc = np.sum if measure.ACCUMULATIVE else np.mean
    return fnc(values)


def is_zip_filepath(filepath):
    return filepath.lower().endswith('.zip')


def is_image_filepath(filepath):
    suffixes = ['png', 'tif', 'tiff']
    return any((filepath.lower().endswith(f'.{suffix}') for suffix in suffixes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image segmentation and object detection performance measures for 2-D image data')
    parser.add_argument('input_seg', help='Path to the segmented image or image archive (ZIP)')
    parser.add_argument('input_gt', help='Path to the ground truth image or image archive (ZIP)')
    parser.add_argument('results', help='Path to the results file (CSV)')
    parser.add_argument('-unzip', action='store_true')
    parser.add_argument('-seg_unique', action='store_true')
    parser.add_argument('-gt_unique', action='store_true')
    for measure in measures:
        parser.add_argument(f'-measure-{measure[0]}', action='store_true', help=f'Include {measure[1]}')

    args = parser.parse_args()
    study = sm.study.Study()

    used_measures = []
    for measure in measures:
        if getattr(args, f'measure_{measure[0]}'):
            used_measures.append(measure)
            study.add_measure(measure[2], measure[1])

    if args.unzip:
        zipfile_seg = zipfile.ZipFile(args.input_seg)
        zipfile_gt = zipfile.ZipFile(args.input_gt)
        namelist = [filepath for filepath in zipfile_seg.namelist() if is_image_filepath(filepath) and filepath in zipfile_gt.namelist()]
        print('namelist:', namelist)
        with tempfile.TemporaryDirectory() as tmpdir:
            basepath = pathlib.Path(tmpdir)
            gt_path, seg_path = basepath / 'gt', basepath / 'seg'
            zipfile_seg.extractall(str(seg_path))
            zipfile_gt.extractall(str(gt_path))
            gt_filelist, seg_filelist = list(), list()
            for filepath in namelist:
                seg_filelist.append(str(seg_path / filepath))
                gt_filelist.append(str(gt_path / filepath))
            process_batch(study, gt_filelist, seg_filelist, namelist, args.gt_unique, args.seg_unique)

    else:
        namelist = ['']
        process_batch(study, [args.input_gt], [args.input_seg], namelist, args.gt_unique, args.seg_unique)

    # define header
    rows = [[''] + [measure[1] for measure in used_measures]]

    # define rows
    if len(namelist) > 1:
        for chunk_id in namelist:
            row = [chunk_id]
            for measure in used_measures:
                measure_name = measure[1]
                measure = study.measures[measure_name]
                chunks = study.results[measure_name]
                row += [aggregate(measure, chunks[chunk_id])]
            rows.append(row)

    # define footer
    rows.append([''])
    for measure in used_measures:
        measure_name = measure[1]
        measure = study.measures[measure_name]
        chunks = study.results[measure_name]
        values = list(itertools.chain(*[chunks[chunk_id] for chunk_id in chunks]))
        val = aggregate(measure, values)
        rows[-1].append(val)

    # write results
    with open(args.results, 'w', newline='') as fout:
        csv_writer = csv.writer(fout, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            csv_writer.writerow(row)
