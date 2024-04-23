import argparse

import giatools.io
import numpy as np
import pandas as pd
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.segmentation


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract image features')

    # TODO create factory for boilerplate code
    features = parser.add_argument_group('compute features')
    features.add_argument('--all', dest='all_features', action='store_true')
    features.add_argument('--label', dest='add_label', action='store_true')
    features.add_argument('--patches', dest='add_roi_patches', action='store_true')
    features.add_argument('--max_intensity', dest='max_intensity', action='store_true')
    features.add_argument('--mean_intensity', dest='mean_intensity', action='store_true')
    features.add_argument('--min_intensity', dest='min_intensity', action='store_true')
    features.add_argument('--moments_hu', dest='moments_hu', action='store_true')
    features.add_argument('--centroid', dest='centroid', action='store_true')
    features.add_argument('--bbox', dest='bbox', action='store_true')
    features.add_argument('--area', dest='area', action='store_true')
    features.add_argument('--filled_area', dest='filled_area', action='store_true')
    features.add_argument('--convex_area', dest='convex_area', action='store_true')
    features.add_argument('--perimeter', dest='perimeter', action='store_true')
    features.add_argument('--extent', dest='extent', action='store_true')
    features.add_argument('--eccentricity', dest='eccentricity', action='store_true')
    features.add_argument('--equivalent_diameter', dest='equivalent_diameter', action='store_true')
    features.add_argument('--euler_number', dest='euler_number', action='store_true')
    features.add_argument('--inertia_tensor_eigvals', dest='inertia_tensor_eigvals', action='store_true')
    features.add_argument('--major_axis_length', dest='major_axis_length', action='store_true')
    features.add_argument('--minor_axis_length', dest='minor_axis_length', action='store_true')
    features.add_argument('--orientation', dest='orientation', action='store_true')
    features.add_argument('--solidity', dest='solidity', action='store_true')
    features.add_argument('--moments', dest='moments', action='store_true')
    features.add_argument('--convexity', dest='convexity', action='store_true')

    parser.add_argument('--label_file_binary', dest='label_file_binary', action='store_true')

    parser.add_argument('--raw', dest='raw_file', type=argparse.FileType('r'),
                        help='Original input file', required=False)
    parser.add_argument('label_file', type=argparse.FileType('r'),
                        help='Label input file')
    parser.add_argument('output_file', type=argparse.FileType('w'),
                        help='Tabular output file')
    args = parser.parse_args()

    label_file_binary = args.label_file_binary
    label_file = args.label_file.name
    out_file = args.output_file.name
    add_patch = args.add_roi_patches

    raw_image = None
    if args.raw_file is not None:
        raw_image = giatools.io.imread(args.raw_file.name)

    raw_label_image = giatools.io.imread(label_file)

    df = pd.DataFrame()
    if label_file_binary:
        raw_label_image = skimage.measure.label(raw_label_image)
    regions = skimage.measure.regionprops(raw_label_image, intensity_image=raw_image)

    df['it'] = np.arange(len(regions))

    if add_patch:
        df['image'] = df['it'].map(lambda ait: regions[ait].image.astype(np.float).tolist())
        df['intensity_image'] = df['it'].map(lambda ait: regions[ait].intensity_image.astype(np.float).tolist())

    # TODO no matrix features, but split in own rows?
    if args.add_label or args.all_features:
        df['label'] = df['it'].map(lambda ait: regions[ait].label)

    if raw_image is not None:
        if args.max_intensity or args.all_features:
            df['max_intensity'] = df['it'].map(lambda ait: regions[ait].max_intensity)
        if args.mean_intensity or args.all_features:
            df['mean_intensity'] = df['it'].map(lambda ait: regions[ait].mean_intensity)
        if args.min_intensity or args.all_features:
            df['min_intensity'] = df['it'].map(lambda ait: regions[ait].min_intensity)
        if args.moments_hu or args.all_features:
            df['moments_hu'] = df['it'].map(lambda ait: regions[ait].moments_hu)

    if args.centroid or args.all_features:
        df['centroid'] = df['it'].map(lambda ait: regions[ait].centroid)
    if args.bbox or args.all_features:
        df['bbox'] = df['it'].map(lambda ait: regions[ait].bbox)
    if args.area or args.all_features:
        df['area'] = df['it'].map(lambda ait: regions[ait].area)
    if args.filled_area or args.all_features:
        df['filled_area'] = df['it'].map(lambda ait: regions[ait].filled_area)
    if args.convex_area or args.all_features:
        df['convex_area'] = df['it'].map(lambda ait: regions[ait].convex_area)
    if args.perimeter or args.all_features:
        df['perimeter'] = df['it'].map(lambda ait: regions[ait].perimeter)
    if args.extent or args.all_features:
        df['extent'] = df['it'].map(lambda ait: regions[ait].extent)
    if args.eccentricity or args.all_features:
        df['eccentricity'] = df['it'].map(lambda ait: regions[ait].eccentricity)
    if args.equivalent_diameter or args.all_features:
        df['equivalent_diameter'] = df['it'].map(lambda ait: regions[ait].equivalent_diameter)
    if args.euler_number or args.all_features:
        df['euler_number'] = df['it'].map(lambda ait: regions[ait].euler_number)
    if args.inertia_tensor_eigvals or args.all_features:
        df['inertia_tensor_eigvals'] = df['it'].map(lambda ait: regions[ait].inertia_tensor_eigvals)
    if args.major_axis_length or args.all_features:
        df['major_axis_length'] = df['it'].map(lambda ait: regions[ait].major_axis_length)
    if args.minor_axis_length or args.all_features:
        df['minor_axis_length'] = df['it'].map(lambda ait: regions[ait].minor_axis_length)
    if args.orientation or args.all_features:
        df['orientation'] = df['it'].map(lambda ait: regions[ait].orientation)
    if args.solidity or args.all_features:
        df['solidity'] = df['it'].map(lambda ait: regions[ait].solidity)
    if args.moments or args.all_features:
        df['moments'] = df['it'].map(lambda ait: regions[ait].moments)
    if args.convexity or args.all_features:
        perimeter = df['it'].map(lambda ait: regions[ait].perimeter)
        area = df['it'].map(lambda ait: regions[ait].area)
        df['convexity'] = area / (perimeter * perimeter)

    del df['it']
    df.to_csv(out_file, sep='\t', lineterminator='\n', index=False)
