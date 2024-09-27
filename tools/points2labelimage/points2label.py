import argparse
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.io


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Returns the column name present in `df` and the list of `candidates`.

    Raises:
        KeyError: If there is no candidate column name present in `df`, or more than one.
    """
    intersection = frozenset(df.columns) & frozenset(candidates)
    if len(intersection) == 0:
        raise KeyError(f'No such column: {", ".join(candidates)}')
    elif len(intersection) > 1:
        raise KeyError(f'The column names {", ".join(intersection)} are ambiguous')
    else:
        return next(iter(intersection))


def points2binaryimage(point_file, out_file, shape, has_header=False, swap_xy=False, bg_value=0, fg_value=None):

    img = np.full(shape, dtype=np.uint16, fill_value=bg_value)
    if os.path.exists(point_file) and os.path.getsize(point_file) > 0:

        # Read the tabular file with information from the header
        if has_header:
            df = pd.read_csv(point_file, delimiter='\t')
            pos_x_column = find_column(df, ['pos_x', 'POS_X'])
            pos_y_column = find_column(df, ['pos_y', 'POS_Y'])
            pos_x_list = df[pos_x_column].round().astype(int)
            pos_y_list = df[pos_y_column].round().astype(int)
            assert len(pos_x_list) == len(pos_y_list)
            try:
                radius_column = find_column(df, ['radius', 'RADIUS'])
                radius_list = df[radius_column]
            except KeyError:
                radius_list = [0] * len(pos_x_list)

        # Read the tabular file without header
        else:
            df = pd.read_csv(point_file, header=None, delimiter='\t')
            pos_x_list = df[0].round().astype(int)
            pos_y_list = df[1].round().astype(int)
            assert len(pos_x_list) == len(pos_y_list)
            radius_list = [0] * len(pos_x_list)

        # Optionally swap the coordinates
        if swap_xy:
            pos_x_list, pos_y_list = pos_y_list, pos_x_list

        # Perform the rasterization
        for pidx, (y, x, radius) in enumerate(zip(pos_y_list, pos_x_list, radius_list)):
            label = pidx + 1 if fg_value is None else fg_value

            if y < 0 or x < 0 or y >= shape[0] or x >= shape[1]:
                raise IndexError(f'The point x={x}, y={y} exceeds the bounds of the image (width: {shape[1]}, height: {shape[0]})')

            if radius > 0:
                mask = np.ones(shape, dtype=bool)
                mask[y, x] = False
                mask = (ndi.distance_transform_edt(mask) <= radius)
                img[mask] = label
            else:
                img[y, x] = label

    else:
        raise Exception("{} is empty or does not exist.".format(point_file))  # appropriate built-in error?

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(out_file, img, plugin='tifffile')  # otherwise we get problems with the .dat extension


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('point_file', type=argparse.FileType('r'), help='point file')
    parser.add_argument('out_file', type=str, help='out file (TIFF)')
    parser.add_argument('shapex', type=int, help='shapex')
    parser.add_argument('shapey', type=int, help='shapey')
    parser.add_argument('--has_header', dest='has_header', default=False, help='set True if point file has header')
    parser.add_argument('--swap_xy', dest='swap_xy', default=False, help='Swap X and Y coordinates')
    parser.add_argument('--binary', dest='binary', default=False, help='Produce binary image')

    args = parser.parse_args()

    # TOOL
    points2binaryimage(
        args.point_file.name,
        args.out_file,
        (args.shapey, args.shapex),
        has_header=args.has_header,
        swap_xy=args.swap_xy,
        fg_value=0xffff if args.binary else None,
    )
