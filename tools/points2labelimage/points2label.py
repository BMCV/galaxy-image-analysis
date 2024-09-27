import argparse
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.io
import skimage.segmentation


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


def rasterize(point_file, out_file, shape, has_header=False, swap_xy=False, bg_value=0, fg_value=None):

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

            # Rasterize circle and distribute overlapping image area
            if radius > 0:
                mask = np.ones(shape, dtype=bool)
                mask[y, x] = False
                mask = (ndi.distance_transform_edt(mask) <= radius)

                # Compute the overlap (pretend there is none if the rasterization is binary)
                if fg_value is None:
                    overlap = np.logical_and(img > 0, mask)
                else:
                    overlap = np.zeros(shape, dtype=bool)

                # Rasterize the part of the circle which is disjoint from other foreground.
                #
                # In the current implementation, the result depends on the order of the rasterized circles if somewhere
                # more than two circles overlap. This is probably negligable for most applications. To achieve results
                # that are invariant to the order, first all circles would need to be rasterized independently, and
                # then blended together. This, however, would either strongly increase the memory consumption, or
                # require a more complex implementation which exploits the sparsity of the rasterized masks.
                #
                disjoint_mask = np.logical_xor(mask, overlap)
                if disjoint_mask.any():
                    img[disjoint_mask] = label

                    # Distribute the remaining part of the circle
                    if overlap.any():
                        dist = ndi.distance_transform_edt(overlap)
                        foreground = (img > 0)
                        img[overlap] = 0
                        img = skimage.segmentation.watershed(dist, img, mask=foreground)

            # Rasterize point (there is no overlapping area to be distributed)
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

    rasterize(
        args.point_file.name,
        args.out_file,
        (args.shapey, args.shapex),
        has_header=args.has_header,
        swap_xy=args.swap_xy,
        fg_value=0xffff if args.binary else None,
    )
