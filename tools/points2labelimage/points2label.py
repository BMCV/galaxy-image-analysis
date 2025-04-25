import argparse
import os
import warnings
import json

import giatools.pandas
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.io
import skimage.segmentation


def geojson_to_tabular(geojson):
    rows = []
    for feature in geojson["features"]:
        name = feature["properties"].get("name")
        coords = feature["geometry"]["coordinates"][0]

        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]

        x = min(xs)
        y = min(ys)
        width = max(xs) - x
        height = max(ys) - y

        rows.append({
            "pos_x": x,
            "pos_y": y,
            "width": width,
            "height": height,
            "label": name
        })
    df = pd.DataFrame(rows)
    point_file = "./point_file.tabular"
    df.to_csv(point_file, sep="\t", index=False)
    return point_file


def rasterize(point_file, out_file, shape, has_header=False, swap_xy=False, bg_value=0, fg_value=None):

    img = np.full(shape, dtype=np.uint16, fill_value=bg_value)
    if os.path.exists(point_file) and os.path.getsize(point_file) > 0:

        # Read the tabular file with information from the header
        if has_header:
            df = pd.read_csv(point_file, delimiter='\t')

            pos_x_column = giatools.pandas.find_column(df, ['pos_x', 'POS_X'])
            pos_y_column = giatools.pandas.find_column(df, ['pos_y', 'POS_Y'])
            pos_x_list = df[pos_x_column].round().astype(int)
            pos_y_list = df[pos_y_column].round().astype(int)
            assert len(pos_x_list) == len(pos_y_list)

            try:
                radius_column = giatools.pandas.find_column(df, ['radius', 'RADIUS'])
                radius_list = df[radius_column]
                assert len(pos_x_list) == len(radius_list)
            except KeyError:
                radius_list = [0] * len(pos_x_list)

            try:
                width_column = giatools.pandas.find_column(df, ['width', 'WIDTH'])
                height_column = giatools.pandas.find_column(df, ['height', 'HEIGHT'])
                width_list = df[width_column]
                height_list = df[height_column]
                assert len(pos_x_list) == len(width_list)
                assert len(pos_x_list) == len(height_list)
            except KeyError:
                width_list = [0] * len(pos_x_list)
                height_list = [0] * len(pos_x_list)

            try:
                label_column = giatools.pandas.find_column(df, ['label', 'LABEL'])
                label_list = df[label_column]
                assert len(pos_x_list) == len(label_list)
            except KeyError:
                label_list = list(range(1, len(pos_x_list) + 1))

        # Read the tabular file without header
        else:
            df = pd.read_csv(point_file, header=None, delimiter='\t')
            pos_x_list = df[0].round().astype(int)
            pos_y_list = df[1].round().astype(int)
            assert len(pos_x_list) == len(pos_y_list)
            radius_list, width_list, height_list = [[0] * len(pos_x_list)] * 3
            label_list = list(range(1, len(pos_x_list) + 1))

        # Optionally swap the coordinates
        if swap_xy:
            pos_x_list, pos_y_list = pos_y_list, pos_x_list

        # Perform the rasterization
        for y, x, radius, width, height, label in zip(
            pos_y_list, pos_x_list, radius_list, width_list, height_list, label_list,
        ):
            if fg_value is not None:
                label = fg_value

            if y < 0 or x < 0 or y >= shape[0] or x >= shape[1]:
                raise IndexError(f'The point x={x}, y={y} exceeds the bounds of the image (width: {shape[1]}, height: {shape[0]})')

            # Rasterize circle and distribute overlapping image area
            # Rasterize primitive geometry
            if radius > 0 or (width > 0 and height > 0):

                # Rasterize circle
                if radius > 0:
                    mask = np.ones(shape, dtype=bool)
                    mask[y, x] = False
                    mask = (ndi.distance_transform_edt(mask) <= radius)
                else:
                    mask = np.zeros(shape, dtype=bool)

                # Rasterize rectangle
                if width > 0 and height > 0:
                    mask[
                        y:min(shape[0], y + width),
                        x:min(shape[1], x + height)
                    ] = True

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
    parser.add_argument('in_file', type=argparse.FileType('r'), help='Input point file or GeoJSON file')
    parser.add_argument('out_file', type=str, help='out file (TIFF)')
    parser.add_argument('shapex', type=int, help='shapex')
    parser.add_argument('shapey', type=int, help='shapey')
    parser.add_argument('--has_header', dest='has_header', default=False, help='set True if point file has header')
    parser.add_argument('--swap_xy', dest='swap_xy', default=False, help='Swap X and Y coordinates')
    parser.add_argument('--binary', dest='binary', default=False, help='Produce binary image')

    args = parser.parse_args()

    point_file = args.in_file.name
    has_header = args.has_header

    try:
        with open(args.in_file.name, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict) and content.get("type") == "FeatureCollection" and isinstance(content.get("features"), list):
                point_file = geojson_to_tabular(content)
                has_header = True  # header included in the converted file
            else:
                raise ValueError("Input is a JSON file but not a valid GeoJSON file")
    except json.JSONDecodeError:
        print("Input is not a valid JSON file. Assuming it a tabular file")

    rasterize(
        point_file,
        args.out_file,
        (args.shapey, args.shapex),
        has_header=has_header,
        swap_xy=args.swap_xy,
        fg_value=0xffff if args.binary else None,
    )
