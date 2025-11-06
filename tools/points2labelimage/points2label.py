import argparse
import json
import warnings
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import giatools.pandas
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage as ndi
import skimage.draw
import skimage.io
import skimage.segmentation


def get_list_depth(nested_list: Any) -> int:
    if isinstance(nested_list, list):
        if len(nested_list) > 0:
            return 1 + max(map(get_list_depth, nested_list))
        else:
            return 1
    else:
        return 0


class AutoLabel:
    """
    Creates a sequence of unique labels (non-negative values).
    """

    def __init__(self, reserved_labels):
        self.reserved_labels = reserved_labels
        self.next_autolabel = 0

    def next(self):
        """
        Retrieve the next auto-label (post-increment).
        """
        # Fast-forward `next_autolabel` to the first free label
        while self.next_autolabel in self.reserved_labels:
            self.next_autolabel += 1

        # Return the free label, then advance `next_autolabel`
        try:
            return self.next_autolabel
        finally:
            self.next_autolabel += 1


def get_feature_label(feature: Dict) -> Optional[int]:
    """
    Get the label of a GeoJSON feature, or `None` if there is no proper label.
    """
    label = feature.get('properties', {}).get('name', None)
    if label is None:
        return None

    # If the `label` is given as a string, try to parse as integer
    if isinstance(label, str):
        try:
            label = int(label)
        except ValueError:
            pass

    # Finally, if `label` is an integer, only use it if it is non-negative
    if isinstance(label, int) and label >= 0:
        return label
    else:
        return None


def rasterize(
    geojson: Dict,
    shape: Tuple[int, int],
    bg_value: int = 0,
    fg_value: Optional[int] = None,
) -> npt.NDArray:
    """
    Rasterize GeoJSON into a pixel image, that is returned as a NumPy array.
    """

    # Determine which labels are reserved (not used by auto-label)
    reserved_labels = [bg_value]
    if fg_value is None:
        for feature in geojson['features']:
            label = get_feature_label(feature)
            if label is not None:
                reserved_labels.append(label)

    # Convert `reserved_labels` into a `set` for faster look-ups
    reserved_labels = frozenset(reserved_labels)

    # Define routine to retrieve the next auto-label
    autolabel = AutoLabel(reserved_labels)

    # Rasterize the image
    img = np.full(shape, dtype=np.uint16, fill_value=bg_value)
    for feature in geojson['features']:
        geom_type = feature['geometry']['type'].lower()
        coords = feature['geometry']['coordinates']

        # Rasterize a `mask` separately for each feature
        if geom_type == 'polygon':

            # Normalization: Let there always be a list of polygons
            if get_list_depth(coords) == 2:
                coords = [coords]

            # Rasterize each polygon separately, then join via XOR
            mask = np.zeros(shape, dtype=bool)
            for polygon_coords in coords:
                polygon_mask = skimage.draw.polygon2mask(
                    shape,
                    [point[::-1] for point in polygon_coords],
                )
                mask = np.logical_xor(mask, polygon_mask)

        elif geom_type == 'point':
            mask = np.zeros(shape, dtype=bool)
            mask[coords[1], coords[0]] = True
            radius = feature.get('properties', {}).get('radius', 0)
            if radius > 0:
                mask = (ndi.distance_transform_edt(~mask) <= radius)

        else:
            raise ValueError(
                f'Unsupported geometry type: "{feature["geometry"]["type"]}"',
            )

        # Determine the `label` for the current `mask`
        if fg_value is None:
            label = get_feature_label(feature)
            if label is None:
                label = autolabel.next()
        else:
            label = fg_value

        # Blend the current `mask` with the rasterized image
        img[mask] = label

    # Return the rasterized image
    return img


def convert_tabular_to_geojson(
    tabular_file: str,
    has_header: bool,
) -> dict:
    """
    Read a tabular file and convert it to GeoJSON.

    The GeoJSON data is returned as a dictionary.
    """

    # Read the tabular file with information from the header
    if has_header:
        df = pd.read_csv(tabular_file, delimiter='\t')

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
        df = pd.read_csv(tabular_file, header=None, delimiter='\t')
        pos_x_list = df[0].round().astype(int)
        pos_y_list = df[1].round().astype(int)
        assert len(pos_x_list) == len(pos_y_list)
        radius_list, width_list, height_list = [[0] * len(pos_x_list)] * 3
        label_list = list(range(1, len(pos_x_list) + 1))

    # Convert to GeoJSON
    features = []
    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }
    for y, x, radius, width, height, label in zip(
        pos_y_list, pos_x_list, radius_list, width_list, height_list, label_list,
    ):
        if radius > 0 and width > 0 and height > 0:
            raise ValueError('Ambiguous shape type (circle or rectangle)')

        # Create a rectangle
        if width > 0 and height > 0:
            geom_type = 'Polygon'
            coords = [
                [x, y],
                [x + width - 1, y],
                [x + width - 1, y + height - 1],
                [x, y + height - 1],
            ]

        # Create a point or circle
        else:
            geom_type = 'Point'
            coords = [x, y]

        # Create a GeoJSON feature
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': geom_type,
                'coordinates': coords,
            },
            'properties': {
                'name': label,
            },
        }
        if radius > 0:
            feature['properties']['radius'] = radius
            feature['properties']['subType'] = 'Circle'
        features.append(feature)

    # Return the GeoJSON object
    return geojson


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_ext', type=str, help='Input file format')
    parser.add_argument('in_file', type=str, help='Input file path (tabular or GeoJSON)')
    parser.add_argument('out_file', type=str, help='Output file path (TIFF)')
    parser.add_argument('shapex', type=int, help='Output image width')
    parser.add_argument('shapey', type=int, help='Output image height')
    parser.add_argument('--has_header', dest='has_header', default=False, help='Set True if tabular file has a header')
    parser.add_argument('--swap_xy', dest='swap_xy', default=False, help='Swap X and Y coordinates')
    parser.add_argument('--binary', dest='binary', default=False, help='Produce binary image')
    args = parser.parse_args()

    # Validate command-line arguments
    assert args.in_ext in ('tabular', 'geojson'), (
        f'Unexpected input file format: {args.in_ext}'
    )

    # Load the GeoJSON data (if the input file is tabular, convert to GeoJSON)
    if args.in_ext == 'tabular':
        geojson = convert_tabular_to_geojson(args.in_file, args.has_header)
    else:
        with open(args.in_file) as f:
            geojson = json.load(f)

    # Rasterize the image from GeoJSON
    shape = (args.shapey, args.shapex)
    img = rasterize(
        geojson,
        shape if not args.swap_xy else shape[::-1],
        fg_value=0xffff if args.binary else None,
    )
    if args.swap_xy:
        img = img.T

    # Write the rasterized image as TIFF
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(args.out_file, img, plugin='tifffile')  # otherwise we get problems with the .dat extension
