import argparse
import pathlib
from typing import Literal, get_args

import giatools
import highdicom
import numpy as np


FrameAxis = Literal['T', 'Z', 'Q']
frame_axes = get_args(FrameAxis)


def dicom_to_tiff(
    dcm_filepath: pathlib.Path,
    tiff_filepath: pathlib.Path,
    multiframe_type: FrameAxis,
    config: dict,
):
    assert multiframe_type in frame_axes, (
        f'Not a valid axis for DICOM frames: "{multiframe_type}"'
    )
    dcm = highdicom.imread(dcm_filepath)

    # If the image is a tiled image, ...
    if dcm.is_tiled:
        print('DICOM dataset is a tiled multi-frame image')

        # ...extract the top-level WSI by combining the frames (tiles) into a mosaic
        arr = dcm.get_total_pixel_matrix(**config)
        axes = 'YX' if arr.ndim == 2 else 'YXC'

    # Check if the image is a 3-D volume. According to the docs [1], `highdicom.Image.get_volume_geometry`
    # will succeed (i.e. return a non-None) if the data *is* a 3-D volume, or if it is a tiled mosaic.
    # Hence, to distinguish tiled images from 3-D volumes, we use the `elif` branching semantic.
    #
    # [1] https://highdicom.readthedocs.io/en/latest/package.html#highdicom.Image.get_volume_geometry
    #
    # So, if the image is a 3-D volume, ...
    elif dcm.get_volume_geometry() is not None:
        print('DICOM dataset is a 3-D volume')

        # ...extract the 3-D volume by joining the frames (z-slices)
        arr = dcm.get_volume(**config).array
        axes = 'ZYX' if arr.ndim == 3 else 'ZYXC'

    # Otherwise, extract a raw stack of frames
    else:
        print('DICOM dataset is an unknown multi-frame image')
        arr = dcm.get_frames(**kwargs)

        # Construct OME-TIFF axes string
        axes = 'YX'
        if dcm.number_of_frames >= 1:
            axes = f'{multiframe_type}{axes}'
        if len(axes) < arr.ndim:
            axes += 'C'

    # Normalize singleton axes
    if axes[0] in frame_axes and arr.shape[0] == 1:
        arr = arr[0]
        axes = axes[1:]

    # Write TIFF file
    print('Output TIFF shape:', arr.shape)
    print('Output TIFF axes:', axes)
    giatools.Image(arr, axes).write(tiff_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dcm', type=pathlib.Path)
    parser.add_argument('tiff', type=pathlib.Path)
    parser.add_argument('multiframe_axis', type=str, help='OME-TIFF axis to be used for the frames of a multi-frame DICOM file')
    parser.add_argument('dtype', type=str)
    parser.add_argument('--apply_voi_transform', type=bool, default=False, action='store_true')
    args = parser.parse_args()

    dicom_to_tiff(
        args.dcm,
        args.tiff,
        args.multiframe_axis,
        dict(
            dtype=args.dtype or None,  # use `None` instead of an empty string 
            apply_voi_transform=args.apply_voi_transform,
        ),
    )
