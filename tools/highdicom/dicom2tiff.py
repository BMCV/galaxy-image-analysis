import argparse
import pathlib
import sys
from typing import (
    get_args,
    Literal,
)

import giatools
import highdicom
import numpy as np
import pydicom


FrameAxis = Literal['T', 'Z', 'Q']
frame_axes = get_args(FrameAxis)


def dicom_to_tiff(
    dcm_filepath: pathlib.Path,
    tiff_filepath: pathlib.Path,
    multiframe_axis: FrameAxis,
    normalize_label_maps: bool,
    config: dict,
):
    assert multiframe_axis in frame_axes, (
        f'Not a valid axis for DICOM frames: "{multiframe_axis}"'
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
        print(
            'DICOM dataset is an unknown multi-frame image'
            if dcm.number_of_frames > 1 else
            'DICOM dataset is a single-frame image'
        )
        arr = dcm.get_frames(**config)

        # Construct OME-TIFF axes string
        axes = 'YX'
        if dcm.number_of_frames >= 1:
            axes = f'{multiframe_axis}{axes}'
        if len(axes) < arr.ndim:
            axes += 'C'

    # Normalize singleton axes
    if axes[0] in frame_axes and arr.shape[0] == 1:
        arr = arr[0]
        axes = axes[1:]

    # Normalize label maps
    if dcm.SOPClassUID == pydicom.uid.SegmentationStorage:
        print('DICOM dataset is a label map')
        if normalize_label_maps:
            arr = normalize_label_map(arr)

    # Write TIFF file
    print('Output TIFF shape:', arr.shape)
    print('Output TIFF axes:', axes)
    giatools.Image(arr, axes).write(str(tiff_filepath))


def normalize_label_map(arr: np.ndarray) -> np.ndarray:
    labels = np.unique(arr)
    num_labels = len(labels)

    # Choose appropriate data type
    if num_labels < 2:
        print(f'Too few labels ({num_labels}), skipping normalization')
        norm_dtype = None
    elif num_labels <= 0xff:
        norm_dtype = np.uint8
        norm_max_label = 0xff
    elif num_labels <= 0xffff:
        norm_dtype = np.uint16
        norm_max_label = 0xffff
    else:
        print(f'Too many labels ({num_labels}), skipping normalization')
        norm_dtype = None

    # Create normalized label map
    if norm_dtype is not None:
        norm_arr = np.zeros(arr.shape, norm_dtype)
        for label_idx, label in enumerate(sorted(labels)):
            norm_label = (norm_max_label * label_idx) // (len(labels) - 1)
            cc = (arr == label)
            norm_arr[cc] = norm_label

        # Verify that the normalization was loss-less, and return
        # (normalization should never fail, but better check)
        if len(np.unique(norm_arr)) != num_labels:
            print('Label map normalization failed', file=sys.stderr)
            return arr  # this should never happen in practice, but better check
        else:
            return norm_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dcm', type=pathlib.Path)
    parser.add_argument('tiff', type=pathlib.Path)
    parser.add_argument('multiframe_axis', type=str, choices=frame_axes, help='OME-TIFF axis to be used for the frames of a multi-frame DICOM file')
    parser.add_argument('dtype', type=str)
    parser.add_argument('--normalize_label_maps', default=False, action='store_true')
    parser.add_argument('--apply_voi_transform', default=False, action='store_true')
    args = parser.parse_args()

    dicom_to_tiff(
        args.dcm,
        args.tiff,
        multiframe_axis=args.multiframe_axis,
        normalize_label_maps=args.normalize_label_maps,
        config=dict(
            dtype=args.dtype or None,  # use `None` instead of an empty string
            apply_voi_transform=args.apply_voi_transform,
        ),
    )
