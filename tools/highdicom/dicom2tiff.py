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
):
    assert multiframe_type in frame_axes, (
        f'Not a valid axis for DICOM frames: "{multiframe_type}"'
    )
    dcm = highdicom.imread(dcm_filepath)

    kwargs = dict()
    kwargs['dtype'] = np.float32  # highdicom default is float64
    kwargs['apply_voi_transform'] = False

    # Load frames (DICOM uses 1-based indexing)
    arr = dcm.get_frames(**kwargs)

    # Construct OME-TIFF axes string
    axes = 'YX'
    if dcm.number_of_frames >= 1:
        axes = f'{multiframe_type}{axes}'
    if len(axes) < arr.ndim:
        axes += 'C'
    if axes[0] in frame_axes and arr.shape[0] == 1:
        arr = arr[0]
        axes = axes[1:]
    print('axes:', axes)
    print('shape:', arr.shape)

    # Write TIFF file
    giatools.Image(arr, axes).write(tiff_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dcm', type=pathlib.Path)
    parser.add_argument('tiff', type=pathlib.Path)
    parser.add_argument('multiframe_axis', type=str, help='OME-TIFF axis to be used for the frames of a multi-frame DICOM file')
    args = parser.parse_args()

    dicom_to_tiff(
        args.dcm,
        args.tiff,
        args.multiframe_axis,
    )
