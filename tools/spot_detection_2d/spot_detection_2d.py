'''
Copyright 2021-2026 Biomedical Computer Vision Group, Heidelberg University.
Authors:
- Qi Gao (qi.gao@bioquant.uni-heidelberg.de)
- Leonid Kostrykin (leonid.kostrykin@bioquant.uni-heidelberg.de)

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
'''

from typing import Iterator

import giatools
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from numpy.typing import NDArray
from skimage.feature import blob_dog, blob_doh, blob_log, peak_local_max


def local_max_detector(
    img: NDArray,
    sigma: float,
    threshold: float,
    threshold_rel: float,
    intensity_offset: tuple[int, int],
) -> Iterator[tuple[int, int, dict]]:

    if sigma > 0:
        img = ndi.gaussian_filter(img.astype(np.float64), sigma)
    img_smooth = img.copy()

    # Handle images with negative intensities
    if img.min() < 0:
        img_min = img.min()
        img -= img_min
        threshold -= img_min

    # Perform thresholding
    img_max = img.max()
    img[img < threshold] = 0
    img[img < threshold_rel * img_max] = 0

    # Find local maxima
    yx_list = peak_local_max(img, min_distance=1).round().astype(int)
    for y, x in yx_list:
        intensity_yx = tuple(np.add((y, x), intensity_offset))
        if (
            0 <= intensity_yx[0] < img.shape[0] and
            0 <= intensity_yx[1] < img.shape[1]
        ):
            intensity = img_smooth[intensity_yx]
            yield y, x, dict(intensity=intensity)


def create_multiscale_blob_detector(func):
    def impl(img: NDArray, **kwargs):
        for y, x, scale in func(img, **kwargs):
            y, x = round(y), round(x)
            radius = scale * np.sqrt(2) * 2
            intensity = mean_intensity(img, y, x, round(radius))
            yield y, x, dict(scale=scale, radius=radius, intensity=intensity)
    return impl


methods = {
    'dog': create_multiscale_blob_detector(blob_dog),
    'doh': create_multiscale_blob_detector(blob_doh),
    'log': create_multiscale_blob_detector(blob_log),
    'local_max': local_max_detector,
}


def mean_intensity(img: NDArray, y: int, x: int, radius: int) -> float:
    assert img.ndim == 2
    assert radius >= 0
    if radius == 0:
        return float(img[y, x])
    else:
        mask = np.ones(img.shape, bool)
        mask[y, x] = False
        mask = (ndi.distance_transform_edt(mask) <= radius)
        return img[mask].mean()


def normalize_frame_number(n_frames: int, frame: int) -> int:
    """
    Translate negative frame numbers into positives by counting from the end.

    Raises:
        ValueError: The given frame is beyond the end of the sequence.

    Returns:
        Integer number between 0 and num_frames - 1.
    """
    if frame >= n_frames:
        raise ValueError(
            f'Frame {frame} is beyond the end of the sequence ({n_frames} frames).',
        )
    return frame % n_frames


def spot_detection(
    image: giatools.Image,
    output: str,
    frame_1st: int,
    frame_end: int | None,
    method: str,
    method_kwargs: dict,
    abs_threshold: float,
    rel_threshold: float,
    boundary: int,
) -> None:
    stack = image.normalize_axes_like('TYX').data

    # Slice the stack
    frame_1st = normalize_frame_number(stack.shape[0], frame_1st)
    if frame_end is not None:
        frame_end = normalize_frame_number(stack.shape[0], frame_end)
        if frame_1st >= frame_end:
            raise ValueError(
                f'Frist frame of the sequence (frame {frame_1st}) '
                f'is beyond the end of the sequence (frame {frame_end}).',
            )
        stack = stack[frame_1st:frame_end]
    else:
        stack = stack[frame_1st:]

    # Select the detection method
    detector = methods[method.lower()]

    # Perform detection on each image of the stack
    detections = list()
    for img_idx, img in enumerate(stack):
        spots = detector(img, threshold=abs_threshold, threshold_rel=rel_threshold, **method_kwargs)
        for y, x, spot_info in spots:

            # Skip the detection if it is too close to the boundary of the image
            if y < boundary or x < boundary or y >= img.shape[0] - boundary or x >= img.shape[1] - boundary:
                continue

            # Add the detection to the list of detections
            detections.append(
                {
                    'frame': img_idx,
                    'pos_x': x,
                    'pos_y': y,
                }
                | spot_info
            )

    # Build and save dataframe
    df = pd.DataFrame.from_dict(detections)
    df.to_csv(output, index=False, float_format='%.2f', sep='\t')


if __name__ == '__main__':
    tool = giatools.ToolBaseplate()
    tool.add_input_image('intensities')
    tool.parser.add_argument('--output', type=str, required=True)
    tool.parse_args()

    try:
        image = tool.args.input_images['intensities']

        # Validate the input image(s)
        if any(image.shape[image.axes.index(axis)] > 1 for axis in image.axes if axis not in 'TYX'):
            raise ValueError(f'This tool is not applicable to images with {image.original_axes} axes.')

        spot_detection(
            image=image,
            output=tool.args.raw_args.output,
            **tool.args.params,
        )

    except ValueError as err:
        exit(err.args[0])
