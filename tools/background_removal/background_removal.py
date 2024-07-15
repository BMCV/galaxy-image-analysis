import argparse
import warnings

import numpy as np
import skimage.io
from skimage.filters import difference_of_gaussians
from skimage.io import imread
from skimage.morphology import disk, white_tophat
from skimage.restoration import rolling_ball


def process_image(args):
    image = imread(args.input_image)

    if args.filter == "rolling_ball":
        background_rolling = rolling_ball(image, radius=args.radius)
        output_image = image - background_rolling

    elif args.filter == "dog":
        output_image = difference_of_gaussians(image, low_sigma=0, high_sigma=args.radius)

    elif args.filter == "top_hat":
        output_image = white_tophat(image, disk(args.radius))

    with warnings.catch_warnings():
        output_image = convert_image_to_format_of(output_image, image)
        skimage.io.imsave(args.output, output_image, plugin="tifffile")


def convert_image_to_format_of(image, format_image):
    """
    Convert the first image to the format of the second image.
    """
    if format_image.dtype == image.dtype:
        return image
    elif format_image.dtype == np.uint8:
        return skimage.util.img_as_ubyte(image)
    elif format_image.dtype == np.uint16:
        return skimage.util.img_as_uint(image)
    elif format_image.dtype == np.int16:
        return skimage.util.img_as_int(image)
    else:
        raise ValueError(f'Unsupported image data type: {format_image.dtype}')


def main():
    parser = argparse.ArgumentParser(description="Background removal script using skiimage")
    parser.add_argument('input_image', help="Input image path")
    parser.add_argument('filter', choices=['rolling_ball', 'dog', 'top_hat'],
                        help="Background removal algorithm")
    parser.add_argument('radius', type=float, help="Radius")
    parser.add_argument('output', help="Output image path")

    args = parser.parse_args()
    process_image(args)


if __name__ == '__main__':
    main()
