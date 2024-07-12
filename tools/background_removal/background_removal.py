import argparse
import warnings


import skimage.io
from skimage.io import imread
from skimage.filters import difference_of_gaussians
from skimage.morphology import white_tophat, disk
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
        output_image = skimage.util.img_as_uint(output_image)
        skimage.io.imsave(args.output, output_image, plugin="tifffile")


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