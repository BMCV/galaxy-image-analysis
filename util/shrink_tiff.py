import argparse
import itertools
import math
import os
import tempfile
import shutil

import humanize
import skimage.transform
import tifffile


DEFAULT_MAX_SIZE = 1024 ** 2


def shrink_tiff(filepath, anti_aliasing, order, channel_axis, max_size=DEFAULT_MAX_SIZE, dry=True):
    im_full = tifffile.imread(filepath)

    for step in itertools.count(1):
        scale = math.sqrt(1 / step)
        im = skimage.transform.rescale(
            im_full,
            scale=scale,
            anti_aliasing=anti_aliasing,
            order=order,
            preserve_range=True,
            channel_axis=channel_axis,
        )

        with tempfile.NamedTemporaryFile(suffix='.tiff', mode='wb') as fp:
            tifffile.imwrite(fp.name, im)
            byte_size = os.path.getsize(fp.name)

            if byte_size <= max_size:
                if not dry:
                    shutil.copy(fp.name, filepath)
                return im, scale, byte_size


if __name__ == '__main__':

    exec_name = 'shrink_tiff.sh'
    long_help = f"""

    Example for single-channel binary images and label maps:

        {exec_name} filepath.tiff

    Example for multi-channel binary images:

        {exec_name} filepath.tiff --channel_axis 2

    Example for single-channel intensity images:

        {exec_name} filepath.tiff --anti_aliasing --order 1

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='TIFF filepath')
    parser.add_argument('--max_size', type=int, default=DEFAULT_MAX_SIZE, help='Maximum size in bytes')
    parser.add_argument('--anti_aliasing', default=False, action='store_true', help='Use anti-aliasing')
    parser.add_argument('--order', type=int, default=0, help='Order of interpolation (0: nearest-neighbor, 1: bi-linear)')
    parser.add_argument('--channel_axis', type=int, default=None, help='Given axis will not be rescaled')
    parser.add_argument('--run', default=False, action='store_true', help='Update the TIFF file (default is dry run)')
    args = parser.parse_args()

    if not args.run:
        print(long_help)

    result, scale, result_size = shrink_tiff(
        filepath=args.filepath,
        max_size=args.max_size,
        anti_aliasing=args.anti_aliasing,
        order=args.order,
        channel_axis=args.channel_axis,
        dry=not args.run,
    )

    result_size_str = humanize.naturalsize(result_size, binary=True)
    if args.run:
        print(f'File {args.filepath} has been shrinked to {result_size_str} and a resolution of {str(result.shape)} pixels by a factor of {scale}.')
    else:
        print(f'File {args.filepath} will be shrinked to {result_size_str} and a resolution of {str(result.shape)} pixels by a factor of {scale}.')
        print('\nUse the switch "--run" to update the file.')
