import argparse

import numpy as np
import skimage.transform
import skimage.io


def scale_image(filepath, **kwargs):
    im = skimage.io.imread(filepath)
    res = skimage.transform.rescale(im, preserve_range=True, **kwargs)

    # Preserve the `dtype` so that both brightness and range of values is preserved
    if res.dtype != im.dtype:
        if np.issubdtype(im.dtype, np.integer):
            res = res.round()
        res = res.astype(im.dtype)

    skimage.io.imsave(filepath, res)
    return res


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Image filepath')
    parser.add_argument('--scale', type=float, help='Scaling factor', required=True)
    parser.add_argument('--anti_aliasing', default=False, action='store_true', help='Use anti-aliasing')
    parser.add_argument('--order', type=int, default=0, help='Order of interpolation (0: nearest-neighbor, 1: bi-linear)')
    parser.add_argument('--channel_axis', type=int, default=None, help='Given axis will not be rescaled')
    args = parser.parse_args()

    res = scale_image(
        filepath=args.filepath,
        scale=args.scale,
        anti_aliasing=args.anti_aliasing,
        order=args.order,
        channel_axis=args.channel_axis,
    )

    print(f'File {args.filepath} has been scaled to a resolution of {str(res.shape)} pixels.')
