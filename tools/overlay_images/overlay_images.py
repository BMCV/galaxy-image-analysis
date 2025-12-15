"""
Copyright 2022-2023 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import argparse

import giatools.io
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.io
import skimage.measure
import tifffile
from contours import ContourPaint


def read_im_gray(fn):
    img = giatools.io.imread(fn)
    nDims = len(img.shape)
    assert nDims in [2, 3], 'this tool only supports single 2D images'
    if nDims == 3 and img.shape[-1] in [3, 4]:
        img = skimage.color.rgb2gray(img)
    if len(img.shape) == 3:
        return img[:, :, 0]
    else:
        return img


def get_rgb8_copy(img, fp_lower, fp_upper):
    img = np.squeeze(img)
    assert img.ndim == 2 or (img.ndim == 3 and img.shape[-1] in (3, 4))
    assert fp_lower == 'min' or abs(float(fp_lower)) < np.inf  # 'min' or number
    assert fp_upper == 'max' or abs(float(fp_upper)) < np.inf  # 'max' or number

    # Convert from floating point
    if str(img.dtype).startswith('float'):
        a = img.min() if fp_lower == 'min' else float(fp_lower)
        b = img.max() if fp_upper == 'max' else float(fp_upper)

        if a > b:
            raise ValueError(
                f'Lower bound ({a:g}) must be less than upper bound ({b:g}).'
            )
        if a == b:
            raise ValueError(
                'Floating point conversion is undefined'
                ' because lower and upper bounds are identical.'
            )

        # Perform linear mapping to [0, 1]
        img = img.clip(a, b)
        img = (img - a) / (b - a)

        # Convert to uint8
        img = np.round(img * 255).astype(np.uint8)

    # Convert from uint16
    elif img.dtype == np.uint16:
        img = (img // 256).astype(np.uint8)

    # Other dtypes than float, uint8, uint16 are not supported
    elif img.dtype != np.uint8:
        raise ValueError(f'unknown dtype: {img.dtype}')

    if img.ndim == 2:
        result = np.dstack([img] * 3).copy()
    else:
        result = img[:, :, :3].copy()
    assert result.dtype == np.uint8, result.dtype
    return result


def coloc_vis(in_red_fn, in_green_fn, out_fn):
    im1 = read_im_gray(in_red_fn)
    im2 = read_im_gray(in_green_fn)
    assert im1.shape == im2.shape, 'Two images should have the same dimension'

    vmin = np.min([np.min(im1), np.min(im2)])
    scal = 255.0 / (np.max([np.max(im1), np.max(im2)]) - vmin)

    out_im = np.zeros(im1.shape + (3,), dtype=np.ubyte)
    out_im[:, :, 0] = (im1 - vmin) * scal
    out_im[:, :, 1] = (im2 - vmin) * scal
    skimage.io.imsave(out_fn, out_im)  # output is RGB


def blending(im1_fn, im2_fn, out_fn, alpha=0.5):
    im1 = giatools.io.imread(im1_fn)
    im2 = giatools.io.imread(im2_fn)
    assert im1.shape == im2.shape, 'Two images should have the same dimension'
    out_im = (1 - alpha) * im1 + alpha * im2
    if len(im1.shape) > 3:
        tifffile.imwrite(out_fn, out_im.astype(im1.dtype), imagej=True)
    else:
        skimage.io.imsave(out_fn, out_im.astype(im1.dtype))  # format of output is the same as input


def seg_contour(im1_fn, im2_fn, out_fn, fp_lower, fp_upper, linewidth, color='#ff0000', show_label=False, label_color='#ffff00'):
    img = giatools.io.imread(im1_fn)
    labels = giatools.io.imread(im2_fn)

    result = get_rgb8_copy(img, fp_lower, fp_upper)
    cp = ContourPaint(labels, linewidth, where='center')
    color_rgb = np.multiply(255, matplotlib.colors.to_rgb(color))

    for label in np.unique(labels):
        if label > 0:
            cc = (labels == label)
            bd = cp.get_contour_mask(cc)
            for i in range(3):
                result[:, :, i][bd] = color_rgb[i]

    if show_label:
        fig = plt.figure(figsize=np.divide(img.shape[:2][::-1], 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(result)
        for reg in skimage.measure.regionprops(labels):
            ax.text(reg.centroid[1], reg.centroid[0], str(reg.label), color=label_color)
        fig.canvas.print_png(out_fn)

    else:
        skimage.io.imsave(out_fn, result)  # format of output is RGB8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay two images")
    parser.add_argument("im1", help="The first image")
    parser.add_argument("im2", help="The second image")
    parser.add_argument("out", help="Output image")
    parser.add_argument('--method', dest='method', default='coloc_vis', help='How to overlay images')
    parser.add_argument('--fp_lower', default='0', type=str, help='Lower bound for floating point conversion')
    parser.add_argument('--fp_upper', default='1', type=str, help='Upper bound for floating point conversion')
    parser.add_argument('--alpha', dest='alpha', default=0.5, type=float, help='Blending weight')
    parser.add_argument('--thickness', dest='thickness', default=2, type=int, help='Contour thickness')
    parser.add_argument('--color', dest='color', default='#FF0000', help='Contour color')
    parser.add_argument('--show_label', dest='show_label', action='store_true', help='Show labels')
    parser.add_argument('--label_color', dest='label_color', default='#FFFF00', help='Label color')
    args = parser.parse_args()

    if args.method == 'coloc_vis':
        coloc_vis(args.im1, args.im2, args.out)
    elif args.method == 'blending':
        blending(args.im1, args.im2, args.out, alpha=args.alpha)
    elif args.method == 'seg_contour':
        seg_contour(
            args.im1,
            args.im2,
            args.out,
            fp_lower=args.fp_lower,
            fp_upper=args.fp_upper,
            linewidth=args.thickness,
            color=args.color,
            show_label=args.show_label,
            label_color=args.label_color,
        )
