"""
Copyright 2022 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.io
import skimage.measure
import tifffile


def read_im_gray(fn):
    img = skimage.io.imread(fn)
    nDims = len(img.shape)
    assert nDims in [2, 3], 'this tool only supports single 2D images'
    if nDims == 3 and img.shape[-1] in [3, 4]:
        img = skimage.color.rgb2gray(img)
    if len(img.shape) == 3:
        return img[:, :, 0]
    else:
        return img


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
    im1 = skimage.io.imread(im1_fn)
    im2 = skimage.io.imread(im2_fn)
    assert im1.shape == im2.shape, 'Two images should have the same dimension'
    out_im = (1 - alpha) * im1 + alpha * im2
    if len(im1.shape) > 3:
        tifffile.imwrite(out_fn, out_im.astype(im1.dtype), imagej=True)
    else:
        skimage.io.imsave(out_fn, out_im.astype(im1.dtype))  # format of output is the same as input


def seg_contour(im1_fn, im2_fn, out_fn, linewidth=0.3, color='#ff0000', show_label=False):
    img = skimage.io.imread(im1_fn)
    label = skimage.io.imread(im2_fn)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    if show_label:
        for reg in skimage.measure.regionprops(label):
            ax.text(reg.centroid[1], reg.centroid[0], str(reg.label), color=color)

    if len(img.shape) == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(img)
    plt.contour(label, linewidths=linewidth, colors=color)
    fig.canvas.print_png(out_fn)  # output is RGB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay two images")
    parser.add_argument("im1", help="The first image")
    parser.add_argument("im2", help="The second image")
    parser.add_argument("out", help="Output image")
    parser.add_argument('--method', dest='method', default='coloc_vis', help='How to overlay images')
    parser.add_argument('--alpha', dest='alpha', default=0.5, type=float, help='Blending weight')
    parser.add_argument('--thickness', dest='thickness', default=0.3, type=float, help='Contour thickness')
    parser.add_argument('--color', dest='color', default='#FFFF00', help='Contour color')
    parser.add_argument('--show_label', dest='show_label', action='store_true', help='Plot label')
    args = parser.parse_args()

    if args.method == 'coloc_vis':
        coloc_vis(args.im1, args.im2, args.out)
    elif args.method == 'blending':
        blending(args.im1, args.im2, args.out, alpha=args.alpha)
    elif args.method == 'seg_contour':
        seg_contour(args.im1, args.im2, args.out,
                    linewidth=args.thickness,
                    color=args.color,
                    show_label=args.show_label)
