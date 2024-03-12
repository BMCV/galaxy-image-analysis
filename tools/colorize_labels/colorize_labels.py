import argparse

import numpy as np
import skimage.io
import skimage.util
import superdsm.render


def color_hex_to_rgb_tuple(hex):
    if hex.startswith('#'):
        hex = hex[1:]
    return (
        int(hex[0:2], 16),
        int(hex[2:4], 16),
        int(hex[4:6], 16),
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--bg_label', type=int)
    parser.add_argument('--bg_color', type=str)
    parser.add_argument('--cmap', type=str, default='hsv')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    im = skimage.io.imread(args.input)
    im = np.squeeze(im)
    assert im.ndim == 2

    im_colorized = superdsm.render.colorize_labels(
        labels=im,
        bg_label=args.bg_label,
        cmap=args.cmap,
        bg_color=np.divide(color_hex_to_rgb_tuple(args.bg_color), 255),
        shuffle=args.seed,
    )

    im_colorized = skimage.util.img_as_ubyte(im_colorized)
    skimage.io.imsave(args.output, im_colorized)
