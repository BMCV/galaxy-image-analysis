import argparse

import numpy as np
import scipy.ndimage as ndi
import skimage.io
import skimage.morphology as morph


def create_selem(args):
    """
    Creates structuring element based on commandline arguments.
    """
    assert args.selem_shape in (
        'square',
        'disk',
    )

    if args.selem_shape == 'square':
        return np.ones((args.selem_size, args.selem_size))

    elif args.selem_shape == 'disk':
        return morph.disk(args.selem_size)


def apply_operation(args, im):
    """
    Applies morphological operation to a 2-D single-channel image.
    """
    assert im.ndim == 2
    selem = create_selem(args)
    values_count = len(np.unique(im))
    if values_count <= 2:
        im_proxy = np.zeros(im.shape, bool)
        im_proxy[im == im.max()] = True
        result_proxy = apply_binary_operation(args, im_proxy, selem)
        result = np.full(im.shape, im.min(), im.dtype)
        result[result_proxy] = im.max()
        return result
    else:
        return apply_intensity_based_operation(args, im, selem)


def apply_intensity_based_operation(args, im, selem):
    operations = {
        'erosion': ndi.grey_erosion,
        'dilation': ndi.grey_dilation,
        'opening': ndi.grey_opening,
        'closing': ndi.grey_closing,
    }
    if args.operation in operations:
        operation = operations[args.operation]
        return operation(input=im, structure=selem)
    else:
        raise ValueError(f'Operation "{args.operation}" not supported for this image type ({im.dtype}).')


def apply_binary_operation(args, im, selem):
    operations = {
        'erosion': ndi.binary_erosion,
        'dilation': ndi.binary_dilation,
        'opening': ndi.binary_opening,
        'closing': ndi.binary_closing,
        'fill_holes': ndi.binary_fill_holes,
    }
    operation = operations[args.operation]
    return operation(input=im, structure=selem)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', type=str)
    parser.add_argument('--selem-shape', type=str)
    parser.add_argument('--selem-size', type=int)
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    im = skimage.io.imread(args.input)
    assert im.ndim in (2, 3), 'Input image must be two-dimensional and either single-channel or multi-channel.'

    if im.ndim == 2:
        im_result = apply_operation(args, im)

    else:
        ch_result_list = []
        for ch_idx in range(im.shape[2]):
            ch = im[:, :, ch_idx]
            ch_result = apply_operation(args, ch)
            ch_result_list.append(ch_result)
        im_result = np.dstack(ch_result_list)

    skimage.io.imsave(args.output, im_result)
