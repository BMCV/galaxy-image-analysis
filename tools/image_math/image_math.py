import argparse

import skimage.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--expression', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', default=list(), action='append', required=True)
    args = parser.parse_args()

    inputs = dict()
    im_shape = None
    for input in args.input:
        name, filepath = input.split(':')
        im = skimage.io.imread(filepath)
        assert name not in inputs, 'Input name "{name}" is ambiguous.'
        inputs[name] = im
        if im_shape is None:
            im_shape = im.shape
        else:
            assert im.shape == im_shape, 'Input images differ in size and/or number of channels.'

    result = eval(
        args.expression,
        dict(),  # globals
        inputs,  # locals
    )

    skimage.io.imsave(args.output, result)
