import argparse
import math
import pathlib

import giatools
import giatools.io
import numpy as np


class OutputWriter:

    def __init__(self, dir_path: pathlib.Path, num_images: int, squeeze: bool, verbose: bool):
        print(f'Writing {num_images} image(s)')
        decimals = math.ceil(math.log10(1 + num_images))
        self.output_filepath_pattern = str(dir_path / f'%0{decimals}d.tiff')
        self.last_idx = 0
        self.squeeze = squeeze
        self.verbose = verbose

    def write(self, img: giatools.Image):
        self.last_idx += 1
        if self.squeeze:
            img = img.squeeze()
        if self.last_idx == 1 or self.verbose:
            prefix = f'Output {self.last_idx}' if self.verbose else 'Output'
            print(f'{prefix} axes:', img.axes)
            print(f'{prefix} shape:', img.data.shape)
        img.write(self.output_filepath_pattern % self.last_idx)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('axis', type=str, choices=list(giatools.default_normalized_axes) + ['S', ''])
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--squeeze', action='store_true', default=False)
    args = parser.parse_args()

    # If splitting a file that contains multiple images...
    if args.axis == '':

        # Peek the number of images
        num_images = giatools.io.peek_num_images_in_file(args.input)
        print(f'Found {num_images} image(s) in file')

        # Extract the individual images
        output = OutputWriter(
            dir_path=args.output,
            num_images=num_images,
            squeeze=args.squeeze,
            verbose=(num_images > 1),
        )
        for position in range(num_images):
            img = giatools.Image.read(args.input, position=position, normalize_axes=None)
            output.write(img)

    # If splitting along an image axes...
    else:

        # Validate and normalize input parameters
        axis = args.axis.replace('S', 'C')

        # Read input image with normalized axes
        img_in = giatools.Image.read(args.input)
        print('Input image axes:', img_in.original_axes)
        print('Input image shape:', img_in.squeeze_like(img_in.original_axes).data.shape)

        # Determine the axis to split along
        axis_pos = img_in.axes.index(axis)

        # Perform the splitting
        arr = np.moveaxis(img_in.data, axis_pos, 0)
        output = OutputWriter(
            dir_path=args.output,
            num_images=arr.shape[0],
            squeeze=args.squeeze,
            verbose=False,
        )
        for img_idx, img in enumerate(arr):
            img = np.moveaxis(img[None], 0, axis_pos)

            # Construct the output image, remove axes added by normalization
            img_out = giatools.Image(
                data=img,
                axes=img_in.axes,
                metadata=img_in.metadata,
            ).squeeze_like(
                img_in.original_axes,
            )

            # Save the result (write stdout during first iteration)
            output.write(img_out)
