import argparse
import math
import os
import pathlib

import giatools
import numpy as np
import tifffile


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

        # Peek the number of series in the input file (if it is a TIFF)
        try:
            with tifffile.TiffFile(args.input) as tiff:
                num_tiff_series = len(tiff.series)
                print(f'Found TIFF with {num_tiff_series} series')
        except tifffile.TiffFileError:
            num_tiff_series = 0  # not a TIFF file
            print('Not a TIFF file')

        # If the file is a multi-series TIFF, extract the individual series
        # (for consistency, also accept only a single series if squeezing is requested)
        if num_tiff_series >= 2 or (num_tiff_series == 1 and args.squeeze):
            output = OutputWriter(
                dir_path=args.output,
                num_images=num_tiff_series,
                squeeze=args.squeeze,
                verbose=True,
            )
            for series in range(num_tiff_series):
                img = giatools.Image.read(args.input, series=series)
                output.write(
                    img.squeeze_like(img.original_axes),
                )

        # Otherwise, there is nothing to be split (or squeeze)
        # (the input is either a single-series TIFF or not a TIFF at all)
        elif num_tiff_series == 1:  # input is a single-series TIFF (output = input)
            os.symlink(args.input, args.output / '1.tiff')
        else:  # input is not a TIFF, conversion needed
            img = giatools.Image.read(args.input)
            OutputWriter(
                dir_path=args.output,
                num_images=1,
                squeeze=args.squeeze,
                verbose=False,
            ).write(
                img.squeeze_like(img.original_axes),
            )

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
