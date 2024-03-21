# Utility scripts

## Use case: An input TIFF file is too large

Assuming that the TIFF file is an RGB file:

```bash
../../util/shrink_tiff.sh test-data/input2_float.tiff --channel_axis 2
```

The script preserves the brightness and range of values of the image.

## Use case: An output TIFF file is too large

Assuming that the TIFF file is an RGB file:

```bash
../../util/shrink_tiff.sh test-data/res_preserve_values.tiff --channel_axis 2
```

This shrinks the file, but the *input files* also need to be shrinked accordingly.
The output of the above command tells the exact scaling factor that was used.
Denote this factor, and then run:

```bash
../../util/scale_image.sh test-data/input2_float.tiff --channel_axis 2 --scale SCALE
```

where you replace `SCALE` by the denoted factor. This also works for PNG files.

The script preserves the brightness and range of values of the image.
