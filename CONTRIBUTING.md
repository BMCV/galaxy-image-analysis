# Contributing

This document is the attempt to collect some rough rules for tools to follow in this repository, to facilitate their consistency and interoperability. This document is an extension to the [Naming and Annotation Conventions for Tools in the Image Community in Galaxy](https://doi.org/10.37044/osf.io/w8dsz) and compatibility should be maintained. This document is work in progress.

**How to Contribute:**

* Make sure you have a [GitHub account](https://github.com/signup/free)
* Make sure you have git [installed](https://help.github.com/articles/set-up-git)
* Fork the repository on [GitHub](https://github.com/BMCV/galaxy-image-analysis/fork)
* Make the desired modifications—consider using a [feature branch](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
* Try to stick to the [Conventions for Tools in the Image Community](https://doi.org/10.37044/osf.io/w8dsz) and the [IUC standards](http://galaxy-iuc-standards.readthedocs.org/en/latest/) whenever you can
* Make sure you have added the necessary tests for your changes and they pass.
* Open a [pull request](https://help.github.com/articles/using-pull-requests) with these changes.

## Terminology

**Label maps** are images with pixel-level annotations, usually corresponding to distinct image regions (e.g., objects). We avoid the terms *label image* and *labeled image*, since these can be easily confused with image-level annotations (instead of pixel-level). The labels (pixel values) must uniquely identify the labeled image regions (i.e. labels must be unique, even for non-adjacent image regions). If a label semantically corresponds to the image background, that label should be 0.

**Binary images** are a special case of label maps with only two labels (e.g., image background and image foreground). To facilitate visual perception, the foreground label should correspond to white (value 255 for `uint8` images and value 65535 for `uint16` images), since background corresponds to the label 0, which is black.

**Intensity images** are images which generally are *not* label maps (and thus neither binary images).

## File types

In tool wrappers which use a Python script, image loading should be performed by using the `giatools` package ([docs](https://giatools.readthedocs.io)).
This gives you out-of-the-box support for a veriety of file types, including TIFF, PNG, JPG, and OME-Zarr.

Another advantage is that `giatools` gives you out-of-the-box support for 3-D images, multi-channel-images, and other rather exotic format flavors, even if the wrapped image processing/analysis operation only supports 2-D image data.
For example, if the wrapped operation only support single-channel 2-D images, the following code structure can be used to process all slices of 3-D images, all channels of multi-channel images, and so on:
```python
image = giatools.Image.read(args.input)
for source_slice, section in image.iterate_jointly('XY'):
    ...  # process the 2-D `section` of the image
```
See the [docs](https://giatools.readthedocs.io/en/latest/giatools.image.html#giatools.image.Image.iterate_jointly) for details.

Instead of using functions from the global namespace of the `numpy` package to process the `section` or `image.data`, it is preferred to use the methods of the `section` and `image.data` objects directly. This is because, for some file types (e.g., OME-Zarr), those objects will be Dask arrays, and although these should fit in transparently into the NumPy framework, using implementation-specific methods promises greater efficiency of the computational performance, especially for large datasets.

Tools with **label map inputs** should accept PNG and TIFF files. Tools with **label map outputs** should produce either `uint16` single-channel PNG or `uint16` single-channel TIFF. Using `uint8` instead of `uint16` is also acceptable, if there definetely are no more than 256 different labels. Using `uint8` should be preferred for binary images.

> [!NOTE]  
> It is a common misconception that PNG files must be RGB or RGBA, and that only `uint8` pixel values are supported. For example, the `cv2` module (OpenCV) can be used to create single-channel PNG files, or PNG files with `uint16` pixel values. Such files can then be read by `giatools.Image.read` or `skimage.io.imread` without issues (however, `skimage.io.imwrite` seems not to be able to write such PNG files).

Tools with **intensity image inputs** should accept PNG and TIFF files. Tools with **intensity image outputs** can be any data type and either PNG or TIFF. Image outputs meant for visualization (e.g., segmentation overlays, charts) should be PNG.

## Testing

We recommend using macros for verification of image outputs. The macros are loaded as follows:
```xml
<macros>
    <import>tests.xml</import>
</macros>
```

### Testing binary image outputs

For testing of **binary image outputs** we recommend using the `mae` metric (mean absolute error). The default value for `eps` of 0.01 is rather strict, and for 0/1 binary images this asserts that at most 1% of the image pixels are labeled differently:
```xml
<expand macro="tests/binary_image_diff" name="output" value="output.tif" ftype="tiff"/>
```
For 0/255 binary images, the same 1% tolerance would be achieved by increasing `eps` to 2.25. The macro also ensures that the image contains two distinct label values.

### Testing label map outputs

For testing of non-binary **label map outputs** with interchangeable labels, we recommend using the `iou` metric (one minus the *intersection over the union*). With the default value of `eps` of 0.01, this asserts that there is no labeled image region with an *intersection over the union* of less than 99%:
```xml
<expand macro="tests/label_image_diff" name="output" value="output.tif" ftype="tiff"/>
```
Label 0 is commonly connotated as the image background, and is not interchangable by default. Use `pin_labels=""` to make it interchangable.

### Testing intensity image outputs

For testing of **intensity image outputs** we recommend the `rms` metric (root mean square), because it is very sensitive to large pixel value differences, but tolerates smaller differences:
```xml
<expand macro="tests/intensity_image_diff" name="output" value="output.tif" ftype="tiff"/>
```
For `uint8` and `uint16` images, increasing the default value of `eps` to `1.0` should be tolerable, if required.

## Future extensions

Below is a list of open questions:

- How do we want to cope with multi-channel label maps? For example, do or can we distinguish RGB labels from multi-channel binary masks, which are sometimes used to represent overlapping objects?

Below is a list of changes that need to be reflected in this document:

- As of https://github.com/galaxyproject/galaxy/pull/18951 and https://github.com/galaxyproject/galaxy/pull/20669, rich image metadata is available in Galaxy. This should be used to define [validators on data inputs](https://docs.galaxyproject.org/en/master/dev/schema.html#validators-for-data-and-data-collection-parameters), to prevent incorrect use of tools (e.g., using 3-D images for tools that only support 2-D images) and negative experiences.