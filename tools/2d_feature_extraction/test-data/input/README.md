# Overview of the test images

## Label maps

### `input9.zarr`:

- axes: `ZYX`
- resolution: `(2, 100, 100)`
- dtype: `bool`
- binary image
- metadata:
  - resolution: `(1.0, 1.0)`
  - z-spacing: `1.0`
  - unit: `um`

### `input11.tiff`

- axes: `YX`
- resolution: `(265, 329)`
- dtype: `uint16`
- binary image

### `input12.png`

- axes: `YX`
- resolution: `(58, 64)`
- dtype: `uint8`
- labels: `0...24`

## Intensity images

### `input1_uint8.tiff`:

- axes: `YX`
- resolution: `(265, 329)`
- dtype: `uint8`
- metadata: none

### `input3_uint16.tiff`:

- axes: `YXC`
- resolution: `(58, 64, 3)`
- dtype: `uint16`
- metadata:
  - resolution: `(2.0, 1.0)`
  - unit: `mm`

### `input8_zyx.zarr`:

- axes: `ZYX`
- resolution: `(2, 100, 100)`
- dtype: `float64`
- metadata:
  - resolution: `(1.0, 1.0)`
  - z-spacing: `1.0`
  - unit: `um`

### `input10.zarr`:

- axes: `CYX`
- resolution: `(2, 64, 64)`
- dtype: `uint8`
- metadata:
  - resolution: `(1.0, 1.0)`
  - z-spacing: `1.0`
  - unit: `um`
