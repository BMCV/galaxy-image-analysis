"""
Copyright 2022 Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse
import warnings

import h5py
import numpy as np
import tifffile
import umap


def feature_dimension_reduction(tensor_fn, tiff_fn, nCh=5):
    with h5py.File(tensor_fn, 'r') as hf:
        ts = np.array(hf[list(hf.keys())[0]])

    assert len(ts.shape) == 3 and ts.shape[-1] > nCh, \
        'the input tensor data must be three-dimensional'

    embedding = umap.UMAP(n_components=nCh).fit_transform(np.reshape(ts, (-1, ts.shape[-1])))
    img = np.reshape(embedding, (ts.shape[0], ts.shape[1], -1)).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tifffile.imwrite(tiff_fn + '.tiff', np.transpose(img, (2, 0, 1)), imagej=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dimensionality reduction for features (channels) of 3D tensor using UMAP")
    parser.add_argument("tensor_fn", help="Path to the 3D tensor data")
    parser.add_argument("nCh", type=int, help="The reduced dimension of features")
    parser.add_argument("tiff_fn", help="Path to the output file")
    args = parser.parse_args()
    feature_dimension_reduction(args.tensor_fn, args.tiff_fn, args.nCh)
