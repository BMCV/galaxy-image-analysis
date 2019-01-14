import argparse
import sys
import skimage.io
import skimage.util
from skimage.measure import regionprops
import scipy.spatial.distance
import numpy as np
import warnings

def merge_n(img, dist=50):
    props = regionprops(img)
    found = False
    for i in range(0, len(props)):
        i_coords = props[i].coords
        for q in range(0, len(props)):
            if i==q:
                continue
            q_coords = props[q].coords
            iq_dist = np.min(scipy.spatial.distance.cdist(i_coords, q_coords, 'euclidean'))
            if iq_dist <= dist:
                props[q].label = props[i].label
                for a_point in range(0, q_coords.shape[0]):
                    img[q_coords[a_point, 0], q_coords[a_point, 1]] = props[i].label
                found = True
    if found:
        merge_n(img, dist)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument(
        '-c',
        dest='cluster_merge',
        type=int,
        required=False,
        default=50,
        help='Distance in pixel of clusters which are merged',
    )
    args = parser.parse_args()

    label_image = skimage.io.imread(args.input_file.name)
    label_image = merge_n(label_image, args.cluster_merge)
    with warnings.catch_warnings():
    	warnings.simplefilter("ignore")
    	res = skimage.util.img_as_uint(label_image)
    	skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
