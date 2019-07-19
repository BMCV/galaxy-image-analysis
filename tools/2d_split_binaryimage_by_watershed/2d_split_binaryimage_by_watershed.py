import argparse
import sys
import skimage.io
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import skimage.util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split binaryimage by watershed')
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('min_distance', type=int, default=100, help='Minimum distance to next object')
    args = parser.parse_args()

    img_in = skimage.io.imread(args.input_file.name)
    distance = ndi.distance_transform_edt(img_in)
    local_maxi = peak_local_max(distance, 
                                indices=False, 
                                min_distance=args.min_distance, 
                                labels=img_in)
    markers = ndi.label(local_maxi)[0]
    res = watershed(-distance, markers, mask=img_in)

    res = skimage.util.img_as_uint(res)
    skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
