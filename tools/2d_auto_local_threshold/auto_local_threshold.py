import argparse
import sys
import skimage.io
import skimage.filters
import skimage.util 

threshOptions = {
    'gaussian': lambda img_raw, bz: skimage.filters.threshold_local(img_raw, bz, method='gaussian'),
    'mean': lambda img_raw, bz: skimage.filters.threshold_local(img_raw, bz, method='mean'),
    'median': lambda img_raw, bz: skimage.filters.threshold_local(img_raw, bz, method='median')
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment Foci')
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
    parser.add_argument('block_size', type=int, default=5, help='Odd size of pixel neighborhood which is used to calculate the threshold value')
    parser.add_argument('thresh_type', choices=threshOptions.keys(), help='thresholding method')
    parser.add_argument('dark_background', default=True, type=bool, help='True if background is dark')
    args = parser.parse_args()

    img_in = skimage.io.imread(args.input_file.name)
    thresh = threshOptions[args.thresh_type](img_in, args.block_size)

    if args.dark_background:
        res = img_in > thresh
    else:
        res = img_in <= thresh

    res = skimage.util.img_as_uint(res)
    skimage.io.imsave(args.out_file.name, res, plugin="tifffile")
