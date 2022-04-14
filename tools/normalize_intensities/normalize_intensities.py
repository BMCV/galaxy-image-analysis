import argparse
import sys
import skimage.io


def normalize(input_file, output_file, eps=1e-10):
    img_in = skimage.io.imread(input_file, plugin='tifffile')
    out_img = (img_in - img_in.mean()) / (img_in.std() + eps)
    skimage.io.imsave(output_file, out_img, plugin='tifffile')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file (Tiff)')
    parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (Tiff)')
    args = parser.parse_args()

    normalize(args.input_file.name, args.out_file.name)
