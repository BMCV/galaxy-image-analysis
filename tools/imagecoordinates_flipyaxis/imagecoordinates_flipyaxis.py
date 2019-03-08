import argparse
import pandas as pd


def imagecoordinates_flipyaxis(input_file, output_file, image_height):
    df = pd.read_csv(input_file, sep='\t')
    df.y=image_height-df.y
    df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='original file')
    parser.add_argument('out_file_str', type=str, help='string of output file name')
    parser.add_argument('image_height', help='height of image')
    args = parser.parse_args()
    imagecoordinates_flipyaxis(args.input_file.name, args.out_file_str, int(args.image_height))
