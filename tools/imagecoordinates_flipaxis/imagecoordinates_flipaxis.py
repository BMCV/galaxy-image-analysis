import argparse
import pandas as pd


def imagecoordinates_flipyaxis(input_file, output_file, image_height):
    df = pd.read_csv(input_file, sep='\t')

    x = df.copy().y # create copy instead of view
    df.y = image_height-(df.x + 1) # since maximal y index = height-1 
    df.x = x
    df.to_csv(output_file, sep="\t", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='original file')
    parser.add_argument('out_file_str', type=str, help='string of output file name')
    # parser.add_argument('image_height', help='height of image')
    parser.add_argument('image_height', type=int, help='height of image')
    args = parser.parse_args()
    print(args.image_height)
    print(type(args.image_height))
    imagecoordinates_flipyaxis(args.input_file.name, args.out_file_str, args.image_height)
    # imagecoordinates_flipyaxis(args.input_file.name, args.out_file_str, int(args.image_height))
