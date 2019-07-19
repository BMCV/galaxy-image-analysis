import argparse
import pandas as pd


def imagecoordinates_flipyaxis(input_file, output_file, image_height, offset=[0,0]): 
    df = pd.read_csv(input_file, sep='\t')

    x = df.copy().y # create copy instead of view
    df.y = image_height-(df.x + 1) + offset[1] # since maximal y index = height-1 
    df.x = x + offset[0]
    df.to_csv(output_file, sep="\t", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='original file')
    parser.add_argument('out_file_str', type=str, help='string of output file name')
    parser.add_argument('image_height', type=int, help='height of image')
    parser.add_argument('offset_x', type=int, help='offset in x direction (width)', default=0)
    parser.add_argument('offset_y', type=int, help='offset in y direction (height)', default=0)
    args = parser.parse_args()
    imagecoordinates_flipyaxis(args.input_file.name, args.out_file_str, args.image_height, [args.offset_x, args.offset_y])
    # imagecoordinates_flipyaxis(args.input_file.name, args.out_file_str, int(args.image_height))
