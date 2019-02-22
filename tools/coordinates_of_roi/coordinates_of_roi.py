import argparse
import pandas as pd
import skimage.io
import skimage.color
import warnings

def get_pixel_values(im, pixel_table, white_obj, threshold, offset=[0,0]):
    data = skimage.io.imread(im)
    if len(data.shape) == 3 and data.shape[-1] > 1:
        data = skimage.color.rgb2grey(data)
    x = []
    y = []
    img_height = data.shape[0]
    img_width = data.shape[1]
    for j in range(img_width):
        for i in range(img_height):
            if white_obj == False:
                if data[i,j] <= threshold:
                    x.append(j + offset[0])
                    y.append(height-(i+1) + offset[1])
            elif data[i,j] >= threshold:
                    x.append(j + offset[0])
                    y.append(height-(i+1) + offset[1])
                    
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df.to_csv(pixel_table, sep="\t", index = False)
                     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Create a csv table with Coordinates of the ROI")
    parser.add_argument("im", help = "Paste path to out.png (output created by transformation)")
    parser.add_argument("pixel_table", help = "Paste path to file in which list with all pixles > threshold should be saved")
    parser.add_argument('offset_x', type=int, help='offset in x direction (width)', default=0)
    parser.add_argument('offset_y', type=int, help='offset in y direction (height)', default=0)
    parser.add_argument("--white_obj", dest = "white_obj", default=False, help = "If set objects in image are white otherwise black", action = "store_true")
    parser.add_argument("--threshold", dest = "threshold", default = 0.5, help = "Enter desired threshold value", type = float)
    
    args = parser.parse_args()  
    # with warnings.catch_warnings():
	#     warnings.simplefilter("ignore") 
    get_pixel_values(args.im, args.pixel_table, args.white_obj, args.threshold, [args.offset_x, args.offset_y])     
