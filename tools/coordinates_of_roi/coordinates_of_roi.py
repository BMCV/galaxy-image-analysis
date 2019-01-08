import argparse
import pandas as pd
import skimage.io
import skimage.color

def get_pixel_values(im, threshold, pixel_table, white_obj):
    data = skimage.io.imread(im)
    if len(data.shape) == 3 and data.shape[-1] > 1:
        data = skimage.color.rgb2grey(data)
    x = []
    y = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if white_obj == True:
                if data[j,i] <= threshold:
                    x.append(i)
                    y.append(j)
            elif data[j,i] >= threshold:
                    x.append(i)
                    y.append(j)
                    
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y    
    df.to_csv(pixel_table, sep="\t", index = False)
                     
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Create a csv table with Coordinates of the ROI")
    parser.add_argument("im", help = "Paste path to out.png (output created by transformation)")
    parser.add_argument("pixel_table", help = "Paste path to file in which list with all pixles > threshold should be saved")
    parser.add_argument("--white_obj", dest = "white_obj", help = "If set objects in image are white otherwise black", action = "store_true", default=True)
    parser.add_argument("--threshold", dest = "threshold", help = "Enter desired threshold value", type = float, default = 0.5)
    args = parser.parse_args()    
    get_pixel_values(args.im, args.threshold, args.pixel_table, args.white_obj)     
