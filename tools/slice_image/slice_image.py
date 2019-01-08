import argparse
import sys
import warnings
import numpy as np
import random
import os.path
import skimage.io
import skimage.util
import skimage.feature
from scipy.stats import entropy as scipy_entropy

def slice_image(input_file, out_folder, label=None, label_out_folder=None, window_size=64, 
                stride=1, bg_thresh=1, limit_slices=False, n_thresh=5000, seed=None):
    #TODO NOT Implemented:process labels
    # --> label and label_out_folder useless so far

    # primarily for testing purposes:
    if seed is not None:
        random.seed(seed)

    img_raw = skimage.io.imread(input_file)
    if len(img_raw.shape) == 2:
        img_raw = np.expand_dims(img_raw, 3)

    with warnings.catch_warnings(): # ignore FutureWarning
        warnings.simplefilter("ignore")
        patches_raw = skimage.util.view_as_windows(img_raw, (window_size, window_size, img_raw.shape[2]), step=stride)
        patches_raw = patches_raw.reshape([-1, window_size, window_size, img_raw.shape[2]])

        filename = os.path.splitext(os.path.basename(input_file))[0]
        new_path = out_folder+"/"+filename+"_%d.tiff" # does the backslash work for every os?
        
        #samples for thresholding the amount of slices
        sample = random.sample(range(patches_raw.shape[0]), n_thresh)

        for i in range(0, patches_raw.shape[0]):
            # TODO improve
            sum_image = np.sum(patches_raw[i], 2)/img_raw.shape[2]
            total_entr = np.var(sum_image.reshape([-1]))

            if bg_thresh > 0:
                sum_image = skimage.util.img_as_uint(sum_image)
                g = skimage.feature.greycomatrix(sum_image, [1,2], [0, np.pi/2], nnormed=True, symmetric=True)
                hom = np.var(skimage.feature.greycoprops(g, prop='homogeneity'))
                if hom > bg_thresh: #0.0005
	                continue
            
            if limit_slices == True:
                if i in sample:
                    res = skimage.util.img_as_uint(patches_raw[i]) #Attention: precision loss
                    skimage.io.imsave(new_path % i, res, plugin='tifffile')
            else:
                res = skimage.util.img_as_uint(patches_raw[i]) #Attention: precision loss
                skimage.io.imsave(new_path % i, res, plugin='tifffile')
                    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('r'), help='input file')
    parser.add_argument('out_folder', help='out folder')
    parser.add_argument('--label', dest='label_file', default=None, help='auxiliary label file to split in the same way')
    parser.add_argument('--label_out_folder', dest='label_out_folder', default=None, help='label out folder')
    parser.add_argument('--stride', dest='stride', type=int, default=1, help='applied stride')
    parser.add_argument('--window_size', dest='window_size', type=int, default=64, help='size of resulting patches')
    parser.add_argument('--bg_thresh', dest='bg_thresh', type=float, default=0, help='skip patches without information using a treshold')
    parser.add_argument('--limit_slices', dest='limit_slices', type=bool, default=False, help='limit amount of slices')
    parser.add_argument('--n_thresh', dest='n_thresh', type=int, default=5000, help='amount of slices')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='seed for random choice of limited slices')
    args = parser.parse_args()

    slice_image(args.input_file.name, args.out_folder,
                label=args.label_file, label_out_folder=args.label_out_folder,
                stride=args.stride, window_size=args.window_size, bg_thresh=args.bg_thresh, 
                limit_slices=args.limit_slices, n_thresh=args.n_thresh, seed=args.seed)
