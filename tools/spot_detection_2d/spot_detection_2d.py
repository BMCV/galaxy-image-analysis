"""
Copyright 2021 Biomedical Computer Vision Group, Heidelberg University.
Author: Qi Gao (qi.gao@bioquant.uni-heidelberg.de)

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import imageio
from skimage.filters import gaussian
from skimage.feature import peak_local_max

import numpy as np
import pandas as pd
import argparse


def getbr(xy,img,nb,firstn):
    ndata = xy.shape[0]
    br = np.empty((ndata,1))
    for j in range(ndata):
        br[j] = np.NaN
        if not np.isnan(xy[j,0]):
            timg  = img[xy[j,1]-nb-1:xy[j,1]+nb,xy[j,0]-nb-1:xy[j,0]+nb]
            br[j] = np.mean(np.sort(timg,axis=None)[-firstn:])
    return br


def spot_detection(fn_in,fn_out,frame_1st=1,frame_end=0,typ_br='smoothed',th=10,ssig=1,bd=10):
    ims_ori = imageio.mimread(fn_in)
    ims_smd = np.zeros((len(ims_ori),ims_ori[0].shape[0],ims_ori[0].shape[1]),dtype='float64')
    if frame_end == 0 or frame_end > len(ims_ori):
        frame_end = len(ims_ori)
        
    for i in range(frame_1st-1,frame_end):
        ims_smd[i,:,:] = gaussian(ims_ori[i].astype('float64'), sigma=ssig)
    ims_smd_max = np.max(ims_smd)
    
    txyb_all = np.array([]).reshape(0,4)
    for i in range(frame_1st-1,frame_end):
        tmp = np.copy(ims_smd[i,:,:])
        tmp[tmp < th*ims_smd_max/100] = 0
        coords = peak_local_max(tmp,min_distance=1)
        idx_to_del = np.where((coords[:,0]<=bd) | (coords[:,0]>=tmp.shape[0]-bd) | (coords[:,1]<=bd) | (coords[:,1]>=tmp.shape[1]-bd))
        coords = np.delete(coords,idx_to_del[0],axis=0)
        xys = coords[:,::-1]
                
        if   typ_br == 'smoothed':
            intens = getbr(xys,ims_smd[i,:,:],0,1);
        elif typ_br == 'robust':
            intens = getbr(xys,ims_ori[i],1,4);
        else:
            intens = getbr(xys,ims_ori[i],0,1);
        
        txyb = np.concatenate(((i+1)*np.ones((xys.shape[0],1)),xys,intens),axis=1)
        txyb_all = np.concatenate((txyb_all, txyb), axis=0)
        
    df = pd.DataFrame()
    df['FRAME']     = txyb_all[:,0].astype(int)
    df['POS_X']     = txyb_all[:,1].astype(int)
    df['POS_Y']     = txyb_all[:,2].astype(int)
    df['INTENSITY'] = txyb_all[:,3]
    df.to_csv(fn_out, index = False, float_format='%.2f', sep="\t")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spot detection based on local maxima")
    parser.add_argument("fn_in",      help="Name of input image sequence (stack)")
    parser.add_argument("fn_out",     help="Name of output file to save the coordinates and intensities of detected spots")
    parser.add_argument("frame_1st", type=int, help="Index for the starting frame to detect spots (1 for first frame of the stack)")
    parser.add_argument("frame_end", type=int, help="Index for the last frame to detect spots (0 for the last frame of the stack)")
    parser.add_argument("typ_intens",          help="smoothed or robust (for measuring the intensities of spots)")
    parser.add_argument("thres", type=float,   help="Percentage of the global maximal intensity for thresholding candidate spots")
    parser.add_argument("ssig", type=float,    help="Sigma of the Gaussian filter for noise suppression")
    parser.add_argument("bndy", type=int,      help="Number of pixels (Spots close to image boundaries will be ignored)")
    args = parser.parse_args()
    spot_detection(args.fn_in,
                   args.fn_out,
                   frame_1st = args.frame_1st,
                   frame_end = args.frame_end,
                   typ_br = args.typ_intens,
                   th     = args.thres,
                   ssig   = args.ssig,
                   bd     = args.bndy)
