import argparse
import sys
import warnings
import numpy as np
import skimage.io
import skimage.color
import skimage.util
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis

convOptions = {
           'hed2rgb' : lambda img_raw: skimage.color.hed2rgb(img_raw),
           'hsv2rgb' : lambda img_raw: skimage.color.hsv2rgb(img_raw),
           'lab2lch' : lambda img_raw: skimage.color.lab2lch(img_raw),
           'lab2rgb' : lambda img_raw: skimage.color.lab2rgb(img_raw),
           'lab2xyz' : lambda img_raw: skimage.color.lab2xyz(img_raw),
           'lch2lab' : lambda img_raw: skimage.color.lch2lab(img_raw),
           'luv2rgb' : lambda img_raw: skimage.color.luv2rgb(img_raw),
           'luv2xyz' : lambda img_raw: skimage.color.luv2xyz(img_raw),
           'rgb2hed' : lambda img_raw: skimage.color.rgb2hed(img_raw),
           'rgb2hsv' : lambda img_raw: skimage.color.rgb2hsv(img_raw),
           'rgb2lab' : lambda img_raw: skimage.color.rgb2lab(img_raw),
           'rgb2luv' : lambda img_raw: skimage.color.rgb2luv(img_raw),
           'rgb2rgbcie' : lambda img_raw: skimage.color.rgb2rgbcie(img_raw),
           'rgb2xyz' : lambda img_raw: skimage.color.rgb2xyz(img_raw),
           #'rgb2ycbcr' : lambda img_raw: skimage.color.rgb2ycbcr(img_raw),
           #'rgb2yiq' : lambda img_raw: skimage.color.rgb2yiq(img_raw),
           #'rgb2ypbpr' : lambda img_raw: skimage.color.rgb2ypbpr(img_raw),
           #'rgb2yuv' : lambda img_raw: skimage.color.rgb2yuv(img_raw),
           #'rgba2rgb' : lambda img_raw: skimage.color.rgba2rgb(img_raw),
           'rgbcie2rgb' : lambda img_raw: skimage.color.rgbcie2rgb(img_raw),
           'xyz2lab' : lambda img_raw: skimage.color.xyz2lab(img_raw),
           'xyz2luv' : lambda img_raw: skimage.color.xyz2luv(img_raw),
           'xyz2rgb' : lambda img_raw: skimage.color.xyz2rgb(img_raw),
           #'ycbcr2rgb' : lambda img_raw: skimage.color.ycbcr2rgb(img_raw),
           #'yiq2rgb' : lambda img_raw: skimage.color.yiq2rgb(img_raw),
           #'ypbpr2rgb' : lambda img_raw: skimage.color.ypbpr2rgb(img_raw),
           #'yuv2rgb' : lambda img_raw: skimage.color.yuv2rgb(img_raw),    
    
           'rgb_from_hed' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_hed),
           'rgb_from_hdx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_hdx),
           'rgb_from_fgx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_fgx),
           'rgb_from_bex' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_bex),
           'rgb_from_rbd' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_rbd),
           'rgb_from_gdx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_gdx),
           'rgb_from_hax' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_hax),
           'rgb_from_bro' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_bro),
           'rgb_from_bpx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_bpx),
           'rgb_from_ahx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_ahx),
           'rgb_from_hpx' : lambda img_raw: skimage.color.combine_stains(img_raw, skimage.color.rgb_from_hpx),
    
           'hed_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.hed_from_rgb),
           'hdx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.hdx_from_rgb),
           'fgx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.fgx_from_rgb),
           'bex_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.bex_from_rgb),
           'rbd_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.rbd_from_rgb),
           'gdx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.gdx_from_rgb),
           'hax_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.hax_from_rgb),
           'bro_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.bro_from_rgb),
           'bpx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.bpx_from_rgb),
           'ahx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.ahx_from_rgb),
           'hpx_from_rgb' : lambda img_raw: skimage.color.separate_stains(img_raw, skimage.color.hpx_from_rgb),    
    
           'pca' : lambda img_raw: np.reshape(PCA(n_components=3).fit_transform(np.reshape(img_raw, [-1, img_raw.shape[2]])), 
                              [img_raw.shape[0],img_raw.shape[1],-1]),    
           'nmf' : lambda img_raw: np.reshape(NMF(n_components=3, init='nndsvda').fit_transform(np.reshape(img_raw, [-1, img_raw.shape[2]])), 
                              [img_raw.shape[0],img_raw.shape[1],-1]),
           'ica' : lambda img_raw: np.reshape(FastICA(n_components=3).fit_transform(np.reshape(img_raw, [-1, img_raw.shape[2]])), 
                              [img_raw.shape[0],img_raw.shape[1],-1]),
           'fa' : lambda img_raw: np.reshape(FactorAnalysis(n_components=3).fit_transform(np.reshape(img_raw, [-1, img_raw.shape[2]])), 
                              [img_raw.shape[0],img_raw.shape[1],-1])
}

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
parser.add_argument('conv_type', choices=convOptions.keys(), help='conversion type')
args = parser.parse_args() 

img_in = skimage.io.imread(args.input_file.name)[:,:,0:3]
res = convOptions[args.conv_type](img_in)
res[res<-1]=-1
res[res>1]=1

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	res = skimage.util.img_as_uint(res) #Attention: precision loss
	skimage.io.imsave(args.out_file.name, res, plugin='tifffile')
