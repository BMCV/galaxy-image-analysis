import skimage.io
from skimage.transform import ProjectiveTransform
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import argparse



def _stackcopy(a, b):
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b



def warp_coords_batch(coord_map, shape, dtype=np.float64, batch_size=1000000):
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    for i in range(0, (tf_coords.shape[0]//batch_size+1)):
        tf_coords[batch_size*i:batch_size*(i+1)] = coord_map(tf_coords[batch_size*i:batch_size*(i+1)])
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    _stackcopy(coords[1, ...], tf_coords[0, ...])
    _stackcopy(coords[0, ...], tf_coords[1, ...])
    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords



def affine_registration(params,moving,fixed):
    tmat = np.eye(3)
    tmat[0,:] = params.take([0,1,2])
    tmat[1,:] = params.take([3,4,5])
    
    trans = ProjectiveTransform(matrix=tmat)
    warped_coords = warp_coords_batch(trans, fixed.shape)
    t = map_coordinates(moving, warped_coords, mode='reflect')
    
    eI = (t - fixed)**2
    return eI.flatten()



def image_registration(fn_moving, fn_fixed, fn_out, smooth_sigma=1):
    moving = skimage.io.imread(fn_moving,as_gray=True)
    fixed = skimage.io.imread(fn_fixed,as_gray=True)

    moving = gaussian(moving, sigma=smooth_sigma)
    fixed = gaussian(fixed, sigma=smooth_sigma)
    
    x = np.array([1, 0, 0, 0, 1, 0],dtype='float64')
    result = least_squares(affine_registration, x, args=(moving,fixed))
        
    tmat = np.eye(3)
    tmat[0,:] = result.x.take([0,1,2])
    tmat[1,:] = result.x.take([3,4,5])
    
    pd.DataFrame(tmat).to_csv(fn_out, header=None, index=False, sep="\t")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate the transformation matrix")
    parser.add_argument("fn_moving", help="Name of the moving image.png")
    parser.add_argument("fn_fixed",  help="Name of the fixed image.png")
    parser.add_argument("fn_tmat",   help="Name of output file to save the transformation matrix")
    args = parser.parse_args()
    
    image_registration(args.fn_moving, args.fn_fixed, args.fn_tmat)
