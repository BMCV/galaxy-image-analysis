from skimage.transform import ProjectiveTransform
from scipy.ndimage import map_coordinates
import numpy as np
import pandas as pd
import argparse


def _stackcopy(a, b):
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_img_coords_batch(coord_map, shape, dtype=np.float64, batch_size=1000000):
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


def warp_coords_batch(coord_map, coords, dtype=np.float64, batch_size=1000000):
    tf_coords = coords.astype(np.float32)[:, ::-1]

    for i in range(0, (tf_coords.shape[0]//batch_size)+1):
        tf_coords[batch_size*i:batch_size*(i+1)] = coord_map(tf_coords[batch_size*i:batch_size*(i+1)])

    return tf_coords[:, ::-1]
 

def transform(fn_roi_coords, fn_warp_matrix, fn_out):
    data = pd.read_csv(fn_roi_coords, delimiter="\t")
    all_data = np.array(data)
    
    nrows = all_data.shape[0]
    ncols = all_data.shape[1]
    roi_coords = all_data.take([0,1],axis=1).astype('int64')
    
    tol = 10
    moving = np.zeros(np.max(roi_coords,axis=0)+tol, dtype=np.uint32)
    idx_roi_coords = (roi_coords[:,0]-1) * moving.shape[1] + roi_coords[:,1] - 1
    moving.flat[idx_roi_coords] = np.transpose(np.arange(nrows)+1)
    
    trans_matrix = np.array(pd.read_csv(fn_warp_matrix, delimiter="\t", header=None))
    transP = ProjectiveTransform(matrix=trans_matrix)
    roi_coords_warped_direct = warp_coords_batch(transP, roi_coords)
    shape_fixed = np.round(np.max(roi_coords_warped_direct,axis=0)).astype(roi_coords.dtype)+tol
    
    transI = ProjectiveTransform(matrix=np.linalg.inv(trans_matrix))
    img_coords_warped = warp_img_coords_batch(transI, shape_fixed)
    
    moving_warped = map_coordinates(moving, img_coords_warped, order=0, mode='constant', cval=0)
    idx_roi_coords_warped = np.where(moving_warped>0)
    roi_annots_warped = moving_warped.compress((moving_warped>0).flat)
    
    df = pd.DataFrame()
    col_names = data.columns.tolist()
    df['x'] = idx_roi_coords_warped[0] + 1
    df['y'] = idx_roi_coords_warped[1] + 1
    if ncols>2:
        for i in range(2,ncols):
            df[col_names[i]] = all_data[:,i].take(roi_annots_warped)
    df.to_csv(fn_out, index = False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform coordinates")
    parser.add_argument("coords", help="Paste path to .csv with coordinates (and labels) to transform (tab separated)")
    parser.add_argument("warp_matrix", help="Paste path to .csv that should be used for transformation (tab separated)")
    parser.add_argument("out", help="Paste path to file in which transformed coords (and labels) should be saved (tab separated)")
    args = parser.parse_args()
    transform(args.coords, args.warp_matrix, args.out)
