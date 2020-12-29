from scipy.linalg import lstsq
import pandas as pd
import numpy as np
import argparse

def landmark_registration_ls(pts_f1, pts_f2, out_f, delimiter="\t"):
    
    points1 = pd.read_csv(pts_f1, delimiter=delimiter)
    points2 = pd.read_csv(pts_f2, delimiter=delimiter)

    src = np.concatenate([np.array(points1['x']).reshape([-1,1]), 
                          np.array(points1['y']).reshape([-1,1])], 
                         axis=-1)
    dst = np.concatenate([np.array(points2['x']).reshape([-1,1]), 
                          np.array(points2['y']).reshape([-1,1])], 
                         axis=-1)

    A = np.zeros((src.size,6))
    A[0:src.shape[0],[0,1]] = src
    A[0:src.shape[0],2] = 1
    A[src.shape[0]:,[3,4]] = src
    A[src.shape[0]:,5] = 1
    b = dst.T.flatten().astype('float64')
    x = lstsq(A,b)
    
    tmat = np.eye(3)
    tmat[0,:] = x[0].take([0,1,2])
    tmat[1,:] = x[0].take([3,4,5])
    pd.DataFrame(tmat).to_csv(out_f, header=None, index=False, sep="\t")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate transformation from points using least squares")
    
    parser.add_argument("fn_pts1", help="File name src points")
    parser.add_argument("fn_pts2", help="File name dst points")
    parser.add_argument("fn_tmat", help="File name transformation matrix")
    args = parser.parse_args()
    
    landmark_registration_ls(args.fn_pts1, args.fn_pts2, args.fn_tmat)


