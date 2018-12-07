from skimage.transform import ProjectiveTransform
import numpy as np
import pandas as pd
import argparse


def warp_coords_batch(coord_map, coords, dtype=np.float64, batch_size=1000000):
    tf_coords = coords.astype(np.float32)

    for i in range(0, (tf_coords.shape[0]//batch_size+1)):
        tf_coords[batch_size*i:batch_size*(i+1)] = coord_map(tf_coords[batch_size*i:batch_size*(i+1)])

    return np.unique(np.round(tf_coords).astype(coords.dtype),axis=0)


def transform(coords, warp_matrix, out):
    indices = np.array(pd.read_csv(coords, delimiter="\t"))
    a_matrix = np.array(pd.read_csv(warp_matrix, delimiter=",", header=None))
    
    trans = ProjectiveTransform(matrix=a_matrix)
    warped_coords = warp_coords_batch(trans, indices)

    df = pd.DataFrame()
    df['x'] = warped_coords[:,0]
    df['y'] = warped_coords[:,1]
    df.to_csv(out, index = False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform coordinates")
    parser.add_argument("coords", help="Paste path to .csv with coordinates to transform (tab separated)")
    parser.add_argument("warp_matrix", help="Paste path to .csv that should be used for transformation (, separated)")
    parser.add_argument("out", help="Paste path to file in which transformed coords should be saved (tab separated)")
    args = parser.parse_args()
    transform(args.coords, args.warp_matrix, args.out)
