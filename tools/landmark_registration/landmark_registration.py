from skimage.measure import ransac
from skimage.transform import AffineTransform
import pandas as pd
import numpy as np
import argparse

def landmark_registration(points_file1, points_file2, out_file, residual_threshold=2, max_trials=100, delimiter="\t"):
    points1 = pd.read_csv(points_file1, delimiter=delimiter)
    points2 = pd.read_csv(points_file2, delimiter=delimiter)

    src = np.concatenate([np.array(points1['x']).reshape([-1,1]), np.array(points1['y']).reshape([-1,1])], axis=-1)
    dst = np.concatenate([np.array(points2['x']).reshape([-1,1]), np.array(points2['y']).reshape([-1,1])], axis=-1)

    model = AffineTransform()
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                   residual_threshold=residual_threshold, max_trials=max_trials)
    pd.DataFrame(model_robust.params).to_csv(out_file, header = None, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate transformation from points")
    parser.add_argument("points_file1", help="Paste path to src points")
    parser.add_argument("points_file2", help="Paste path to dst points")
    parser.add_argument("warp_matrix", help="Paste path to warp_matrix.csv that should be used for transformation")
    parser.add_argument("--residual_threshold", dest="residual_threshold", help="Maximum distance for a data point to be classified as an inlier.", type=float, default=2)
    parser.add_argument("--max_trials", dest="max_trials", help="Maximum number of iterations for random sample selection.", type=int, default=100)
    args = parser.parse_args()
    landmark_registration(args.points_file1, args.points_file2, args.warp_matrix, residual_threshold=args.residual_threshold, max_trials=args.max_trials)
