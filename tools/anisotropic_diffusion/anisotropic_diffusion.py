import argparse
import sys
import warnings

import skimage.io
import skimage.util
from medpy.filter.smoothing import anisotropic_diffusion

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=argparse.FileType('r'), default=sys.stdin, help='input file')
parser.add_argument('out_file', type=argparse.FileType('w'), default=sys.stdin, help='out file (TIFF)')
parser.add_argument('niter', type=int, help='Number of iterations', default=1)
parser.add_argument('kappa', type=int, help='Conduction coefficient', default=50)
parser.add_argument('gamma', type=float, help='Speed of diffusion', default=0.1)
parser.add_argument('eqoption', type=int, choices=[1, 2], help='Perona Malik diffusion equation', default=1)
args = parser.parse_args()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # to ignore FutureWarning as well

    img_in = skimage.io.imread(args.input_file.name, plugin='tifffile')
    res = anisotropic_diffusion(img_in, niter=args.niter, kappa=args.kappa, gamma=args.gamma, option=args.eqoption)
    res[res < -1] = -1
    res[res > +1] = +1

    res = skimage.util.img_as_uint(res)  # Attention: precision loss

    skimage.io.imsave(args.out_file.name, res, plugin='tifffile')
