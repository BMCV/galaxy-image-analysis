from .output import get_output

import numpy as np
import cvxpy as cp
import scipy.sparse


def solve_maxsetpack(objects, out=None):
    accepted_objects  = []  ## primal variable
    remaining_objects = list(objects)

    out = get_output(out)
    w = lambda c: c.energy
    while len(remaining_objects) > 0:

        # choose the best remaining object
        best_object = max(remaining_objects, key=w)
        accepted_objects.append(best_object)

        # discard conflicting objects
        remaining_objects = [c for c in remaining_objects if len(c.footprint & best_object.footprint) == 0]

    out.write(f'MAXSETPACK - GREEDY accepted objects: {len(accepted_objects)}')
    return accepted_objects
