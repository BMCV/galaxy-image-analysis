"""
Copyright 2021 Biomedical Computer Vision Group, Heidelberg University.
Author: Qi Gao (qi.gao@bioquant.uni-heidelberg.de)

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def compute_curve(x, par):
    assert len(par) in [2, 3], 'The degree of curve must be 1 or 2!'
    if len(par) == 3:
        return par[0] * x + par[1] + par[2] * x ** 2
    elif len(par) == 2:
        return par[0] * x + par[1]


def fitting_err(par, xx, seq, penalty):
    assert penalty in ['abs', 'quadratic', 'student-t'], 'Unknown penalty function!'
    curve = compute_curve(xx, par)
    if penalty == 'abs':
        err = np.sqrt(np.abs(curve - seq))
    elif penalty == 'quadratic':
        err = (curve - seq)
    elif penalty == 'student-t':
        a = 1000
        b = 0.001
        err = np.sqrt(a * np.log(1 + (b * (curve - seq)) ** 2))
    return err


def curve_fitting(seq, degree=2, penalty='abs'):
    assert len(seq) > 5, 'Data is too short for curve fitting!'
    assert degree in [1, 2], 'The degree of curve must be 1 or 2!'

    par0 = [(seq[-1] - seq[0]) / len(seq), np.mean(seq[0:3])]
    if degree == 2:
        par0.append(-0.001)

    xx = np.array([i for i in range(len(seq))]) + 1
    x = np.array(par0, dtype='float64')
    result = least_squares(fitting_err, x, args=(xx, seq, penalty))

    return compute_curve(xx, result.x)


def curve_fitting_io(fn_in, fn_out, degree=2, penalty='abs', alpha=0.01):
    # read all sheets
    xl = pd.ExcelFile(fn_in)
    nSpots = len(xl.sheet_names)
    data_all = []
    for i in range(nSpots):
        df = pd.read_excel(xl, xl.sheet_names[i])
        data_all.append(np.array(df))
    col_names = df.columns.tolist()
    ncols_ori = len(col_names)

    # curve_fitting
    diff = np.array([], dtype=('float64'))
    for i in range(nSpots):
        seq = data_all[i][:, -1]
        seq_fit = seq.copy()
        idx_valid = ~np.isnan(seq)
        seq_fit[idx_valid] = curve_fitting(seq[idx_valid], degree=degree, penalty=penalty)
        data_all[i] = np.concatenate((data_all[i], seq_fit.reshape(-1, 1)), axis=1)
        if alpha > 0:
            diff = np.concatenate((diff, seq_fit[idx_valid] - seq[idx_valid]))

    # add assistive curve
    if alpha > 0:
        sorted_diff = np.sort(diff)
        fac = 1 - alpha / 2
        sig3 = sorted_diff[int(diff.size * fac)]
        for i in range(nSpots):
            seq_assist = data_all[i][:, -1] + sig3
            data_all[i] = np.concatenate((data_all[i], seq_assist.reshape(-1, 1)), axis=1)

    # write to file
    with pd.ExcelWriter(fn_out) as writer:
        for i in range(nSpots):
            df = pd.DataFrame()
            for c in range(ncols_ori):
                df[col_names[c]] = data_all[i][:, c]
            df['CURVE'] = data_all[i][:, ncols_ori]
            if alpha > 0:
                df['CURVE_A'] = data_all[i][:, ncols_ori + 1]
            df.to_excel(writer, sheet_name=xl.sheet_names[i], index=False, float_format='%.2f')
        writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit (1st- or 2nd-degree) polynomial curves to data points")
    parser.add_argument("fn_in", help="File name of input data points (xlsx)")
    parser.add_argument("fn_out", help="File name of output fitted curves (xlsx)")
    parser.add_argument("degree", type=int, help="Degree of the polynomial function")
    parser.add_argument("penalty", help="Optimization objective for fitting")
    parser.add_argument("alpha", type=float, help="Significance level for generating assistive curves")
    args = parser.parse_args()
    curve_fitting_io(args.fn_in, args.fn_out, args.degree, args.penalty, args.alpha)
