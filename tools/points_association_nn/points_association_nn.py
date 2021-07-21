"""
Copyright 2021 Biomedical Computer Vision Group, Heidelberg University.
Author: Qi Gao (qi.gao@bioquant.uni-heidelberg.de)

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""

import argparse

import numpy as np
import pandas as pd
import skimage.util


def disk_mask(imsz, ir, ic, nbpx):
    ys, xs = np.ogrid[-nbpx:nbpx + 1, -nbpx:nbpx + 1]
    se = xs ** 2 + ys ** 2 <= nbpx ** 2
    mask = np.zeros(imsz, dtype=int)
    if ir - nbpx < 0 or ic - nbpx < 0 or ir + nbpx + 1 > imsz[0] or ic + nbpx + 1 > imsz[1]:
        mask = skimage.util.pad(mask, nbpx)
        mask[ir:ir + 2 * nbpx + 1, ic:ic + 2 * nbpx + 1] = se
        mask = skimage.util.crop(mask, nbpx)
    else:
        mask[ir - nbpx:ir + nbpx + 1, ic - nbpx:ic + nbpx + 1] = se
    return mask


def find_nn(cim, icy, icx, nim, nbpx):
    mask = disk_mask(cim.shape, icy, icx, nbpx)
    iys_nim, ixs_nim = np.where(nim * mask)
    if iys_nim.size == 0:
        return np.NaN, np.NaN

    d2 = (icy - iys_nim) ** 2 + (icx - ixs_nim) ** 2
    I1 = np.argsort(d2)
    iy_nim = iys_nim[I1[0]]
    ix_nim = ixs_nim[I1[0]]

    mask = disk_mask(cim.shape, iy_nim, ix_nim, nbpx)
    iys_cim, ixs_cim = np.where(cim * mask)
    d2 = (iy_nim - iys_cim) ** 2 + (ix_nim - ixs_cim) ** 2
    I2 = np.argsort(d2)
    if not iys_cim[I2[0]] == icy or not ixs_cim[I2[0]] == icx:
        return np.NaN, np.NaN

    return iy_nim, ix_nim


def points_linking(fn_in, fn_out, nbpx=6, th=25, minlen=50):
    data = pd.read_csv(fn_in, delimiter="\t")
    all_data = np.array(data)
    assert all_data.shape[1] in [3, 4], 'unknow collum(s) in input data!'

    coords = all_data[:, :3].astype('int64')

    frame_1st = np.min(coords[:, 0])
    frame_end = np.max(coords[:, 0])
    assert set([i for i in range(frame_1st, frame_end + 1)]).issubset(set(coords[:, 0].tolist())), "spots missing at some time point!"

    nSlices = frame_end
    stack_h = np.max(coords[:, 2]) + nbpx
    stack_w = np.max(coords[:, 1]) + nbpx
    stack = np.zeros((stack_h, stack_w, nSlices), dtype='int8')
    stack_r = np.zeros((stack_h, stack_w, nSlices), dtype='float64')

    for i in range(all_data.shape[0]):
        iyxz = tuple(coords[i, ::-1] - 1)
        stack[iyxz] = 1
        stack_r[iyxz] = all_data[i, -1]

    tracks_all = np.array([], dtype=float).reshape(0, nSlices, 4)
    maxv = np.max(stack_r)
    br_max = maxv
    idx_max = np.argmax(stack_r)
    while 1:
        iyxz = np.unravel_index(idx_max, stack.shape)

        spot_br = np.empty((nSlices, 1))
        track = np.empty((nSlices, 3))
        for i in range(nSlices):
            spot_br[i] = np.NaN
            track[i, :] = np.array((np.NaN, np.NaN, np.NaN))

        spot_br[iyxz[2]] = maxv
        track[iyxz[2], :] = np.array(iyxz[::-1]) + 1

        # forward
        icy = iyxz[0]
        icx = iyxz[1]
        for inz in range(iyxz[2] + 1, nSlices):
            iny, inx = find_nn(stack[:, :, inz - 1], icy, icx, stack[:, :, inz], nbpx)
            if np.isnan(iny) and not inz == nSlices - 1:
                iny, inx = find_nn(stack[:, :, inz - 1], icy, icx, stack[:, :, inz + 1], nbpx)
                if np.isnan(iny):
                    break
                else:
                    iny = icy
                    inx = icx
                    stack[iny, inx, inz] = 1
                    stack_r[iny, inx, inz] = stack_r[iny, inx, inz - 1]
            elif np.isnan(iny) and inz == nSlices - 1:
                break

            track[inz, :] = np.array((inz, inx, iny)) + 1
            spot_br[inz] = stack_r[iny, inx, inz]
            icy = iny
            icx = inx

        # backward
        icy = iyxz[0]
        icx = iyxz[1]
        for inz in range(iyxz[2] - 1, -1, -1):
            iny, inx = find_nn(stack[:, :, inz + 1], icy, icx, stack[:, :, inz], nbpx)
            if np.isnan(iny) and not inz == 0:
                iny, inx = find_nn(stack[:, :, inz + 1], icy, icx, stack[:, :, inz - 1], nbpx)
                if np.isnan(iny):
                    break
                else:
                    iny = icy
                    inx = icx
                    stack[iny, inx, inz] = 1
                    stack_r[iny, inx, inz] = stack_r[iny, inx, inz + 1]
            elif np.isnan(iny) and inz == 0:
                break

            track[inz, :] = np.array((inz, inx, iny)) + 1
            spot_br[inz] = stack_r[iny, inx, inz]
            icy = iny
            icx = inx

        for iz in range(nSlices):
            if not np.isnan(track[iz, 0]):
                stack[track[iz, 2].astype(int) - 1, track[iz, 1].astype(int) - 1, iz] = 0
                stack_r[track[iz, 2].astype(int) - 1, track[iz, 1].astype(int) - 1, iz] = 0

        # discard short trajectories
        if np.count_nonzero(~np.isnan(spot_br)) > minlen * (frame_end - frame_1st) / 100:
            tmp = np.concatenate((track, spot_br), axis=1)
            tracks_all = np.concatenate((tracks_all, tmp.reshape(1, -1, 4)), axis=0)

        maxv = np.max(stack_r)
        idx_max = np.argmax(stack_r)
        if maxv < th * br_max / 100:
            break

    with pd.ExcelWriter(fn_out) as writer:
        for i in range(tracks_all.shape[0]):
            df = pd.DataFrame()
            df['FRAME'] = tracks_all[i, :, 0]
            df['POS_X'] = tracks_all[i, :, 1]
            df['POS_Y'] = tracks_all[i, :, 2]
            df['INTENSITY'] = tracks_all[i, :, 3]
            df.to_excel(writer, sheet_name='spot%s' % (i + 1), index=False, float_format='%.2f')
        writer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Association of points in consecutive frames using the nearest neighbor algorithm")
    parser.add_argument("fn_in", help="Name of input file (tsv tabular)")
    parser.add_argument("fn_out", help="Name of output file (xlsx)")
    parser.add_argument("nbpx", type=int, help="Neighborhood size in pixel")
    parser.add_argument("thres", type=float, help="Percentage of the global maximal intensity for thresholding some event")
    parser.add_argument("minlen", type=float, help="Minimum length of tracks (percentage of senquence length)")
    args = parser.parse_args()
    points_linking(args.fn_in, args.fn_out, args.nbpx, args.thres, args.minlen)
