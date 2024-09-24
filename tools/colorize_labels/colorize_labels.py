import argparse

import giatools.io
import matplotlib.colors as mpl
import networkx as nx
import numpy as np
import scipy.ndimage as ndi
import skimage.io
import skimage.util


def color_hex_to_rgb_tuple(hex):
    if hex.startswith('#'):
        hex = hex[1:]
    return (
        int(hex[0:2], 16),
        int(hex[2:4], 16),
        int(hex[4:6], 16),
    )


def build_label_adjacency_graph(im, radius, bg_label):
    G = nx.Graph()
    for label in np.unique(im):

        if label == bg_label:
            continue

        G.add_node(label)

        cc = (im == label)
        neighborhood = (ndi.distance_transform_edt(~cc) <= radius)
        adjacent_labels = np.unique(im[neighborhood])

        for adjacent_label in adjacent_labels:

            if adjacent_label == bg_label or adjacent_label <= label:
                continue

            G.add_edge(label, adjacent_label)
            G.add_edge(adjacent_label, label)

    return G


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--bg_label', type=int)
    parser.add_argument('--bg_color', type=str)
    parser.add_argument('--radius', type=int)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    # Load image and normalize
    im = giatools.io.imread(args.input)
    im = np.squeeze(im)
    assert im.ndim == 2

    # Build adjacency graph of the labels
    G = build_label_adjacency_graph(im, args.radius, args.bg_label)
    print('---')

    # Apply greedy coloring
    graph_coloring = nx.greedy_color(G)
    unique_colors = frozenset(graph_coloring.values())

    # Assign colors to nodes based on the greedy coloring
    graph_color_to_mpl_color = dict(zip(unique_colors, mpl.TABLEAU_COLORS.values()))
    node_colors = [graph_color_to_mpl_color[graph_coloring[n]] for n in G.nodes()]

    # Render result
    bg_color_rgb = color_hex_to_rgb_tuple(args.bg_color)
    result = np.dstack([np.full(im.shape, bg_color_rgb[ch], np.uint8) for ch in range(3)])
    for label, label_color in zip(G.nodes(), node_colors):

        cc = (im == label)
        label_color = color_hex_to_rgb_tuple(label_color)
        for ch in range(3):
            result[:, :, ch][cc] = label_color[ch]

    # Write result image
    skimage.io.imsave(args.output, result)
