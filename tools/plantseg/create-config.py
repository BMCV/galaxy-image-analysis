import argparse
import json

import yaml


# This script genereates the config file required by PlantSeg.
# For an overview of the config fields, see:
# https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, help='Path to the inputs file', required=True)
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--img_in', type=str, help='Path to the input image', required=True)
    args = parser.parse_args()

    with open(args.inputs, 'r') as fp:
        inputs = json.load(fp)

    cfg = dict(path=args.img_in)
    for section_name in (
        'preprocessing',
        'cnn_prediction',
        'cnn_postprocessing',
        'segmentation',
        'segmentation_postprocessing',
    ):
        cfg[section_name] = inputs[section_name]

    with open(args.config, 'w') as fp:
        fp.write(yaml.dump(cfg))
