import argparse
import json

import yaml


# This script genereates the config file required by PlantSeg.
# For an overview of the config fields, see:
# https://github.com/kreshuklab/plant-seg/blob/master/examples/config.yaml


def listify(d, k, sep=',', dtype=float):
    if k not in d:
        return
    d[k] = [dtype(token.strip()) for token in str(d[k]).split(sep)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, help='Path to the inputs file', required=True)
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    parser.add_argument('--img_in', type=str, help='Path to the input image', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    with open(args.inputs, 'r') as fp:
        inputs = json.load(fp)

    # Set configuration options from the tool wrapper
    cfg = dict(path=args.img_in)
    for section_name in (
        'preprocessing',
        'cnn_prediction',
        'cnn_postprocessing',
        'segmentation',
        'segmentation_postprocessing',
    ):
        cfg[section_name] = inputs[section_name]

    # Set additional required configuration options
    cfg['preprocessing']['save_directory'] = 'PreProcessing'
    cfg['preprocessing']['crop_volume'] = '[:,:,:]'
    cfg['preprocessing']['filter'] = dict(state=False, type='gaussian', filter_param=1.0)

    cfg['cnn_prediction']['device'] = 'cuda'
    cfg['cnn_prediction']['num_workers'] = args.workers
    cfg['cnn_prediction']['model_update'] = False

    cfg['segmentation']['name'] = 'MultiCut'
    cfg['segmentation']['save_directory'] = 'MultiCut'

    # Parse lists of values encoded as strings as actual lists of values
    listify(cfg['preprocessing'], 'factor')
    listify(cfg['cnn_prediction'], 'patch')

    with open(args.config, 'w') as fp:
        fp.write(yaml.dump(cfg))
