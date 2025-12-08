import argparse
import json
import pathlib

import pydicom
import yaml


def dicom_to_text(
    dcm_filepath: pathlib.Path,
    text_filepath: pathlib.Path,
):
    dcm = pydicom.dcmread(dcm_filepath, stop_before_pixels=True)
    data = dcm.to_json_dict()
    fmt_suffix = text_filepath.suffix.lower()

    # Export JSON
    if fmt_suffix == '.json':
        with text_filepath.open('w') as json_fp:
            json.dump(data, json_fp, indent=2)

    # Export YAML
    elif fmt_suffix in ('.yml', '.yaml'):
        with text_filepath.open('w') as fp:
            yaml.dump(data, fp)

    else:
        raise ValueError(f'Unknown suffix: "{fmt_suffix}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dcm', type=pathlib.Path)
    parser.add_argument('text', type=pathlib.Path)
    args = parser.parse_args()

    dicom_to_text(
        args.dcm,
        args.text,
    )
