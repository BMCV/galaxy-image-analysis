#!/bin/env python

import glob
import pathlib
import yaml


gia_root_path = pathlib.Path(__file__).resolve().parent.parent
gia_root_dir_names = [gia_root_path.name] + [p.name for p in gia_root_path.parents if p.name != '']


for path_str in glob.glob('./**/.shed.yml', recursive=True):

    # Read the .shed.yml file:
    shed_file_path = pathlib.Path(path_str).resolve()
    with shed_file_path.open('r') as shed_file:
        shed = yaml.safe_load(shed_file)
        ts_repo_name = shed['name']

    # Check that the toolshed repo name corresponds to one of the parent directory names:
    parent_dir_names = frozenset(p.name for p in shed_file_path.parents if p.name != '') - frozenset(gia_root_dir_names)
    if ts_repo_name not in parent_dir_names:
        print(
            f'{shed_file_path.relative_to(gia_root_path)}: '
            f'{ts_repo_name} not in {", ".join(str(dir_name) for dir_name in parent_dir_names)}'
        )
