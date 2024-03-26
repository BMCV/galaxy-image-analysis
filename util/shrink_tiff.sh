#!/bin/bash

set -e

script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cwd=$( pwd )
cd "$script_path"

if [ ! -d ".shrink_tiff.venv" ]
then
    python -m venv ".shrink_tiff.venv"
    source ".shrink_tiff.venv/bin/activate"
    pip install "scikit-image>=0.19" "humanize" "tifffile"
else
    source ".shrink_tiff.venv/bin/activate"
fi

cd "$cwd"
python "$script_path/shrink_tiff.py" $*
