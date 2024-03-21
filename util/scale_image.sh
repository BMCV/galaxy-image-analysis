#!/bin/bash

set -e

script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cwd=$( pwd )
cd "$script_path"

if [ ! -d ".scale_image.venv" ]
then
    python -m venv ".scale_image.venv"
    source ".scale_image.venv/bin/activate"
    pip install "scikit-image>=0.19" "tifffile" "pillow"
else
    source ".scale_image.venv/bin/activate"
fi

cd "$cwd"
python "$script_path/scale_image.py" $*
