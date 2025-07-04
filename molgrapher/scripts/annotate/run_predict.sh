#!/usr/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python3 "$parent_path"/predict_molgrapher.py \
    --input-images-paths "$parent_path"/input_images_paths_default.jsonl \
    --no-force-cpu \
    --save-mol-folder "$parent_path"/../../../data/predictions/default/ \
    --assign-stereo 
