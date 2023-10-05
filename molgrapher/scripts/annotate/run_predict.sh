#!/usr/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python3 predict_molgrapher.py \
    --input-images-paths input_images_paths_default.jsonl \
    --no-force-cpu \
    --num-threads-pytorch 10 \
    --num-processes-mp 10 \
    --chunk-size 10000 \
    --save-mol-folder "$dir"/../../../data/predictions/molgrapher/default/ \
    --no-assign-stereo 
    