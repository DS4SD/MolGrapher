#!/usr/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

python3 "$parent_path"/create_input_json.py 
head "$parent_path"/input_images_paths_default.jsonl

bash "$parent_path"/run_predict.sh