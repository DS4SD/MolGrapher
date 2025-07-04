import argparse
import ast
import os
from time import time

import cv2
import torch

from molgrapher.models.molgrapher_model import MolgrapherModel

os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-images-paths", type=str)
    parser.add_argument(
        "--force-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--force-no-multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
    )
    parser.add_argument("--num-threads-pytorch", type=int, default=10)
    parser.add_argument("--num-processes-mp", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument(
        "--assign-stereo",
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
    )
    parser.add_argument("--align-rdkit-output", type=bool, default=False)
    parser.add_argument(
        "--remove-captions",
        action=argparse.BooleanOptionalAction,
        default=True,
        required=False,
    )
    parser.add_argument("--save-mol-folder", type=str, default="")
    parser.add_argument("--predict", type=bool, default=True)
    parser.add_argument("--preprocess", type=bool, default=True)
    parser.add_argument("--clean", type=bool, default=True)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--visualize_rdkit", type=bool, default=False)
    parser.add_argument(
        "--node-classifier-variant", type=str, default="gc_no_stereo_model"
    )
    parser.add_argument(
        "--config-dataset-graph-path",
        type=str,
        default=os.path.dirname(__file__)
        + "/../../../data/config_dataset_graph_2.json",
    )
    parser.add_argument(
        "--config-training-graph-path",
        type=str,
        default=os.path.dirname(__file__) + "/../../../data/config_training_graph.json",
    )
    parser.add_argument(
        "--config-dataset_keypoint-path",
        type=str,
        default=os.path.dirname(__file__)
        + "/../../../data/config_dataset_keypoint.json",
    )
    parser.add_argument(
        "--config-training-keypoint-path",
        type=str,
        default=os.path.dirname(__file__)
        + "/../../../data/config_training_keypoint.json",
    )
    args = parser.parse_args()

    # Read input images
    with open(args.input_images_paths, "r") as f:
        _input_images_paths = []
        for line in f.readlines():
            _input_images_paths.append(ast.literal_eval(line.strip())["path"])
        print("Number of images to annotate: ", len(_input_images_paths))

    # Instantiate MolGrapher
    model = MolgrapherModel(vars(args))

    # Annotate
    starting_time = time()
    annotations = model.predict_batch(_input_images_paths)
    print(f"Annotation completed in: {round(time() - starting_time, 2)}")


if __name__ == "__main__":
    main()
