#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json 
import os 
import argparse
import base64
import glob
from PIL import Image
from io import BytesIO

from molgrapher.models.molgrapher_model import MolgrapherModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docling-document-directory-path', type = str, default = os.path.dirname(__file__) + f"/../../../../data/docling_documents/US9259003_p4/")
    args = parser.parse_args()   

    # Read docling document
    docling_document_filename = [e for e in args.docling_document_directory_path.split("/") if e != ""][-1] + ".json"
    docling_document_path = args.docling_document_directory_path + "/" + docling_document_filename 
    with open(docling_document_path, "r", encoding="utf-8") as fp:
        docling_document = json.load(fp) 

    # Extract images
    images = {}
    for element in docling_document["pictures"]:
        # Select only chemical-structure 
        if (element["annotations"][0]["predicted_classes"][0]["class_name"] != "chemistry_molecular_structure"):
            continue

        image_bytes = base64.b64decode(element["image"]["uri"].split(",")[1])
        images[element["self_ref"]] = Image.open(BytesIO(image_bytes))

    # Save images to disk
    images_folder = args.docling_document_directory_path + "/images/"
    if not(os.path.exists(images_folder)):
        os.mkdir(images_folder)

    for i, image in enumerate(images.values()):
        image.save(images_folder + f"/{i}.png")

    # Run MolGrapher 
    input_images_paths = [p for p in glob.glob(images_folder + "/*")]
    model = MolgrapherModel({"visualize": False})
    annotations = model.predict_batch(input_images_paths) 

    # Save annotations in docling document
    for (self_ref, image), annotation in zip(images.items(), annotations):
        for element in docling_document["pictures"]:
            if element["self_ref"] == self_ref:
                docling_annotation = {
                    "provenance": f"{annotation['annotator']['program']}-{annotation['annotator']['version']}", 
                    "smiles": annotation["smi"], 
                    "confidence": annotation["conf"],
                }
                element["annotations"].append(docling_annotation)

    # Save docling document 
    with open(docling_document_path[:-5] + "_enriched.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps(docling_document))

if __name__ == "__main__":
    main()