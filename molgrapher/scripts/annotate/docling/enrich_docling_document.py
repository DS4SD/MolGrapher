#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json 
import os 
import tempfile
import argparse
import base64
import glob
from PIL import Image
from io import BytesIO

from docling_core.types.doc import DoclingDocument

from molgrapher.models.molgrapher_model import MolgrapherModel


def enrich_docling_document_with_smiles(docling_document: DoclingDocument, docling_document_directory_path: str = "") -> DoclingDocument:
    """Enrich a docling document with MolGrapher predictions.

    Extract pictures classified as "chemistry_molecular_structure" in a docling document. 
    Process them using MolGrapher. Save as pictures annotations the MolGrapher's SMILES predictions
    """
    # Extract images
    images = {}
    for element in docling_document.pictures:
        # Select only chemical-structures
        if (element.annotations[0].predicted_classes[0].class_name != "chemistry_molecular_structure"):
            continue
        image_bytes = base64.b64decode(str(element.image.uri).split(",")[1])
        images[element.self_ref] = Image.open(BytesIO(image_bytes))

    # Save images to disk
    docling_document_directory_tmp = None
    if docling_document_directory_path == "":
        docling_document_directory_tmp = tempfile.TemporaryDirectory()
        docling_document_directory_path = docling_document_directory_tmp.name
    images_folder = docling_document_directory_path + "/images/"
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
        for element in docling_document.pictures:
            if element.self_ref == self_ref:
                docling_annotation = {
                    "provenance": f"{annotation['annotator']['program']}-{annotation['annotator']['version']}", 
                    "smiles": annotation["smi"], 
                    "confidence": annotation["conf"],
                }
                element.annotations.append(docling_annotation)
    if docling_document_directory_tmp:
        docling_document_directory_tmp.cleanup()

    return docling_document

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docling-document-directory-path', type = str, default = os.path.dirname(__file__) + f"/../../../../data/docling_documents/US9259003_p4/")
    args = parser.parse_args()   

    # Read docling document
    docling_document_filename = [e for e in args.docling_document_directory_path.split("/") if e != ""][-1] + ".json"
    docling_document_path = args.docling_document_directory_path + "/" + docling_document_filename 
    docling_document = DoclingDocument.load_from_json(docling_document_path)

    # Enrich document with SMILES
    docling_document = enrich_docling_document_with_smiles(docling_document, args.docling_document_directory_path)

    # Save docling document 
    with open(docling_document_path[:-5] + "_enriched.json", "w", encoding="utf-8") as fp:
        fp.write(json.dumps(docling_document.export_to_dict()))

if __name__ == "__main__":
    main()