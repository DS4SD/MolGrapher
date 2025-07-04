#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import glob
import json
import os
import tempfile
from io import BytesIO

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import PictureMoleculeData
from PIL import Image

from molgrapher.models.molgrapher_model import MolgrapherModel


def enrich_docling_document_with_smiles(
    docling_document: DoclingDocument,
) -> DoclingDocument:
    """Enrich a docling document with MolGrapher predictions.

    Extract pictures classified as "chemistry_molecular_structure" in a docling document.
    Process them using MolGrapher. Save as pictures annotations the MolGrapher's SMILES predictions.
    """
    # Extract images
    images = {}
    for element in docling_document.pictures:
        # Select only chemical-structures
        if (
            element.annotations[0].predicted_classes[0].class_name
            == "chemistry_molecular_structure"
        ):
            images[element.self_ref] = element.get_image(doc=docling_document)

    # Run MolGrapher
    model = MolgrapherModel({"visualize": False})
    annotations = model.predict_batch(images.values())

    # Save annotations in docling document
    for self_ref, annotation in zip(images.keys(), annotations):
        for element in docling_document.pictures:
            if element.self_ref != self_ref:
                continue

            molecule_annotation = PictureMoleculeData(
                provenance=f"{annotation['annotator']['program']}-{annotation['annotator']['version']}",
                smi=annotation["smi"],
                confidence=annotation["conf"],
                class_name="chemistry_molecular_structure",
                segmentation=[
                    (0, 0),
                    (0, element.image.size.height),
                    (element.image.size.width, element.image.size.height),
                    (element.image.size.width, 0),
                ],
            )
            element.annotations.append(molecule_annotation)

    return docling_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docling-document-directory-path",
        type=str,
        default=os.path.dirname(__file__)
        + f"/../../../../data/docling_documents/US9259003_p4/",
    )
    args = parser.parse_args()

    # Read docling document
    docling_document_filename = [
        e for e in args.docling_document_directory_path.split("/") if e != ""
    ][-1] + ".json"
    docling_document_path = (
        args.docling_document_directory_path + "/" + docling_document_filename
    )
    docling_document = DoclingDocument.load_from_json(docling_document_path)

    # Enrich document with SMILES
    docling_document = enrich_docling_document_with_smiles(docling_document)

    # Save docling document
    with open(
        docling_document_path[:-5] + "_enriched.json", "w", encoding="utf-8"
    ) as fp:
        fp.write(json.dumps(docling_document.export_to_dict()))


if __name__ == "__main__":
    main()
