"""Test methods in module molgrapher."""

import unittest

from PIL import Image
from pathlib import Path
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import (
        PictureItem,
    )
    from molgrapher.docling.picture_molecule_annotator_pipeline import PictureMoleculeAnnotatorPipelineOptions, PictureMoleculeAnnotatorPipeline
    has_docling = True
except ImportError:
    has_docling = False


@unittest.skipIf(not has_docling, "docling package not installed")
class TestDocling(unittest.TestCase):
    """Test the methods in molgrapher."""

    def test_docl_1(self):

        input_doc_path = Path(__file__).parent / "../images/CN119912385A-5.pdf"

        pipeline_options = PictureMoleculeAnnotatorPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=PictureMoleculeAnnotatorPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
        result = doc_converter.convert(input_doc_path)

        smis = []
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                print(
                    f"The model populated the `data` portion of picture {element.self_ref}:\n{element.meta}"
                )
                if element.meta is not None:
                    smis.append(element.meta.molecule.smi)

        refs = ["[H]C1=CC=C(C=CC(=O)C2=CC=C3C(=C2)C2=C(C=CC(C(=O)C=CC4=CC=C(F)C=C4)=C2)N3CC)C=C1", "CC(=O)C1=CC=N2=[I-](=C1)C1=C(C=CC(C(C)=O)=C1)N21CC1"]
        for smi, ref in zip(smis, refs):
            print(smi)
            self.assertEqual(smi, ref)

