import logging
from pathlib import Path

from docling_core.types.doc import (
    PictureItem,
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
    
from molgrapher.docling.picture_molecule_annotator_pipeline import PictureMoleculeAnnotatorPipelineOptions, PictureMoleculeAnnotatorPipeline

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../vw-tests/docling/"
    input_doc_path = data_folder / "CN119912385A-5.pdf"

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

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"The model populated the `data` portion of picture {element.self_ref}:\n{element.meta}"
            )


if __name__ == "__main__":
    main()
