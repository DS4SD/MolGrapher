import logging
from pathlib import Path

from docling_core.types.doc import (
    PictureItem,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    
from molgrapher.docling.picture_molecule_annotator import PictureMoleculeAnnotator, PictureMoleculeAnnotatorOptions


class PictureMoleculeAnnotatorPipelineOptions(PdfPipelineOptions):
    do_picture_annotator: bool = True


class PictureMoleculeAnnotatorPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: PictureMoleculeAnnotatorPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: PictureMoleculeAnnotatorPipeline

        self.enrichment_pipe = [
            PictureMoleculeAnnotator(
                enabled=pipeline_options.do_picture_annotator,
                options=PictureMoleculeAnnotatorOptions(),
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
            )
        ]

    @classmethod
    def get_default_options(cls) -> PictureMoleculeAnnotatorPipelineOptions:
        return PictureMoleculeAnnotatorPipelineOptions()
