#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

from docling.backend.docling_parse_v2_backend import \
    DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=os.path.dirname(__file__) + f"/../../../../data/pdfs/US9259003_p4.pdf",
    )
    parser.add_argument(
        "--docling-document-directory-path",
        type=str,
        default=os.path.dirname(__file__) + f"/../../../../data/pdfs/US9259003_p4/",
    )
    args = parser.parse_args()

    print(args.docling_document_directory_path)
    Path(args.docling_document_directory_path).mkdir(parents=True, exist_ok=True)

    # Convert
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.do_picture_classification = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 4
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseV2DocumentBackend,
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    conv_res = doc_converter.convert(Path(args.pdf_path))

    # Save document
    doc_filename = conv_res.input.file.stem
    with (Path(args.docling_document_directory_path) / f"{doc_filename}.json").open(
        "w", encoding="utf-8"
    ) as fp:
        fp.write(json.dumps(conv_res.document.export_to_dict()))


if __name__ == "__main__":
    main()
