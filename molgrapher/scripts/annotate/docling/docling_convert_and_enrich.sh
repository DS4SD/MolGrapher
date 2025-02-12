#!/usr/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Get the PDF path from the user argument
pdf_path="$1"
if [ -z "$pdf_path" ]; then
    echo "Usage: $0 <pdf-path> <docling-document-directory-path>"
    exit 1
fi
pdf_path=$(realpath "$pdf_path")

# Get the PDF path from the user argument
docling_document_directory_path="$2"
if [ -z "$docling_document_directory_path" ]; then
    echo "Usage: $0 <pdf-path> <docling-document-directory-path>"
    exit 1
fi
docling_document_directory_path_abs=$(realpath "$docling_document_directory_path")

# Extract the end file name without the extension
filename=$(basename -- "$pdf_path")
filename_no_ext="${filename%.*}"

# Run the Python scripts 
python3 "$parent_path"/docling_convert.py \
    --pdf-path "$pdf_path" \
    --docling-document-directory-path "$docling_document_directory_path"

python3 "$parent_path"/enrich_docling_document.py \
    --docling-document-directory-path "$docling_document_directory_path"