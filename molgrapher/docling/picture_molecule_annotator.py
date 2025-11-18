from collections.abc import Iterable
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureItem,
    PictureMeta,
    MoleculeMetaField,
)
from PIL import Image
from pydantic import BaseModel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

from molgrapher.models.molgrapher_model import MolgrapherModel


class PictureMoleculeAnnotatorOptions(BaseModel):
    """
    Options for configuring the PictureMoleculeAnnotator.

    Attributes
    ----------
    kind : Literal["picture_molecule_annotator"]
        Identifier for the annotator.
    """

    kind: Literal["picture_molecule_annotator"] = "picture_molecule_annotator"


class PictureMoleculeAnnotator(BaseItemAndImageEnrichmentModel):
    """
    A model for annotating molecule pictures in documents.

    This class enriches document pictures with predicted annotations
    based on a predefined set of classes.

    Attributes
    ----------
    enabled : bool
        Whether the annotator is enabled for use.
    options : PictureMoleculeAnnotatorOptions
        Configuration options for the annotator.

    Methods
    -------
    __init__(enabled, options, accelerator_options)
        Initializes the annotator with specified configurations.
    is_processable(doc, element)
        Checks if the given element can be processed by the annotator.
    __call__(doc, element_batch)
        Processes a batch of elements and adds molecule annotations.
    """

    def __init__(
        self,
        enabled: bool,
        options: PictureMoleculeAnnotatorOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """
        Initializes the PictureMoleculeAnnotator.

        Parameters
        ----------
        enabled : bool
            Indicates whether the annotator is enabled.
        options : PictureMoleculeAnnotatorOptions
            Configuration options for the annotator.
        accelerator_options : AcceleratorOptions
            Options for configuring the device and parallelism.
        """
        self.enabled = enabled
        self.options = options

        if self.enabled:
            device = decide_device(accelerator_options.device)

            # enforce only cpu for the moment
            assert device == "cpu"

            self.picture_molecule_annotator = MolgrapherModel()

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """
        Determines if the given element can be processed by the annotator.

        Parameters
        ----------
        doc : DoclingDocument
            The document containing the element.
        element : NodeItem
            The element to be checked.

        Returns
        -------
        bool
            True if the element is a PictureItem and processing is enabled; False otherwise.
        """
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """
        Processes a batch of elements and enriches them with molecule annotation predictions.

        Parameters
        ----------
        doc : DoclingDocument
            The document containing the elements to be processed.
        element_batch : Iterable[ItemAndImageEnrichmentElement]
            A batch of pictures to annotate.

        Returns
        -------
        Iterable[NodeItem]
            An iterable of NodeItem objects after processing. The field
            'meta.molecule' is added containing the annotation for each picture.
        """
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Union[Image.Image, np.ndarray]] = []
        elements: List[PictureItem] = []
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            elements.append(el.item)
            images.append(el.image)

        outputs = self.picture_molecule_annotator.predict(images)

        for item, output in zip(elements, outputs):
            if output.get("smi", "") == "" or \
               output.get("smi", "") == "C":
                continue
            if item.meta is None:
                item.meta = PictureMeta()
            #            
            item.meta.molecule = MoleculeMetaField(
                smi=output.get("smi", ""),
                confidence=output.get("conf", 0.0),
                created_by=output.get("annotator", {}).get("program", "")
            )

            yield item
