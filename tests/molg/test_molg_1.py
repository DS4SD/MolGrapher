"""Test methods in module molgrapher."""

import unittest
import json

from molgrapher.models.molgrapher_model import MolgrapherModel
from PIL import Image
from pathlib import Path


class TestMolG(unittest.TestCase):
    """Test the methods in molgrapher."""

    def test_molg_1(self):
        model = MolgrapherModel()
        image_paths = [
            Path(__file__).parent /"../images/CN119912385A-5-1.png",
            Path(__file__).parent /"../images/CN119912385A-5-2.png"
        ]
        batch = [Image.open(image_path) for image_path in image_paths]
        annos = model.predict(batch)
        self.assertEqual(annos[0].get("smi"), "C")
        self.assertEqual(annos[1].get("smi"), "C=CC(=CC1=CN2(CC2)N2=CC=C(C(C)=O)C=C=21)C(C)=O")
