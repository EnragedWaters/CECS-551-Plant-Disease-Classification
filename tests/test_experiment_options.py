"""Check optional augmentation and its recorded recipe without plant datasets.

Run: python -m unittest discover -s tests -p test_experiment_options.py -v
These checks use synthetic images, not model training or research results.
"""

import unittest

import torch
from PIL import Image

from plant_disease.data import augmentation_recipe, make_augmentation, oriented_rgb


class AugmentationTests(unittest.TestCase):
    def test_none_disables_random_changes(self):
        self.assertIsNone(make_augmentation())
        self.assertEqual(augmentation_recipe("none"), "none")
        with self.assertRaises(ValueError):
            make_augmentation("typo")

    def test_recorded_recipe_matches_enabled_transforms(self):
        recipe = augmentation_recipe("basic-v1")
        crop, flip, rotation, jitter = make_augmentation("basic-v1").transforms
        self.assertEqual(list(crop.scale), recipe["random_resized_crop"]["scale"])
        self.assertEqual(list(crop.ratio), recipe["random_resized_crop"]["ratio"])
        self.assertEqual(crop.interpolation.value, recipe["random_resized_crop"]["interpolation"])
        self.assertEqual(crop.antialias, recipe["random_resized_crop"]["antialias"])
        self.assertEqual(flip.p, recipe["horizontal_flip"]["p"])
        self.assertEqual(rotation.degrees, recipe["rotation"]["degrees"])
        self.assertEqual(list(rotation.fill), recipe["rotation"]["fill"])
        self.assertEqual(jitter.hue, (-.05, .05))

    def test_seeded_changes_repeat_and_accept_grayscale_and_transparency(self):
        transform = make_augmentation("basic-v1")
        for mode in ("L", "RGBA"):
            image = oriented_rgb(Image.new(mode, (32, 24)))
            torch.manual_seed(551)
            first = transform(image)
            torch.manual_seed(551)
            second = transform(image)
            self.assertEqual(first.mode, "RGB")
            self.assertEqual(first.size, (224, 224))
            self.assertEqual(first.tobytes(), second.tobytes())


if __name__ == "__main__":
    unittest.main()
