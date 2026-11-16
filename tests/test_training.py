"""Check preprocessing and learning modes without datasets or downloads.

Run: python -m unittest discover -s tests -v
The model test uses random starting weights; it is not an accuracy experiment.
"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v3_small

from plant_disease.data import (BACKGROUND, PlantImages, load_source_manifests,
                                prepare_image, resize_with_padding, sha256_file,
                                trial_subset)
from plant_disease.model import batch_norm_state, set_training_mode, state_fingerprint


class ImagePreparationTests(unittest.TestCase):
    def test_wide_image_preserves_content_and_pads(self):
        image = Image.new("RGB", (400, 100), "red")
        result = resize_with_padding(image)
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.getpixel((0, 0)), BACKGROUND)
        pixels = np.asarray(result)
        red = np.all(pixels == (255, 0, 0), axis=2)
        self.assertEqual(int(red.sum()), 224 * 56)
        tensor = prepare_image(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())

    def test_grayscale_and_transparency_are_rgb(self):
        self.assertEqual(resize_with_padding(Image.new("L", (100, 200))).mode, "RGB")
        transparent = Image.new("RGBA", (224, 224), (255, 0, 0, 0))
        self.assertEqual(resize_with_padding(transparent).getpixel((100, 100)), BACKGROUND)

    def test_image_checksum_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            Image.new("RGB", (4, 4)).save(path)
            rows = [{"relative_path": path.name, "sha256": sha256_file(path), "class_id": "0"}]
            data = PlantImages(rows, tmp)
            self.assertEqual(data[0][1], 0)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since the audit"):
                data[0]


class LearningModeTests(unittest.TestCase):
    def test_weights_and_statistics_follow_the_selected_mode(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = mobilenet_v3_small(weights=None)
        model.classifier = nn.Linear(576, 3)
        features_before = state_fingerprint(model.features.state_dict())
        head_before = state_fingerprint(model.classifier.state_dict())
        stats_before = state_fingerprint(batch_norm_state(model))
        images, labels = torch.randn(2, 3, 224, 224), torch.tensor([0, 1])
        for fine_tune in (False, True):
            set_training_mode(model, fine_tune)
            self.assertTrue(all(not m.training for m in model.modules()
                                if isinstance(m, nn.modules.batchnorm._BatchNorm)))
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertEqual(state_fingerprint(batch_norm_state(model)), stats_before)
            self.assertNotEqual(state_fingerprint(model.classifier.state_dict()), head_before)
            if fine_tune:
                self.assertNotEqual(state_fingerprint(model.features.state_dict()), features_before)
            else:
                self.assertEqual(state_fingerprint(model.features.state_dict()), features_before)


class ManifestGuardTests(unittest.TestCase):
    def write_fixture(self, directory, invalid_dataset=False, overlap=False):
        root = Path(directory)
        categories = root / "categories.json"
        categories.write_text(json.dumps({"classes": [{"id": 0, "plantvillage": "A"}]}))
        hashes = {}
        for split in ("train", "validation"):
            row = {"relative_path": f"plantvillage/raw/color/A/{split}.jpg",
                   "dataset": "plantdoc" if invalid_dataset else "plantvillage",
                   "source_split": "train", "split": split, "source_category": "A",
                   "class_name": "A", "class_id": "0", "sha256": split,
                   "group_id": "same" if overlap else split}
            path = root / f"{split}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            hashes[path.name] = sha256_file(path)
        (root / "summary.json").write_text(json.dumps({"categories_sha256": sha256_file(categories),
            "manifest_sha256": hashes, "counts": {"train": 1, "validation": 1}}))
        return root, categories

    def test_source_inputs_work_without_any_test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, categories = self.write_fixture(tmp)
            rows, _, _ = load_source_manifests(root, categories)
            self.assertEqual(set(rows), {"train", "validation"})
            with (root / "train.csv").open("a") as stream:
                stream.write("tampered")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_source_manifests(root, categories)

    def test_target_rows_and_related_overlap_are_rejected(self):
        for options in ({"invalid_dataset": True}, {"overlap": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root, categories = self.write_fixture(tmp, **options)
                with self.assertRaises(ValueError):
                    load_source_manifests(root, categories)

    def assert_invalid_training_row(self, changes):
        """Keep checksums valid so the test exercises row validation itself."""
        with tempfile.TemporaryDirectory() as tmp:
            root, categories = self.write_fixture(tmp)
            path = root / "train.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0].update(changes)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["manifest_sha256"][path.name] = sha256_file(path)
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "mapped training images"):
                load_source_manifests(root, categories)

    def test_training_path_must_match_the_category_folder(self):
        self.assert_invalid_training_row({"relative_path": "plantvillage/raw/color/B/train.jpg"})

    def test_training_pool_rejects_original_validation_and_test_splits(self):
        # Our validation set is carved out of the original training pool.
        for split in ("validation", "test"):
            with self.subTest(split=split):
                self.assert_invalid_training_row({"source_split": split})

    def test_trial_subset_is_repeatable_and_covers_every_class(self):
        rows = [{"class_id": c, "relative_path": f"{c}/{i}.jpg"}
                for c in range(3) for i in range(10)]
        a = trial_subset(rows, 2, 551)
        self.assertEqual(a, trial_subset(list(reversed(rows)), 2, 551))
        self.assertEqual(len(a), 6)
        self.assertEqual({r["class_id"] for r in a}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
