"""Check optional experiments using invented images and a tiny CPU model.

Run: python -m unittest discover -s tests -v
Temporary outputs are deleted automatically. These tests do not train the real
MobileNet, read our plant datasets, download weights, or produce research results.
"""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from plant_disease import train
from plant_disease.data import (PlantImages, augmentation_recipe, load_source_manifests,
                                make_augmentation, prepare_image, sha256_file,
                                training_data_record)
from plant_disease.evaluate import load_test_manifests
from plant_disease.summarize_training import (collect_summary, protect_report_destination,
                                            write_report)
from plant_disease.verify_training import verify
from scripts.build_splits import build_manifests, main as build_main, verify_manifests
from test_build_splits import CONFIG, fixture, image
from test_repetitions import examples
from plant_disease.summarize_repetitions import aggregate


class TinyModel(nn.Module):
    """Exercise the real training loop cheaply; this is not our project model."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=4), nn.BatchNorm2d(4),
                                      nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(4, 2)

    def forward(self, images):
        return self.classifier(self.features(images))


def tiny_model(class_count, weights, seed):
    torch.manual_seed(seed)
    return TinyModel()


def write_manifests(root, mixed=False):
    """Create two-category image lists with distinct bytes and group IDs."""
    root.mkdir(parents=True, exist_ok=True)
    categories = root / "categories.json"
    categories.write_text(json.dumps({"classes": [
        {"id": i, "plantvillage": f"PV{i}", "plantdoc": f"PD{i}"} for i in range(2)]}))
    rows = {}
    for split in ("train", "validation", "source_test", "target_test"):
        rows[split] = []
        for label in range(2):
            dataset = "plantdoc" if split == "target_test" or (mixed and label == 0 and split in ("train", "validation")) else "plantvillage"
            category = f"PD{label}" if dataset == "plantdoc" else f"PV{label}"
            source_split = "test" if split.endswith("test") else "train"
            prefix = f"plantdoc/{source_split}" if dataset == "plantdoc" else "plantvillage/raw/color"
            relative = f"{prefix}/{category}/{split}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Different sizes/colors keep all image hashes distinct.
            color = (40 + label*100, 30 + len(rows)*30, 80)
            Image.new("RGB", (31 + len(rows), 21), color).save(path)
            rows[split].append(dict(relative_path=relative, dataset=dataset, source_category=category,
                source_split=source_split, class_id=label, class_name=f"PV{label}",
                sha256=sha256_file(path), group_id=f"{split}-{label}", leaf_id="", split=split))
    save_manifests(root, rows, categories, mixed)
    return rows, categories


def save_manifests(root, rows, categories, mixed):
    hashes = {}
    for split, records in rows.items():
        path = root / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        hashes[path.name] = sha256_file(path)
    (root / "summary.json").write_text(json.dumps({"categories_sha256": sha256_file(categories),
        "include_plantdoc_train": mixed, "counts": {s: len(v) for s, v in rows.items()},
        "manifest_sha256": hashes, "classes": 2}))


class AugmentationTests(unittest.TestCase):
    def test_none_preserves_original_pixels_and_does_not_use_randomness(self):
        self.assertIsNone(make_augmentation())
        self.assertEqual(augmentation_recipe("none"), "none")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, _ = write_manifests(root)
            dataset = PlantImages(rows["train"], root, transform=make_augmentation("none"))
            with Image.open(root / rows["train"][0]["relative_path"]) as original:
                expected = prepare_image(original)
            torch.manual_seed(1)
            self.assertTrue(torch.equal(dataset[0][0], expected))
            torch.manual_seed(100)
            self.assertTrue(torch.equal(dataset[0][0], expected))

    def test_recipe_matches_transform_and_handles_grayscale_and_alpha(self):
        transform = make_augmentation("basic-v1")
        recipe = augmentation_recipe("basic-v1")
        crop, flip, rotation, jitter = transform.transforms
        self.assertEqual(list(crop.scale), recipe["random_resized_crop"]["scale"])
        self.assertEqual(list(crop.ratio), recipe["random_resized_crop"]["ratio"])
        self.assertEqual(flip.p, recipe["horizontal_flip"]["p"])
        self.assertEqual(rotation.degrees, recipe["rotation"]["degrees"])
        self.assertEqual(list(rotation.fill), recipe["rotation"]["fill"])
        self.assertEqual(jitter.hue, (-.05, .05))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("L", "RGBA"):
                path = root / f"{mode}.png"
                Image.new(mode, (24, 32)).save(path)
                rows = [dict(relative_path=path.name, sha256=sha256_file(path), class_id=0)]
                tensor, _ = PlantImages(rows, root, transform=transform)[0]
                self.assertEqual(tuple(tensor.shape), (3, 224, 224))
                self.assertTrue(torch.isfinite(tensor).all())
        with self.assertRaises(ValueError):
            make_augmentation("typo")

    def test_seeded_augmentation_is_repeatable_with_and_without_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, _ = write_manifests(root)
            dataset = PlantImages(rows["train"], root, make_augmentation("basic-v1"))
            for workers in (0, 2):
                def sample(seed):
                    torch.manual_seed(seed)
                    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=workers,
                                        generator=torch.Generator().manual_seed(seed))
                    return next(iter(loader))[0]
                self.assertTrue(torch.equal(sample(551), sample(551)))
                self.assertFalse(torch.equal(sample(551), sample(552)))


class ExperimentGuardTests(unittest.TestCase):
    def test_mixed_data_requires_opt_in_and_never_accepts_test_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, categories = write_manifests(root, mixed=True)
            with self.assertRaises(ValueError):
                load_source_manifests(root, categories)
            loaded, _, summary = load_source_manifests(root, categories, True)
            selection = dict(categories_sha256=sha256_file(categories),
                split_manifest_sha256=summary["manifest_sha256"],
                training_data=training_data_record(loaded, True))
            tests, _, _ = load_test_manifests(root, categories, selection)
            self.assertEqual(set(tests), {"source_test", "target_test"})
            for bad in ("test", "validation"):
                rows["train"][0]["source_split"] = bad
                save_manifests(root, rows, categories, True)
                with self.assertRaisesRegex(ValueError, "mapped training"):
                    load_source_manifests(root, categories, True)

    def test_unknown_dataset_and_wrong_alias_raise_validation_errors(self):
        for change in ({"dataset": "unknown"}, {"source_category": "wrong"},
                       {"relative_path": "plantvillage/raw/color/wrong/image.png"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows, categories = write_manifests(root)
                rows["train"][0].update(change)
                save_manifests(root, rows, categories, False)
                with self.assertRaises(ValueError):
                    load_source_manifests(root, categories)

    def test_optional_pool_preserves_holdouts_and_rejects_related_training_images(self):
        inventory = fixture()
        original, _ = build_manifests(inventory, CONFIG)
        mixed, excluded = build_manifests(inventory, CONFIG, include_plantdoc_train=True)
        for split in ("source_test", "target_test"):
            self.assertEqual(mixed[split], original[split])
        self.assertTrue(any(r["dataset"] == "plantdoc" for s in ("train", "validation") for r in mixed[s]))
        self.assertFalse(excluded)
        inventory.append(image("target_test_copy", dataset="plantdoc", digest=inventory[11]["sha256"]))
        mixed, excluded = build_manifests(inventory, CONFIG, include_plantdoc_train=True)
        self.assertIn("related_to_reserved_test_group", [r["reason"] for r in excluded])
        mixed["source_test"][0]["dataset"] = "plantdoc"
        with self.assertRaisesRegex(ValueError, "Wrong data source"):
            verify_manifests(mixed, excluded, inventory, 1, include_plantdoc_train=True)

    def test_split_cli_refuses_overwrite_and_original_folder_for_mixed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "train.csv"
            marker.write_text("preserve me")
            for args in (["--output-dir", str(root)], ["--include-plantdoc-train"]):
                with patch("sys.argv", ["build_splits", *args]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        build_main()
            self.assertEqual(marker.read_text(), "preserve me")

    def test_reports_refuse_overwrites_and_followup_in_original_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "report.json").write_text("preserve")
            with self.assertRaisesRegex(ValueError, "already exist"):
                protect_report_destination(root, ["report.json"])
        repo = Path(__file__).resolve().parents[1]
        for folder in ("docs/training-runs", "docs/evaluation"):
            with self.assertRaisesRegex(ValueError, "separate"):
                protect_report_destination(repo / folder, ["new.json"], followup=True)

    def test_repetition_summary_rejects_different_policies_and_recipes(self):
        for key, value in (("training_data", {"include_plantdoc_train": True}),
                           ("preprocessing", {"augmentation": augmentation_recipe("basic-v1")})):
            runs = examples()
            runs[1][key] = value
            with self.assertRaises(ValueError):
                aggregate(runs)


class TrainingIntegrationTests(unittest.TestCase):
    def test_cli_records_verifies_and_reports_each_option_combination(self):
        torch.set_num_threads(2)
        for mixed in (False, True):
            for augmentation in ("none", "basic-v1"):
                with self.subTest(mixed=mixed, augmentation=augmentation), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    data = root / "data"
                    _, categories = write_manifests(data, mixed=mixed)
                    weights = root / "weights.pth"
                    weights.write_bytes(b"synthetic fixture, not a model")
                    output = root / "run"
                    arguments = ["train", "--data-root", str(data), "--weights", str(weights),
                        "--splits-dir", str(data), "--categories", str(categories), "--output-dir", str(output),
                        "--epochs", "2", "--batch-size", "2", "--workers", "0", "--device", "cpu"]
                    if augmentation != "none":
                        arguments += ["--augmentation", augmentation]
                    if mixed:
                        arguments += ["--include-plantdoc-train"]
                    with patch("sys.argv", arguments), patch.object(train, "WEIGHTS_SHA256", sha256_file(weights)), \
                         patch.object(train, "build_model", tiny_model), contextlib.redirect_stdout(io.StringIO()):
                        train.main()
                    metadata = json.loads((output / "run.json").read_text())
                    self.assertEqual(metadata["preprocessing"]["augmentation"], augmentation_recipe(augmentation))
                    self.assertEqual(metadata["training_data"]["include_plantdoc_train"], mixed)
                    with patch("plant_disease.verify_training.build_model", tiny_model), contextlib.redirect_stdout(io.StringIO()):
                        verify(output, data, data, categories)
                    summary = collect_summary(output)
                    write_report(summary, root / "reports", "fixture")
                    report = (root / "reports/fixture.md").read_text()
                    self.assertIn(f"Training augmentation: `{augmentation}`", report)
                    if mixed:
                        self.assertIn("not a source-only transfer experiment", report)
                        self.assertNotIn("No PlantDoc training images were used", report)
                    else:
                        self.assertIn("No PlantDoc training images were used", report)
                    for metric in ("loss", "accuracy", "macro_f1"):
                        self.assertEqual(summary["models"]["frozen"]["history"][0]["train"][metric],
                                         summary["models"]["finetune"]["history"][0]["train"][metric])
                    # A false "none" entry must not be accepted for an augmented run.
                    if augmentation != "none":
                        metadata["preprocessing"]["augmentation"] = "none"
                        (output / "run.json").write_text(json.dumps(metadata))
                        with self.assertRaisesRegex(ValueError, "augmentation differs"):
                            collect_summary(output)

    def test_mixed_augmented_runs_reach_selection_and_evaluation_reports(self):
        from plant_disease import evaluate
        from plant_disease.summarize_repetitions import collect, write_report as write_repetitions
        from plant_disease.summarize_evaluation import collect_evaluation, write_report as write_evaluation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            _, categories = write_manifests(data, mixed=True)
            weights = root / "weights.pth"
            weights.write_bytes(b"synthetic fixture")
            for seed in (551, 552, 553):
                output = root / "runs" / f"source-seed-{seed}"
                argv = ["train", "--data-root", str(data), "--weights", str(weights),
                    "--splits-dir", str(data), "--categories", str(categories), "--output-dir", str(output),
                    "--epochs", "2", "--batch-size", "2", "--workers", "0", "--device", "cpu",
                    "--seed", str(seed), "--augmentation", "basic-v1", "--include-plantdoc-train"]
                with patch("sys.argv", argv), patch.object(train, "WEIGHTS_SHA256", sha256_file(weights)), \
                     patch.object(train, "build_model", tiny_model), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                with patch("plant_disease.verify_training.build_model", tiny_model), contextlib.redirect_stdout(io.StringIO()):
                    verify(output, data, data, categories)
            combined, selection = collect(root / "runs")
            self.assertTrue(selection["training_data"]["include_plantdoc_train"])
            self.assertEqual(selection["augmentation"]["name"], "basic-v1")
            write_repetitions(combined, selection, root / "reports")
            repeated = (root / "reports/repeated-training.md").read_text()
            self.assertIn("not a source-only transfer experiment", repeated)
            argv = ["evaluate", "--selection", str(root / "reports/selected-checkpoints.json"),
                "--runs-root", str(root / "runs"), "--splits-dir", str(data),
                "--categories", str(categories), "--data-root", str(data),
                "--output-dir", str(root / "evaluation"), "--device", "cpu", "--workers", "0"]
            with patch("sys.argv", argv), patch.object(evaluate, "build_model", tiny_model), \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
            evaluated = collect_evaluation(root / "evaluation")
            write_evaluation(evaluated, root / "evaluation-report")
            report = (root / "evaluation-report/final-results.md").read_text()
            self.assertIn("not a source-only transfer experiment", report)
            self.assertIn("Training augmentation: `basic-v1`", report)
            self.assertNotIn("No model was retrained", report)
            self.assertEqual(len(evaluated["results"]), 12)


if __name__ == "__main__":
    unittest.main()
