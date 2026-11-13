"""Test held-out input guards, known metric answers, and inference-only behavior.

Uses invented tiny manifests and tensors; no real test images or GPU required.
Run: python -m unittest discover -s tests -v
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from plant_disease.data import sha256_file
from plant_disease.evaluate import classification_metrics, load_test_manifests, predict
from plant_disease.model import state_fingerprint
from plant_disease.summarize_evaluation import check_predictions


class EvaluationMetricsTests(unittest.TestCase):
    def test_known_confusion_matrix_and_macro_f1(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["macro_f1"], (2/3 + .8)/2)
        self.assertEqual([r["support"] for r in metrics["per_class"]], [2, 2])

    def test_unpredicted_category_still_counts(self):
        metrics = classification_metrics([0, 1], [0, 0], 2)
        self.assertAlmostEqual(metrics["macro_f1"], 1/3)
        self.assertEqual(metrics["per_class"][1]["f1"], 0)

    def test_bad_labels_or_lengths_stop(self):
        for actual, predicted in (([0], [2]), ([0], []), ([], []), ([0], [0.5])):
            with self.assertRaises(ValueError):
                classification_metrics(actual, predicted, 2)

    def test_forward_pass_does_not_update_weights_or_batchnorm(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        before = state_fingerprint(model.state_dict())
        data = TensorDataset(torch.tensor([[1., 2.], [2., 1.], [0., 1.]]), torch.tensor([0, 1, 0]))
        result, actual, guesses, scores = predict(model, DataLoader(data, batch_size=2), torch.device("cpu"), 2)
        self.assertEqual(result["images"], 3)
        self.assertEqual(actual, [0, 1, 0])
        self.assertEqual(len(guesses), 3)
        self.assertTrue(all(0 <= x <= 1 for x in scores))
        self.assertEqual(before, state_fingerprint(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class EvaluationManifestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.categories = self.root / "categories.json"
        self.categories.write_text(json.dumps({"classes": [
            {"id": 0, "plantvillage": "A", "plantdoc": "Alias A"},
            {"id": 1, "plantvillage": "B", "plantdoc": "Alias B"}]}))
        self.rows = {}
        for split in ("train", "validation", "source_test", "target_test"):
            target = split == "target_test"
            self.rows[split] = []
            for label, name in enumerate(("A", "B")):
                category = f"Alias {name}" if target else name
                prefix = f"plantdoc/test/{category}" if target else f"plantvillage/raw/color/{category}"
                self.rows[split].append(dict(relative_path=f"{prefix}/{split}-{label}.jpg",
                    dataset="plantdoc" if target else "plantvillage", source_category=category,
                    source_split="test" if split.endswith("test") else "train", class_id=label,
                    class_name=name, sha256=f"hash-{split}-{label}", leaf_id="",
                    group_id=f"group-{split}-{label}", split=split))
        self.write()

    def write(self):
        hashes = {}
        for split, rows in self.rows.items():
            path = self.root / f"{split}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            hashes[path.name] = sha256_file(path)
        summary = {"categories_sha256": sha256_file(self.categories), "manifest_sha256": hashes,
                   "counts": {split: len(rows) for split, rows in self.rows.items()}}
        (self.root / "summary.json").write_text(json.dumps(summary))
        self.selection = {"categories_sha256": summary["categories_sha256"], "split_manifest_sha256": hashes}

    def test_correct_target_alias_maps_to_shared_output(self):
        rows, _, _ = load_test_manifests(self.root, self.categories, self.selection)
        self.assertEqual(rows["target_test"][0]["class_name"], "A")
        self.assertEqual(rows["target_test"][0]["source_category"], "Alias A")

    def test_changed_test_file_is_rejected(self):
        with (self.root / "target_test.csv").open("a") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(ValueError, "Changed test manifest"):
            load_test_manifests(self.root, self.categories, self.selection)

    def test_wrong_alias_training_rows_and_unsafe_paths_are_rejected(self):
        for key, bad in (("source_category", "wrong"), ("source_split", "train"),
                         ("relative_path", "../outside.jpg")):
            original = self.rows["target_test"][0][key]
            self.rows["target_test"][0][key] = bad
            self.write()
            with self.assertRaisesRegex(ValueError, "Test row"):
                load_test_manifests(self.root, self.categories, self.selection)
            self.rows["target_test"][0][key] = original

    def test_known_train_test_relation_is_rejected(self):
        self.rows["target_test"][0]["group_id"] = self.rows["train"][0]["group_id"]
        self.write()
        with self.assertRaisesRegex(ValueError, "Cross-split overlap"):
            load_test_manifests(self.root, self.categories, self.selection)


class SavedPredictionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        path = self.root / "predictions.csv"
        path.write_text("relative_path,true_class_id,predicted_class_id,confidence\n"
                        "a.jpg,0,0,0.8\nb.jpg,0,1,0.6\nc.jpg,1,1,0.9\nd.jpg,1,1,0.7\n")
        self.record = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
        self.record.update(predictions_file=path.name, predictions_sha256=sha256_file(path))

    def test_independent_recalculation_matches_known_labels(self):
        self.assertEqual(check_predictions(self.root, self.record, 2),
                         [("a.jpg", 0), ("b.jpg", 0), ("c.jpg", 1), ("d.jpg", 1)])

    def test_changed_predictions_are_rejected(self):
        (self.root / "predictions.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed after evaluation"):
            check_predictions(self.root, self.record, 2)

    def test_wrong_recorded_score_is_rejected(self):
        self.record["accuracy"] = 1.0
        with self.assertRaisesRegex(ValueError, "overall scores disagree"):
            check_predictions(self.root, self.record, 2)


if __name__ == "__main__":
    unittest.main()
