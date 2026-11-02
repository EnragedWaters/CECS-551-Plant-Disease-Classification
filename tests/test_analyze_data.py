"""Small synthetic fixtures test the audit without downloading project datasets."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_data import analyze, image_key, leaf_identifier, read_splits


class DataAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "datasets"
        self.output = self.base / "out"
        pv = self.data / "plantvillage"
        color = pv / "raw" / "color" / "Apple___healthy"
        color.mkdir(parents=True)
        Image.new("RGB", (21, 17), "green").save(color / "a.jpg")
        shutil.copyfile(color / "a.jpg", color / "b.jpg")
        (color / "broken.jpg").write_bytes(b"not a JPEG")
        splits = pv / "splits"
        splits.mkdir()
        (splits / "color_train.txt").write_text("raw/color/Apple___healthy/a.jpg\nraw/color/Apple___healthy/broken.jpg\n")
        (splits / "color_test.txt").write_text("raw/color/Apple___healthy/b.jpg\n")
        leaves = pv / "leaf_grouping"
        leaves.mkdir()
        (leaves / "leaf-map.json").write_text(json.dumps({"a": ["Apple___healthy_leaf1"],
                                                          "b": ["Apple___healthy_leaf1"]}))
        for split in ("train", "test"):
            pd = self.data / "plantdoc" / split / "Apple leaf"
            pd.mkdir(parents=True)
            Image.new("RGB", (31, 23), "orange" if split == "train" else "blue").save(pd / f"{split}.png")
        (self.data / "download-manifest.json").write_text(json.dumps({"plantvillage_revision": "sample-revision",
                                                                        "plantdoc_revision": "another-revision"}))

    def test_source_keys_and_leaf_groups(self):
        self.assertEqual(image_key("raw/color/Apple___healthy/a.jpg"), "Apple___healthy/a.jpg")
        self.assertEqual(read_splits(self.data / "plantvillage")["Apple___healthy/b.jpg"], "test")
        self.assertEqual(leaf_identifier("a.jpg", "Apple___healthy", {"a": ["leaf1"]}), "leaf1")
        self.assertEqual(leaf_identifier("a.jpg", "Apple___healthy", {"a": ["X", "Apple___healthy_1"]}),
                         "Apple___healthy_1")
        self.assertEqual(leaf_identifier("missing.jpg", "Apple___healthy", {}), "")

    def test_audit_flags_counts_and_reproducible_samples(self):
        result = analyze(self.data, self.output, samples_per_category=2, seed=13)
        self.assertEqual(result["images_by_dataset"], {"plantdoc": 2, "plantvillage": 3})
        self.assertEqual(result["unreadable_images"], 1)
        self.assertEqual(result["exact_duplicate_hash_groups"], 1)
        self.assertEqual(result["exact_duplicate_cross_partition_groups"], 1)
        self.assertEqual(result["same_leaf_cross_split_groups"], 1)
        self.assertEqual(result["missing_source_images"], 0)
        self.assertEqual(result["plantvillage_revision"], "sample-revision")
        self.assertEqual(result["plantdoc_revision"], "another-revision")
        with (self.output / "quality_flags.csv").open(newline="") as stream:
            flags = list(csv.DictReader(stream))
        self.assertEqual(sum(f["flag"] == "exact_duplicate" for f in flags), 2)
        self.assertEqual(sum(f["flag"] == "unreadable" for f in flags), 1)
        self.assertEqual(sum(f["flag"] == "same_leaf_across_splits" for f in flags), 2)
        with (self.output / "class_counts.csv").open(newline="") as stream:
            counts = list(csv.DictReader(stream))
        self.assertEqual(sum(int(c["file_count"]) for c in counts), 5)
        self.assertEqual({c["dataset"] for c in counts}, {"plantvillage", "plantdoc"})
        with (self.output / "sample_manifest.csv").open(newline="") as stream:
            samples = list(csv.DictReader(stream))
        self.assertEqual(len(samples), 4)
        self.assertEqual({s["dataset"] for s in samples}, {"plantvillage", "plantdoc"})
        first_paths = [r["relative_path"] for r in samples]
        analyze(self.data, self.output, samples_per_category=2, seed=13)
        with (self.output / "sample_manifest.csv").open(newline="") as stream:
            self.assertEqual(first_paths, [r["relative_path"] for r in csv.DictReader(stream)])

    def test_missing_source_file_flagged(self):
        path = self.data / "plantvillage" / "splits" / "color_test.txt"
        path.write_text(path.read_text() + "raw/color/Apple___healthy/missing.jpg\n")
        result = analyze(self.data, self.output)
        self.assertEqual(result["missing_source_images"], 1)

    def test_conflicting_split_entries_rejected(self):
        path = self.data / "plantvillage" / "splits" / "color_test.txt"
        path.write_text(path.read_text() + "raw/color/Apple___healthy/a.jpg\n")
        with self.assertRaisesRegex(ValueError, "Repeated or conflicting"):
            analyze(self.data, self.output)


if __name__ == "__main__":
    unittest.main()
