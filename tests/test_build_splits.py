"""Exercise split safety on invented metadata, without touching plant images.

Run from the repository: python -m unittest discover -s tests -v
"""

import copy
import hashlib
import unittest

from scripts.build_splits import build_manifests, related_groups


CONFIG = {"classes": [{"id": 0, "plantvillage": "source_class", "plantdoc": "target_class"}]}


def image(tag, dataset="plantvillage", split="train", category=None, leaf_id="", digest=None):
    return {"relative_path": f"{dataset}/{split}/{tag}.jpg", "dataset": dataset,
            "split": split, "category": category or ("source_class" if dataset == "plantvillage" else "target_class"),
            "leaf_id": leaf_id, "sha256": digest or hashlib.sha256(tag.encode()).hexdigest(),
            "readable": "True"}


def fixture():
    return [image(f"train{i}") for i in range(10)] + [
        image("source_holdout", split="test"),
        image("target_holdout", dataset="plantdoc", split="test"),
        image("unused_target", dataset="plantdoc"),
    ]


class SplitTests(unittest.TestCase):
    def test_repeatable_and_input_order_independent(self):
        rows = fixture()
        before = copy.deepcopy(rows)
        a = build_manifests(rows, CONFIG)
        b = build_manifests(list(reversed(rows)), CONFIG)
        self.assertEqual(a, b)
        self.assertEqual(rows, before)
        self.assertEqual(len(a[0]["validation"]), 2)
        self.assertEqual([r["reason"] for r in a[1]], ["target_training_not_used"])

    def test_source_test_kept_and_its_training_copy_excluded(self):
        rows = fixture()
        rows[0]["sha256"] = rows[10]["sha256"]
        manifests, excluded = build_manifests(rows, CONFIG)
        self.assertEqual(manifests["source_test"][0]["relative_path"], rows[10]["relative_path"])
        self.assertIn((rows[0]["relative_path"], "related_to_reserved_test_group"),
                      [(r["relative_path"], r["reason"]) for r in excluded])

    def test_leaf_groups_do_not_cross_training_validation(self):
        rows = fixture()
        rows[0]["leaf_id"] = rows[1]["leaf_id"] = "same-leaf"
        manifests, _ = build_manifests(rows, CONFIG)
        positions = {r["relative_path"]: split for split, records in manifests.items() for r in records}
        self.assertEqual(positions[rows[0]["relative_path"]], positions[rows[1]["relative_path"]])

    def test_transitive_hash_and_leaf_evidence_forms_one_group(self):
        rows = [image("a"), image("b", leaf_id="leaf"), image("c", leaf_id="leaf")]
        rows[0]["sha256"] = rows[1]["sha256"]
        self.assertEqual(len(related_groups(rows)), 1)

    def test_conflicting_unused_label_excludes_affected_test_image(self):
        rows = fixture()
        rows += [image("second_target", dataset="plantdoc", split="test")]
        rows += [image("bad_label", dataset="plantdoc", category="excluded_other_class", digest=rows[11]["sha256"])]
        manifests, excluded = build_manifests(rows, CONFIG)
        self.assertEqual(len(manifests["target_test"]), 1)
        conflict_paths = {r["relative_path"] for r in excluded if r["reason"] == "conflicting_labels_in_related_group"}
        self.assertIn(rows[11]["relative_path"], conflict_paths)
        self.assertIn(rows[-1]["relative_path"], conflict_paths)

    def test_deduplicate_within_test_and_keep_category_coverage(self):
        rows = fixture()
        rows.append(image("test_copy", split="test", digest=rows[10]["sha256"]))
        manifests, excluded = build_manifests(rows, CONFIG)
        self.assertEqual(len(manifests["source_test"]), 1)
        self.assertEqual(sum(r["reason"] == "byte_identical_copy" for r in excluded), 1)

    def test_cross_dataset_test_overlap_excludes_source_copy(self):
        rows = fixture()
        rows.append(image("extra_source_test", split="test"))
        rows[10]["sha256"] = rows[11]["sha256"]
        manifests, excluded = build_manifests(rows, CONFIG)
        self.assertEqual(len(manifests["source_test"]), 1)
        self.assertEqual(len(manifests["target_test"]), 1)
        self.assertIn(rows[10]["relative_path"], {r["relative_path"] for r in excluded})

    def test_missing_test_coverage_stops_instead_of_dropping_class(self):
        rows = [r for r in fixture() if r["dataset"] != "plantdoc" or r["split"] != "test"]
        with self.assertRaisesRegex(ValueError, "target_test"):
            build_manifests(rows, CONFIG)

    def test_bad_inventory_stops(self):
        for change in ({"sha256": ""}, {"relative_path": "../outside.jpg"}, {"split": "validation"}):
            rows = fixture()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_manifests(rows, CONFIG)


if __name__ == "__main__":
    unittest.main()
