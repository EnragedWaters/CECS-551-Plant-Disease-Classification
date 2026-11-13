"""Check duplicate guidance without downloading, changing, or training on images.

Run from the project root: python -m unittest discover -s tests -v
Small made-up records exercise the cases that previously hid test-label risks.
"""

import unittest

from scripts.analyze_data import classify_duplicate_groups


def record(dataset, split, category):
    """Create just the image metadata needed by the recommendation function."""
    return {
        "dataset": dataset,
        "split": split,
        "category": category,
        "relative_path": f"{dataset}/{split}/{category}/example.jpg",
    }


class DuplicateRecommendationTests(unittest.TestCase):
    def test_unused_copy_still_requires_review_of_conflicting_test_label(self):
        groups, _ = classify_duplicate_groups({"same-bytes": [
            record("plantdoc", "train", "Potato leaf early blight"),
            record("plantdoc", "test", "Potato leaf late blight"),
        ]})
        group = groups[0]
        self.assertEqual(group["experiment_impact"], "no_direct_impact")
        self.assertEqual(group["evaluation_copies"], 1)
        self.assertTrue(group["label_conflict"])
        self.assertIn("Review conflicting original labels", group["recommended_action"])
        self.assertIn("including test images", group["recommended_action"])

    def test_matching_labels_do_not_trigger_a_label_warning(self):
        groups, _ = classify_duplicate_groups({"same-bytes": [
            record("plantdoc", "train", "Apple leaf"),
            record("plantdoc", "test", "Apple leaf"),
        ]})
        self.assertFalse(groups[0]["label_conflict"])
        self.assertNotIn("Review conflicting", groups[0]["recommended_action"])

    def test_training_overlap_keeps_its_exclusion_and_label_review(self):
        groups, _ = classify_duplicate_groups({"same-bytes": [
            record("plantvillage", "train", "Tomato___healthy"),
            record("plantvillage", "test", "Tomato___Late_blight"),
        ]})
        group = groups[0]
        self.assertEqual(group["experiment_impact"], "training_test_overlap")
        self.assertIn("Exclude image from training", group["recommended_action"])
        self.assertIn("Review conflicting original labels", group["recommended_action"])


if __name__ == "__main__":
    unittest.main()
