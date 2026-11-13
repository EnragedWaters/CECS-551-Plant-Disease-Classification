"""Guard against averaging duplicate seeds, changed data, or unlike experiments.

These records are invented test inputs, not research results. Run with:
    python -m unittest discover -s tests -v
"""

import unittest

from plant_disease.summarize_repetitions import CORE_FILES, SEEDS, aggregate


def examples():
    runs = []
    for seed, accuracy in zip(SEEDS, (.8, .9, 1.0)):
        metric = {"images": 10, "loss": .5, "accuracy": .5, "macro_f1": .5}
        run = {"settings": {"seed": seed, "warmup_epochs": 1, "head_lr": .001},
               "counts": {"train": 100, "validation": 10}, "software": {}, "device": "cpu",
               "preprocessing": {}, "weights_sha256": "same", "selected_manifest_sha256": {"train": "same"},
               "categories_sha256": "same", "checkpoint_rule": "loss", "precision": "float32",
               "deterministic_algorithms": True, "split_summary": {},
               "code_sha256": {name: "same" for name in CORE_FILES}, "models": {}}
        for mode in ("frozen", "finetune"):
            run["models"][mode] = {"batch_norm_statistics_unchanged": True,
                "features_changed": mode == "finetune", "validation": {
                    "accuracy": accuracy, "macro_f1": accuracy, "loss": 1 - accuracy},
                "history": [{"train": dict(metric), "validation": dict(metric)}]}
        runs.append(run)
    return runs


class RepetitionTests(unittest.TestCase):
    def test_reports_all_seeds_and_sample_not_population_spread(self):
        summary = aggregate(list(reversed(examples())))
        stats = summary["statistics"]["frozen"]["accuracy"]
        self.assertEqual(stats["values"], [.8, .9, 1.0])
        self.assertAlmostEqual(stats["mean"], .9)
        self.assertAlmostEqual(stats["sample_standard_deviation"], .1)

    def test_rejects_duplicate_or_missing_seed(self):
        runs = examples()
        for bad in (runs[:2], [runs[0], runs[0], runs[2]]):
            with self.assertRaisesRegex(ValueError, "each declared seed"):
                aggregate(bad)

    def test_rejects_changed_data_settings_or_training_code(self):
        for key in ("selected_manifest_sha256", "settings", "code_sha256"):
            runs = examples()
            if key == "settings":
                runs[1][key]["head_lr"] = .01
            elif key == "code_sha256":
                runs[1][key]["train.py"] = "different"
            else:
                runs[1][key]["train"] = "different"
            with self.assertRaises(ValueError):
                aggregate(runs)

    def test_reporting_helper_changes_do_not_change_the_experiment(self):
        runs = examples()
        runs[2]["code_sha256"]["summarize_repetitions.py"] = "added later"
        aggregate(runs)

    def test_rejects_unmatched_warmup(self):
        runs = examples()
        runs[1]["models"]["finetune"]["history"][0]["train"]["loss"] = 2
        with self.assertRaisesRegex(ValueError, "warmup"):
            aggregate(runs)

    def test_rejects_frozen_feature_changes(self):
        runs = examples()
        runs[0]["models"]["frozen"]["features_changed"] = True
        with self.assertRaisesRegex(ValueError, "weight changes"):
            aggregate(runs)


if __name__ == "__main__":
    unittest.main()
