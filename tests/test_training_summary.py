"""Check that exported scores describe the selected model, not the last epoch.

Uses tiny invented records and bytes in a temporary folder; no GPU or images.
Run with: python -m unittest discover -s tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from plant_disease.data import sha256_file
from plant_disease.summarize_training import collect_summary


class TrainingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metadata = {
            "scope": "source_training_and_validation",
            "arguments": dict(epochs=3, warmup_epochs=1, batch_size=2, workers=0,
                              seed=551, head_lr=.001, backbone_lr=.0001,
                              weight_decay=.0001, patience=3, device="cpu"),
            "counts": {"train": 4, "validation": 2}, "software": {}, "device": "cpu",
            "preprocessing": {}, "weights_sha256": "fixture", "code_sha256": {},
            "split_summary": {"categories_sha256": "fixture"},
            "checkpoint_rule": "minimum source validation loss after warmup",
            "selected_manifest_sha256": {},
        }
        for split in ("train", "validation"):
            path = self.root / f"selected_{split}.csv"
            path.write_text("fixture\n")
            metadata["selected_manifest_sha256"][split] = sha256_file(path)
        (self.root / "run.json").write_text(json.dumps(metadata))
        history = [dict(epoch=epoch, train={"seconds": 1},
                        validation={"loss": loss, "accuracy": accuracy, "seconds": 1},
                        peak_allocated_mib=None, peak_reserved_mib=None)
                   for epoch, loss, accuracy in ((1, .1, .99), (2, .2, .9), (3, .3, .8))]
        self.results = {}
        for mode in ("frozen", "finetune"):
            (self.root / mode).mkdir()
            (self.root / mode / "best.pt").write_bytes(b"fixture checkpoint")
            self.results[mode] = dict(history=history, best_epoch=2, epochs_completed=3,
                trial=False, initial_feature_sha256="same", initial_classifier_sha256="same",
                final_feature_sha256="same" if mode == "frozen" else "changed",
                batch_norm_statistics_unchanged=True)
        self.save_results()

    def save_results(self):
        (self.root / "comparison.json").write_text(json.dumps(self.results))

    def test_uses_best_post_warmup_epoch_not_last_or_warmup(self):
        summary = collect_summary(self.root)
        self.assertEqual(summary["models"]["frozen"]["validation"]["accuracy"], .9)
        self.assertEqual(summary["models"]["frozen"]["best_epoch"], 2)
        self.assertIsNone(summary["models"]["frozen"]["peak_allocated_mib"])

    def test_rejects_incomplete_pair(self):
        del self.results["finetune"]
        self.save_results()
        with self.assertRaisesRegex(ValueError, "Both model versions"):
            collect_summary(self.root)

    def test_rejects_changed_image_list(self):
        (self.root / "selected_train.csv").write_text("different\n")
        with self.assertRaisesRegex(ValueError, "image list changed"):
            collect_summary(self.root)


if __name__ == "__main__":
    unittest.main()
