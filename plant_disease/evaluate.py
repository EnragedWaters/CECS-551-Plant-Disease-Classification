"""Evaluate the six preselected models on the two reserved test sets.

Run: python -m plant_disease.evaluate --output-dir outputs/evaluation/final-01
Uses the recorded category mapping and the exact training image preparation.
No fitting, tuning, relabeling, or seed selection occurs. Checksums protect the
model files and image lists; raw data and checkpoints are never changed.
Predictions and measurements go into a new, Git-ignored output folder.
"""

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# Set deterministic CUDA configuration before any GPU operation.
from .train import synchronize, save_json
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .data import PlantImages, load_source_manifests, sha256_file
from .model import build_model, state_fingerprint
from .summarize_repetitions import CORE_FILES, collect


TESTS = {"source_test": "plantvillage", "target_test": "plantdoc"}


def load_test_manifests(directory, categories_path, selection):
    """Check the reserved lists and known overlaps before reading test pixels."""
    source, categories, summary = load_source_manifests(
        directory, categories_path, (selection.get("training_data") or {}).get("include_plantdoc_train", False))
    if (summary["categories_sha256"] != selection["categories_sha256"]
            or summary["manifest_sha256"] != selection["split_manifest_sha256"]):
        raise ValueError("Category or split fingerprints changed after model selection")
    classes = categories["classes"]
    all_rows = dict(source)
    for split, dataset in TESTS.items():
        path = directory / f"{split}.csv"
        if sha256_file(path) != selection["split_manifest_sha256"][path.name]:
            raise ValueError(f"Changed test manifest: {split}")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != summary["counts"][split]:
            raise ValueError(f"Wrong test count: {split}")
        for row in rows:
            relative = PurePosixPath(row["relative_path"])
            label = int(row["class_id"])
            if not 0 <= label < len(classes):
                raise ValueError("Unknown test category ID")
            category = classes[label][dataset]
            parent = (("plantvillage", "raw", "color", category) if dataset == "plantvillage"
                      else ("plantdoc", "test", category))
            if (relative.is_absolute() or ".." in relative.parts or "\\" in str(relative)
                    or relative.parts[:-1] != parent or row["dataset"] != dataset
                    or row["source_split"] != "test" or row["split"] != split
                    or row["source_category"] != category
                    or row["class_name"] != classes[label]["plantvillage"]):
                raise ValueError("Test row does not match the fixed category/dataset/split")
        if {int(r["class_id"]) for r in rows} != set(range(len(classes))):
            raise ValueError(f"Missing test category: {split}")
        all_rows[split] = rows
    # Related but nonidentical photos may share a group WITHIN a split. No known
    # path, byte hash, group, or resolved leaf ID may cross these four splits.
    seen = {key: set() for key in ("relative_path", "sha256", "group_id", "leaf_id")}
    for split, rows in all_rows.items():
        for key in seen:
            values = {r[key] for r in rows if r[key]}
            if values & seen[key]:
                raise ValueError(f"Cross-split overlap: {split}, {key}")
            if key in ("relative_path", "sha256") and len(values) != len(rows):
                raise ValueError(f"Repeated or missing {key} inside {split}")
            seen[key].update(values)
    return {split: all_rows[split] for split in TESTS}, categories, summary


def classification_metrics(actual, predicted, class_count):
    """Keep every category, including categories with no correct predictions."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Need equally sized, nonempty label lists")
    if any(not 0 <= int(label) < class_count or int(label) != label
           for label in np.concatenate((actual, predicted))):
        raise ValueError("Prediction or truth outside the category order")
    labels = list(range(class_count))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    return {"images": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "per_class": [{"id": i, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i in labels]}


def predict(model, loader, device, class_count):
    """Only forward passes: eval mode disables state updates, gradients stay off."""
    model.eval()
    actual, predicted, confidence = [], [], []
    total_loss = 0.0
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            if logits.shape != (len(labels), class_count) or not torch.isfinite(logits).all():
                raise ValueError("Invalid model output")
            labels_gpu = labels.to(device, non_blocking=True)
            total_loss += torch.nn.functional.cross_entropy(logits, labels_gpu, reduction="sum").item()
            probabilities = logits.softmax(1)
            scores, guesses = probabilities.max(1)
            actual.extend(labels.tolist())
            predicted.extend(guesses.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
    synchronize(device)
    seconds = time.perf_counter() - start
    result = classification_metrics(actual, predicted, class_count)
    result.update(loss=total_loss / len(actual), seconds=seconds)
    return result, actual, predicted, confidence


def write_predictions(path, rows, actual, predicted, confidence):
    if len(rows) != len(actual):
        raise ValueError("Prediction count differs from the fixed image list")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_path", "true_class_id", "predicted_class_id", "confidence", "correct"))
        for row, truth, guess, score in zip(rows, actual, predicted, confidence, strict=True):
            if int(row["class_id"]) != truth:
                raise ValueError("Prediction order differs from the image list")
            writer.writerow((row["relative_path"], truth, guess, score, truth == guess))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("docs/training-runs/selected-checkpoints.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--categories", type=Path, default=Path("configs/categories.json"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "datasets/cecs551-plant-disease")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output folder already exists; do not overwrite a test record")
    if args.workers < 0 or (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("Invalid workers or CUDA unavailable")
    args.data_root = args.data_root.expanduser()
    selection = json.loads(args.selection.read_text())
    # Reuse all training/verification checks and match the committed selection.
    # A later checkpoint or a favorable seed cannot silently replace this list.
    combined, current_selection = collect(args.runs_root)
    if selection != current_selection:
        raise ValueError("Saved checkpoints no longer match the pre-test selection record")
    reference = combined["runs"][0]
    for name in CORE_FILES:
        if sha256_file(Path(__file__).parent / name) != reference["code_sha256"][name]:
            raise ValueError(f"Core training/preprocessing code changed: {name}")
    rows, categories, split_summary = load_test_manifests(args.splits_dir, args.categories, selection)
    if split_summary != reference["split_summary"]:
        raise ValueError("Split metadata differs from the trained experiment")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    batch_size = reference["settings"]["batch_size"]
    loaders = {split: DataLoader(PlantImages(records, args.data_root), batch_size=batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
               for split, records in rows.items()}
    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir / "selection.json", selection)
    metadata = {"scope": "evaluation on reserved source and target lists; no fitting during evaluation",
                "training_data": reference.get("training_data"),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "selection_sha256": sha256_file(args.selection), "selection": selection,
                "categories": categories["classes"], "split_summary": split_summary,
                "preprocessing": reference["preprocessing"], "test_augmentation": "none",
                "training_settings": {k: v for k, v in reference["settings"].items() if k != "seed"},
                "training_seeds": selection["seeds"],
                "evaluation": {"batch_size": batch_size, "workers": args.workers, "precision": "float32",
                               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                               "deterministic_algorithms": True},
                "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                             "torchvision": str(torchvision.__version__)},
                "code_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    save_json(args.output_dir / "run.json", metadata)
    results = []
    for chosen in selection["checkpoints"]:
        checkpoint_path = args.runs_root / chosen["run_name"] / chosen["checkpoint"]
        if sha256_file(checkpoint_path) != chosen["sha256"]:
            raise ValueError("Selected checkpoint changed during evaluation")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["trial"] or checkpoint["epoch"] != chosen["epoch"]
                or checkpoint["mode"] != chosen["mode"]
                or checkpoint["metadata"]["arguments"]["seed"] != chosen["seed"]
                or checkpoint["class_count"] != len(categories["classes"])
                or checkpoint["metadata"]["categories"] != categories["classes"]):
            raise ValueError("Checkpoint metadata disagrees with the selected experiment")
        model = build_model(len(categories["classes"]),
                            args.data_root / "weights/mobilenet_v3_small-047dcff4.pth", chosen["seed"])
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        before = state_fingerprint(model.state_dict())
        for split, loader in loaders.items():
            metrics, actual, guesses, confidence = predict(model, loader, device, len(categories["classes"]))
            name = f"seed-{chosen['seed']}-{chosen['mode']}-{split}"
            path = args.output_dir / f"{name}-predictions.csv"
            write_predictions(path, rows[split], actual, guesses, confidence)
            record = {"seed": chosen["seed"], "mode": chosen["mode"], "epoch": chosen["epoch"],
                      "split": split, "checkpoint_sha256": chosen["sha256"],
                      "predictions_file": path.name, "predictions_sha256": sha256_file(path), **metrics}
            save_json(args.output_dir / f"{name}.json", record)
            results.append(record)
            print(f"Seed {chosen['seed']} {chosen['mode']} {split}: "
                  f"accuracy={metrics['accuracy']:.2%}, macro-F1={metrics['macro_f1']:.4f}", flush=True)
        if state_fingerprint(model.state_dict()) != before:
            raise RuntimeError("Evaluation unexpectedly changed model weights or buffers")
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(args.output_dir / "results.json", {"metadata": metadata, "results": results,
                                                "model_states_unchanged": True})
    print(f"Completed final evaluation: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
