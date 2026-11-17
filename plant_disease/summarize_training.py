"""Turn a completed training run into a shareable table and learning curves.

Run from the repository root:
    python -m plant_disease.summarize_training --run-dir outputs/runs/source-seed-551

Reads recorded metrics and checkpoint bytes, not images or test manifests. Writes
a small JSON summary, Markdown explanation and PNG plot under docs/training-runs/.
Personal absolute paths and optimizer/model weights are not exported.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import augmentation_recipe, sha256_file, training_data_record


def describe_training_data(record):
    """Explain the recorded population rather than assuming source-only input."""
    if record is None:
        return "The recorded training and validation images are PlantVillage only; no PlantDoc training images were used."
    names = {"plantvillage": "PlantVillage", "plantdoc": "PlantDoc"}
    descriptions = []
    for split in ("train", "validation"):
        counts = record["images_by_dataset"][split]
        images = ", ".join(f"{count:,} {names[dataset]}"
                           for dataset, count in sorted(counts.items()))
        descriptions.append(f"{split.capitalize()}: {images} images.")
    if record["include_plantdoc_train"]:
        descriptions.append("This experiment allows PlantDoc training data and splits the combined training pool. "
                            "It is not a source-only transfer experiment; its validation scores are not directly "
                            "comparable to the original PlantVillage-only validation scores.")
    else:
        descriptions.append("No PlantDoc training images were used.")
    return " ".join(descriptions)


def is_followup(settings):
    return settings.get("augmentation", "none") != "none" or settings.get("include_plantdoc_train", False)


def protect_report_destination(destination, filenames, followup=False):
    """Keep new experiments separate and never replace an existing report."""
    root = Path(__file__).resolve().parents[1]
    if followup and any(destination.resolve().is_relative_to(root / folder)
                       for folder in ("docs/training-runs", "docs/evaluation")):
        raise ValueError("Use a separate --output-dir for follow-up experiment reports")
    if any((destination / name).exists() for name in filenames):
        raise ValueError("Report files already exist; choose a new output directory")


def collect_summary(run_dir):
    """Use the chosen checkpoint's metrics, rather than the last epoch's scores."""
    metadata = json.loads((run_dir / "run.json").read_text())
    results = json.loads((run_dir / "comparison.json").read_text())
    if metadata["scope"] not in ("source_training_and_validation", "mixed_training_and_validation"):
        raise ValueError("Use a completed full run, not a feasibility trial")
    if set(results) != {"frozen", "finetune"}:
        raise ValueError("Both model versions must finish before comparison")
    for split, digest in metadata["selected_manifest_sha256"].items():
        if sha256_file(run_dir / f"selected_{split}.csv") != digest:
            raise ValueError("Selected image list changed after training")
    if "training_data" in metadata:
        # New records must agree with their actual CSVs and transform recipe.
        rows = {}
        for split in ("train", "validation"):
            with (run_dir / f"selected_{split}.csv").open(newline="") as stream:
                rows[split] = list(csv.DictReader(stream))
        include = metadata["arguments"].get("include_plantdoc_train", False)
        if (training_data_record(rows, include) != metadata["training_data"]
                or {key: len(value) for key, value in rows.items()} != metadata["counts"]):
            raise ValueError("Recorded training population differs from the selected images")
        if metadata["preprocessing"]["augmentation"] != augmentation_recipe(metadata["arguments"].get("augmentation", "none")):
            raise ValueError("Recorded augmentation differs from the selected option")
        if (include != (metadata["scope"] == "mixed_training_and_validation")
                or include != bool(metadata["split_summary"].get("include_plantdoc_train", False))
                or (not include and any(r["dataset"] != "plantvillage" for records in rows.values() for r in records))):
            raise ValueError("Recorded training policy is inconsistent")
    for key in ("initial_feature_sha256", "initial_classifier_sha256"):
        if results["frozen"][key] != results["finetune"][key]:
            raise ValueError("The two starting models did not match")
    modes = {}
    for mode, result in results.items():
        history = result["history"]
        eligible = [h for h in history if h["epoch"] > metadata["arguments"]["warmup_epochs"]]
        best = min(eligible, key=lambda h: h["validation"]["loss"])
        if best["epoch"] != result["best_epoch"] or result["trial"]:
            raise ValueError("Checkpoint selection does not match the full-run record")
        modes[mode] = {
            "epochs_completed": result["epochs_completed"], "best_epoch": best["epoch"],
            "validation": best["validation"],
            "loop_seconds": sum(h["train"]["seconds"] + h["validation"]["seconds"] for h in history),
            "peak_allocated_mib": max((h["peak_allocated_mib"] for h in history
                                        if h["peak_allocated_mib"] is not None), default=None),
            "peak_reserved_mib": max((h["peak_reserved_mib"] for h in history
                                       if h["peak_reserved_mib"] is not None), default=None),
            "best_checkpoint_sha256": sha256_file(run_dir / mode / "best.pt"),
            "batch_norm_statistics_unchanged": result["batch_norm_statistics_unchanged"],
            "features_changed": result["initial_feature_sha256"] != result["final_feature_sha256"],
            "history": history,
        }
    # Explicitly select portable settings; never copy local absolute paths.
    settings = {key: metadata["arguments"][key] for key in (
        "epochs", "warmup_epochs", "batch_size", "workers", "seed", "head_lr",
        "backbone_lr", "weight_decay", "patience", "device")}
    for key in ("augmentation", "include_plantdoc_train"):
        if key in metadata["arguments"]:
            settings[key] = metadata["arguments"][key]
    return {"scope": "validation only; not final test results",
            "training_data": metadata.get("training_data"),
            "settings": settings, "counts": metadata["counts"],
            "software": metadata["software"], "device": metadata["device"],
            "preprocessing": metadata["preprocessing"],
            "weights_sha256": metadata["weights_sha256"],
            "code_sha256": metadata["code_sha256"],
            "selected_manifest_sha256": metadata["selected_manifest_sha256"],
            "categories_sha256": metadata["split_summary"]["categories_sha256"],
            "checkpoint_rule": metadata["checkpoint_rule"], "models": modes}


def write_report(summary, destination, name):
    """Export the same measured values as a table and a two-panel plot."""
    protect_report_destination(destination, [f"{name}.{ext}" for ext in ("json", "md", "png")],
                               is_followup(summary["settings"]))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rows = []
    for mode, result in summary["models"].items():
        label = "Frozen features" if mode == "frozen" else "Fine-tuned features"
        history = result["history"]
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["validation"]["loss"] for h in history], marker="o", label=label)
        axes[1].plot(epochs, [h["validation"]["macro_f1"] for h in history], marker="o", label=label)
        metrics = result["validation"]
        memory = f"{result['peak_allocated_mib']:.0f}" if result["peak_allocated_mib"] is not None else "N/A"
        rows.append(f"| {label} | {result['epochs_completed']} | {result['best_epoch']} | "
                    f"{metrics['accuracy']:.2%} | {metrics['macro_f1']:.4f} | {metrics['loss']:.4f} | "
                    f"{result['loop_seconds'] / 60:.1f} | {memory} |")
    for ax, title, ylabel in zip(axes, ("Validation error", "Validation macro-F1"),
                                 ("Cross-entropy loss (lower is better)", "Macro-F1 (higher is better)")):
        ax.set(title=title, xlabel="Epoch (one pass through training images)", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.savefig(destination / f"{name}.png", dpi=160)
    plt.close(fig)
    counts, settings = summary["counts"], summary["settings"]
    text = f"""# Training comparison: seed {settings['seed']}

These are **validation results**, not final test results.
{describe_training_data(summary.get('training_data'))}
Both models used {counts['train']:,} training and {counts['validation']:,} validation images.
Training augmentation: `{settings.get('augmentation', 'none')}`. The complete
recipe is in the JSON record; validation images receive no random augmentation.

| Model | Epochs completed | Selected epoch | Validation accuracy | Macro-F1 | Validation loss | Loop minutes | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + f"""

Accuracy is the fraction classified correctly. Macro-F1 gives each category
equal weight when combining precision and recall. The checkpoint was selected
using the lowest validation loss after warmup, not the highest accuracy or F1.
All three metrics in each row come from that same selected epoch.

![Validation learning curves]({name}.png)

The initial budget was {settings['epochs']} epochs, including
{settings['warmup_epochs']} head-only warmup epoch, batch size {settings['batch_size']},
and random seed {settings['seed']}. Early stopping waits for {settings['patience']}
post-warmup epochs without lower validation loss. The category layer learning
rate is {settings['head_lr']} and the unfrozen feature learning rate is
{settings['backbone_lr']}. Both use identical image lists and starting weights.

Loop times include training and validation, but exclude setup, checkpoint writing,
and reporting. GPU memory is PyTorch's allocated memory, excluding CUDA context
and other applications. Different stopping points mean different total work.

This is one pair of runs. It does not measure variation across random seeds,
prove robustness, or establish performance on held-out PlantDoc images. Validation influenced
checkpoint selection, so these scores cannot substitute for held-out test scores.
Fix experiment settings before evaluation. Follow-up experiments after viewing
the original test results are exploratory; they do not create a fresh unseen test set.

See [the machine-readable record]({name}.json) for all epoch metrics, settings,
software versions, and fingerprints. See `docs/training.md` in the repository
for reproduction instructions. Model weights remain in ignored local outputs.
"""
    (destination / f"{name}.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/training-runs"))
    args = parser.parse_args()
    summary = collect_summary(args.run_dir)
    write_report(summary, args.output_dir, args.run_dir.name)
    print(args.output_dir / f"{args.run_dir.name}.md")


if __name__ == "__main__":
    main()
