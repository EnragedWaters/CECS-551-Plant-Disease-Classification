"""Compare the three declared training seeds without using final test images.

Run: python -m plant_disease.summarize_repetitions
Requires completed and verified runs for seeds 551, 552, 553 in outputs/runs/.
Writes a portable report, plot, JSON measurements and selected-checkpoint list
under docs/training-runs/. Does not train models or evaluate test images.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .summarize_training import collect_summary


SEEDS = (551, 552, 553)
CORE_FILES = ("__init__.py", "data.py", "model.py", "train.py")


def aggregate(summaries):
    """Reject unlike experiments; report sample spread across all declared seeds."""
    seeds = [s["settings"]["seed"] for s in summaries]
    if sorted(seeds) != list(SEEDS):
        raise ValueError("Need exactly one completed pair for each declared seed: 551, 552, 553")
    summaries = sorted(summaries, key=lambda s: s["settings"]["seed"])
    reference = summaries[0]
    for summary in summaries:
        settings = {k: v for k, v in summary["settings"].items() if k != "seed"}
        expected = {k: v for k, v in reference["settings"].items() if k != "seed"}
        if settings != expected:
            raise ValueError("Training settings differ between seeds")
        for key in ("counts", "software", "device", "preprocessing", "weights_sha256",
                    "selected_manifest_sha256", "categories_sha256", "checkpoint_rule",
                    "precision", "deterministic_algorithms", "split_summary"):
            if summary[key] != reference[key]:
                raise ValueError(f"Experiments differ: {key}")
        # Reporting helpers may have been added since the first run; compare
        # only the files that actually build inputs and train the models.
        for name in CORE_FILES:
            if summary["code_sha256"][name] != reference["code_sha256"][name]:
                raise ValueError(f"Training implementation differs: {name}")
        if set(summary["models"]) != {"frozen", "finetune"}:
            raise ValueError("Each seed needs both model versions")
        for mode, result in summary["models"].items():
            if not result["batch_norm_statistics_unchanged"] or result["features_changed"] != (mode == "finetune"):
                raise ValueError("Unexpected frozen or trainable weight changes")
        for index in range(summary["settings"]["warmup_epochs"]):
            for split in ("train", "validation"):
                for metric in ("images", "loss", "accuracy", "macro_f1"):
                    if (summary["models"]["frozen"]["history"][index][split][metric] !=
                            summary["models"]["finetune"]["history"][index][split][metric]):
                        raise ValueError("The paired warmup results do not match")
    statistics_by_mode = {}
    for mode in ("frozen", "finetune"):
        statistics_by_mode[mode] = {}
        for metric in ("accuracy", "macro_f1", "loss"):
            values = [s["models"][mode]["validation"][metric] for s in summaries]
            statistics_by_mode[mode][metric] = {
                "values": values, "mean": statistics.mean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "min": min(values), "max": max(values),
            }
    differences = [s["models"]["finetune"]["validation"]["accuracy"] -
                   s["models"]["frozen"]["validation"]["accuracy"] for s in summaries]
    return {"scope": "source validation only; three seeds on one fixed split",
            "seeds": list(SEEDS), "statistics": statistics_by_mode,
            "paired_accuracy_difference": {"values": differences, "mean": statistics.mean(differences),
                                            "sample_standard_deviation": statistics.stdev(differences)},
            "runs": summaries}


def collect(runs_root):
    summaries, checkpoints = [], []
    for seed in SEEDS:
        run_dir = runs_root / f"source-seed-{seed}"
        summary = collect_summary(run_dir)
        metadata = json.loads((run_dir / "run.json").read_text())
        checks = json.loads((run_dir / "checkpoint-verification.json").read_text())
        for key in ("precision", "deterministic_algorithms", "split_summary"):
            summary[key] = metadata[key]
        if metadata["test_manifests_opened"] or summary["settings"]["seed"] != seed:
            raise ValueError("Unexpected test access or seed in run metadata")
        for mode, result in summary["models"].items():
            check = checks[mode]
            if (not check["reloaded_validation_matches_record"] or check["epoch"] != result["best_epoch"]
                    or check["best_checkpoint_sha256"] != result["best_checkpoint_sha256"]):
                raise ValueError("Selected checkpoint has not passed reload verification")
            for metric in ("images", "loss", "accuracy", "macro_f1"):
                if abs(check["measured"][metric] - result["validation"][metric]) > 1e-7:
                    raise ValueError("Reload verification scores differ from the report")
            checkpoints.append({"seed": seed, "mode": mode, "run_name": run_dir.name,
                                "checkpoint": f"{mode}/best.pt", "epoch": result["best_epoch"],
                                "sha256": result["best_checkpoint_sha256"]})
        summaries.append(summary)
    combined = aggregate(summaries)
    reference = summaries[0]
    selection = {"status": "selected using source validation before final test evaluation",
                 "selection_rule": reference["checkpoint_rule"], "seeds": list(SEEDS),
                 "categories_sha256": reference["categories_sha256"],
                 "split_manifest_sha256": reference["split_summary"]["manifest_sha256"],
                 "checkpoints": checkpoints}
    return combined, selection


def write_report(combined, selection, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in (("repeated-training", combined), ("selected-checkpoints", selection)):
        (destination / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    rows, averages = [], []
    names = {"frozen": "Frozen features", "finetune": "Fine-tuned features"}
    for run in combined["runs"]:
        for mode, result in run["models"].items():
            score = result["validation"]
            rows.append(f"| {run['settings']['seed']} | {names[mode]} | {result['epochs_completed']} | "
                        f"{result['best_epoch']} | {score['accuracy']:.2%} | {score['macro_f1']:.4f} | "
                        f"{score['loss']:.4f} | {result['loop_seconds']/60:.1f} |")
    for mode, metrics in combined["statistics"].items():
        acc, f1 = metrics["accuracy"], metrics["macro_f1"]
        averages.append(f"| {names[mode]} | {100*acc['mean']:.2f}% ± "
                        f"{100*acc['sample_standard_deviation']:.2f} percentage points | "
                        f"{f1['mean']:.4f} ± {f1['sample_standard_deviation']:.4f} |")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in names:
        for ax, metric, scale in zip(axes, ("accuracy", "macro_f1"), (100, 1)):
            values = combined["statistics"][mode][metric]["values"]
            ax.plot(SEEDS, [scale*x for x in values], marker="o", linestyle="none", label=names[mode])
    for ax, ylabel in zip(axes, ("Validation accuracy (%)", "Validation macro-F1")):
        ax.set(xlabel="Training seed (same image split)", ylabel=ylabel, xticks=SEEDS)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(destination / "repeated-training.png", dpi=160)
    plt.close(fig)
    counts = combined["runs"][0]["counts"]
    classes = combined["runs"][0]["split_summary"]["classes"]
    text = f"""# Training repeated three times

All three planned pairs are complete. These are **PlantVillage validation
results**, not final test scores. Each pair used the same {counts['train']:,} training and
{counts['validation']:,} validation images across {classes} categories. Only the training seed changed.
That changes the new classifier's initial weights and the training image order.

| Seed | Model | Epochs completed | Selected epoch | Validation accuracy | Macro-F1 | Loss | Loop minutes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + """

The selected epoch had the lowest validation loss after warmup. The table uses
that epoch's accuracy and macro-F1, even if another epoch scored higher on those
metrics. Early stopping and the ten-epoch limit are the same for all runs.
Loop time excludes setup, saving checkpoints, and the later verification pass.

| Model | Mean validation accuracy ± sample standard deviation | Mean macro-F1 ± sample standard deviation |
| --- | --- | --- |
""" + "\n".join(averages) + """

The ± values describe variation across three training seeds. They are not
confidence intervals, uncertainty from different image splits, or evidence of
statistical significance. All runs share one validation set, and that set was
used to select their checkpoints. We report all seeds, not a winning seed.

![Variation between the three training seeds](repeated-training.png)

Each point is a separate run; a larger seed number has no directional meaning.
The plot axes are zoomed to show small differences; the full metric ranges are
0–100% accuracy and 0–1 macro-F1. Strong validation scores do not establish
performance on PlantDoc, casual photographs, or new weather conditions.

Every saved model was reloaded and reproduced its scores on all validation
images. Frozen feature weights stayed fixed; fine-tuned feature weights changed;
batch-normalization running statistics stayed fixed in both versions. Training
settings, source image lists, preprocessing, and core training code match across
all three seeds. Both models had matching warmup results within each pair.

The [selected-checkpoint record](selected-checkpoints.json) identifies all six
models by their file fingerprints before final test evaluation. The next step
is to evaluate **all six** on the same reserved PlantVillage and PlantDoc test
lists, without changing settings or choosing seeds based on test results.
Neither final test set was evaluated during these training repetitions.

See [all recorded measurements](repeated-training.json), the individual reports
for [551](source-seed-551.md), [552](source-seed-552.md), and
[553](source-seed-553.md), and [run instructions](../training.md).
"""
    (destination / "repeated-training.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/training-runs"))
    args = parser.parse_args()
    combined, selection = collect(args.runs_root)
    write_report(combined, selection, args.output_dir)
    print(args.output_dir / "repeated-training.md")


if __name__ == "__main__":
    main()
