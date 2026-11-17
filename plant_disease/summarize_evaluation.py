"""Check saved test predictions and export the final comparison without inference.

Run: python -m plant_disease.summarize_evaluation --evaluation-dir outputs/evaluation/final-01
Checks every saved prediction CSV against its fingerprint and recorded metrics.
Writes a Markdown report, JSON evidence, per-class CSV, and two plots under
docs/evaluation/. Does not open images, fit models, or change selected checkpoints.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import sha256_file
from .summarize_training import describe_training_data, is_followup, protect_report_destination


MODES = ("frozen", "finetune")
SPLITS = ("source_test", "target_test")
NAMES = {"frozen": "Frozen features", "finetune": "Fine-tuned features",
         "source_test": "PlantVillage", "target_test": "PlantDoc"}


def spread(values):
    return {"values": values, "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values)}


def check_predictions(directory, record, class_count):
    """Recompute the confusion matrix and scores directly from recorded labels."""
    filename = record["predictions_file"]
    if Path(filename).name != filename:
        raise ValueError("Prediction file must be inside the evaluation folder")
    path = directory / filename
    if sha256_file(path) != record["predictions_sha256"]:
        raise ValueError("Prediction file changed after evaluation")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matrix = np.zeros((class_count, class_count), dtype=int)
    identities = []
    for row in rows:
        truth, guess = int(row["true_class_id"]), int(row["predicted_class_id"])
        if not (0 <= truth < class_count and 0 <= guess < class_count):
            raise ValueError("Saved prediction has an unknown category")
        if not 0 <= float(row["confidence"]) <= 1:
            raise ValueError("Invalid prediction confidence")
        matrix[truth, guess] += 1
        identities.append((row["relative_path"], truth))
    if (not rows or len({p for p, _ in identities}) != len(rows)
            or len(rows) != record["images"]
            or not np.array_equal(matrix, record["confusion_matrix"])):
        raise ValueError("Prediction count/order or confusion matrix mismatch")
    support, predicted_count = matrix.sum(1), matrix.sum(0)
    correct = matrix.diagonal()
    recall = np.divide(correct, support, out=np.zeros(class_count), where=support != 0)
    precision = np.divide(correct, predicted_count, out=np.zeros(class_count), where=predicted_count != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(class_count), where=precision+recall != 0)
    if abs(float(correct.sum()/len(rows)) - record["accuracy"]) > 1e-10 or abs(float(f1.mean()) - record["macro_f1"]) > 1e-10:
        raise ValueError("Recorded overall scores disagree with saved predictions")
    if len(record["per_class"]) != class_count:
        raise ValueError("Missing per-class results")
    for i, item in enumerate(record["per_class"]):
        if item["id"] != i or item["support"] != int(support[i]):
            raise ValueError("Wrong per-class order or count")
        for key, expected in (("precision", precision[i]), ("recall", recall[i]), ("f1", f1[i])):
            if abs(item[key] - expected) > 1e-10:
                raise ValueError("Per-class score mismatch")
    return identities


def collect_evaluation(directory):
    raw = json.loads((directory / "results.json").read_text())
    if not raw["model_states_unchanged"]:
        raise ValueError("Evaluation changed model state")
    metadata, records = raw["metadata"], raw["results"]
    selection = metadata["selection"]
    expected = {(s, m, d) for s in selection["seeds"] for m in MODES for d in SPLITS}
    keys = [(r["seed"], r["mode"], r["split"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Need each selected model on both datasets exactly once")
    chosen = {(c["seed"], c["mode"]): c for c in selection["checkpoints"]}
    identities = {}
    for record in records:
        entry = chosen[record["seed"], record["mode"]]
        if record["epoch"] != entry["epoch"] or record["checkpoint_sha256"] != entry["sha256"]:
            raise ValueError("Results do not match the preselected checkpoint")
        if record["images"] != metadata["split_summary"]["counts"][record["split"]]:
            raise ValueError("Wrong test image count")
        order = check_predictions(directory, record, len(metadata["categories"]))
        if record["split"] in identities and order != identities[record["split"]]:
            raise ValueError("Models were evaluated on different images or labels")
        identities[record["split"]] = order
    records.sort(key=lambda r: (r["seed"], MODES.index(r["mode"]), SPLITS.index(r["split"])))
    groups, gaps = {}, {}
    for mode in MODES:
        groups[mode] = {}
        for split in SPLITS:
            group = [r for r in records if r["mode"] == mode and r["split"] == split]
            groups[mode][split] = {metric: spread([r[metric] for r in group]) for metric in ("accuracy", "macro_f1")}
            groups[mode][split]["images"] = group[0]["images"]
        gaps[mode] = {metric: spread([a-b for a, b in zip(groups[mode]["source_test"][metric]["values"],
                                                        groups[mode]["target_test"][metric]["values"], strict=True)])
                      for metric in ("accuracy", "macro_f1")}
    return {"metadata": metadata, "results": records, "groups": groups,
            "source_minus_target_gap": gaps, "saved_predictions_verified": True}


def write_report(summary, destination):
    protect_report_destination(destination,
        ["final-results.json", "final-results.md", "per-class-results.csv", "test-comparison.png", "confusion-matrices.png"],
        is_followup(summary["metadata"].get("training_settings", {})))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "final-results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    records, classes = summary["results"], summary["metadata"]["categories"]
    class_count = len(classes)
    # Per-category means retain the real number of distinct test images. Three
    # model runs do not turn 183 photos into 549 independent photographs.
    class_rows = []
    for mode in MODES:
        for split in SPLITS:
            group = [r for r in records if r["mode"] == mode and r["split"] == split]
            for i, category in enumerate(classes):
                row = {"dataset": NAMES[split], "model": mode, "class_id": i,
                       "plantvillage_label": category["plantvillage"], "plantdoc_label": category["plantdoc"],
                       "distinct_test_images": group[0]["per_class"][i]["support"]}
                for metric in ("precision", "recall", "f1"):
                    values = [r["per_class"][i][metric] for r in group]
                    row[f"mean_{metric}"] = statistics.mean(values)
                    row[f"sample_sd_{metric}"] = statistics.stdev(values)
                class_rows.append(row)
    with (destination / "per-class-results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, scale, ylabel in zip(axes, ("accuracy", "macro_f1"), (100, 1), ("Accuracy (%)", "Macro-F1")):
        for offset, mode in zip((-.18, .18), MODES):
            stats = [summary["groups"][mode][split][metric] for split in SPLITS]
            ax.bar(np.arange(2)+offset, [scale*s["mean"] for s in stats], width=.36,
                   yerr=[scale*s["sample_standard_deviation"] for s in stats], capsize=4, label=NAMES[mode])
        ax.set(xticks=[0, 1], xticklabels=[NAMES[s] for s in SPLITS], ylabel=ylabel, ylim=(0, scale))
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.savefig(destination / "test-comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for row, mode in enumerate(MODES):
        for col, split in enumerate(SPLITS):
            matrices = np.array([r["confusion_matrix"] for r in records if r["mode"] == mode and r["split"] == split])
            row_totals = matrices.sum(2, keepdims=True)
            proportions = np.divide(matrices*100., row_totals, out=np.zeros_like(matrices, dtype=float), where=row_totals != 0)
            ax = axes[row, col]
            im = ax.imshow(proportions.mean(0), vmin=0, vmax=100, cmap="Blues")
            ax.set(title=f"{NAMES[mode]} — {NAMES[split]}", xlabel="Predicted category ID", ylabel="True category ID",
                   xticks=range(class_count), yticks=range(class_count))
            ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Mean percent of each true category across three runs", shrink=.8)
    fig.savefig(destination / "confusion-matrices.png", dpi=160)
    plt.close(fig)
    rows, individual, gap_rows, labels = [], [], [], []
    for mode in MODES:
        for split in SPLITS:
            group = summary["groups"][mode][split]
            a, f = group["accuracy"], group["macro_f1"]
            rows.append(f"| {NAMES[mode]} | {NAMES[split]} | {group['images']:,} | "
                        f"{100*a['mean']:.2f}% ± {100*a['sample_standard_deviation']:.2f} pp | "
                        f"{f['mean']:.4f} ± {f['sample_standard_deviation']:.4f} |")
        gap = summary["source_minus_target_gap"][mode]["accuracy"]
        gap_rows.append(f"| {NAMES[mode]} | {100*gap['mean']:.2f} ± {100*gap['sample_standard_deviation']:.2f} percentage points |")
    for r in records:
        individual.append(f"| {r['seed']} | {NAMES[r['mode']]} | {NAMES[r['split']]} | "
                          f"{r['accuracy']:.2%} | {r['macro_f1']:.4f} |")
    support = {split: next(r for r in records if r["split"] == split)["per_class"] for split in SPLITS}
    for i, category in enumerate(classes):
        labels.append(f"| {i} | {category['plantvillage'].replace('___', ' — ').replace('_', ' ')} | "
                      f"{support['source_test'][i]['support']} | {support['target_test'][i]['support']} |")
    failure_notes = []
    for mode in MODES:
        group = [r for r in records if r["mode"] == mode and r["split"] == "target_test"]
        counts = np.array([r["confusion_matrix"] for r in group]).sum(1)
        common = int(counts.sum(0).argmax())
        name = classes[common]["plantvillage"].replace("___", " — ").replace("_", " ")
        failure_notes.append(f"{NAMES[mode]} most often predicted **{name}** on PlantDoc: "
                             f"{counts[:, common].min()}–{counts[:, common].max()} of "
                             f"{group[0]['images']} images per run, while "
                             f"{support['target_test'][common]['support']} images carried that annotation.")
    fine = summary["groups"]["finetune"]
    main_finding = (f"The fine-tuned models averaged {fine['source_test']['accuracy']['mean']:.2%} on "
                    f"PlantVillage and {fine['target_test']['accuracy']['mean']:.2%} on PlantDoc. "
                    "These scores describe the recorded models on these selected test images; "
                    "they do not establish general robustness.")
    population = describe_training_data(summary["metadata"].get("training_data"))
    augmentation = summary["metadata"].get("training_settings", {}).get("augmentation", "none")
    text = f"""# Test results: PlantVillage and PlantDoc

All six models selected using their recorded validation set were evaluated on
the reserved test images. Evaluation did not update model weights.

{population}
Training augmentation: `{augmentation}`. Test images receive no random augmentation.
Follow-up experiments after viewing the original test scores are exploratory;
reusing those images does not provide a fresh unseen test set.

## Overall results

| Model | Test dataset | Distinct images | Mean accuracy ± sample SD | Mean macro-F1 ± sample SD |
| --- | --- | ---: | ---: | ---: |
""" + "\n".join(rows) + "\n\n" + main_finding + """

Accuracy is the fraction classified correctly. Macro-F1 combines precision and
recall while giving each selected category equal weight. The ± values are
the sample standard deviation across the three training seeds. They are not
confidence intervals or uncertainty from drawing different test photographs.
“pp” means percentage points. All three runs reuse the same test images.

![Test comparison](test-comparison.png)

The error bars show variation across seeds. These results describe the selected
category subsets, not the entire original datasets or all casual-user photos.

## Difference between datasets

Each gap is PlantVillage accuracy minus PlantDoc accuracy for the same model
and seed, then averaged across the three seeds. Positive means lower accuracy
on PlantDoc; negative would mean higher accuracy on PlantDoc.

| Model | Mean source-minus-target accuracy gap ± sample SD |
| --- | ---: |
""" + "\n".join(gap_rows) + """

This comparison measures a difference between datasets. It does not isolate
photography skill, weather, background, disease severity, label quality, or
class proportions as causes. Those factors may differ together. No causal
claim about weather or photographer ability follows from these scores.

PlantDoc contains a limited number of eligible test images; the category counts
below show the support for this particular experiment. A few errors
can strongly affect category scores. The three runs do not provide three
independent test sets. Known duplicates and resolved leaf groups were separated,
but missing leaf metadata and re-encoded/cropped copies remain limitations.
Folder annotations have not been independently diagnosed. The recorded category
mapping identifies which pairs were selected before training.

## Every model and seed

| Seed | Model | Test dataset | Accuracy | Macro-F1 |
| --- | --- | --- | ---: | ---: |
""" + "\n".join(individual) + """

## Confusion matrices and category counts

Rows are the annotated categories; columns are the model's guesses. Dark cells
on the diagonal mean correct predictions. Off-diagonal cells show confusions.
Each row is expressed as a percentage, averaged across the three seeds for
that model. These are averages of separate models, not ensemble predictions.
The JSON contains every individual matrix as raw image counts.

""" + "\n\n".join(failure_notes) + """

This describes a prediction pattern, not its cause. The category table below
connects the matrix IDs to the original labels.

![Average row-normalized confusion matrices](confusion-matrices.png)

| ID | Shared category (PlantVillage name) | PlantVillage test images | PlantDoc test images |
| --- | --- | ---: | ---: |
""" + "\n".join(labels) + """

The [per-class CSV](per-class-results.csv) includes both datasets' original names,
counts, and mean/spread of precision, recall, and F1. Recall answers: “Of the
images actually in this category, how many did the model recognize?”

## Reproduction and checks

The [machine-readable evidence](final-results.json) records every model's scores,
confusion matrix, class metrics, checkpoint and image-list fingerprints, software,
and evaluation settings. Per-image prediction CSVs stay in the ignored local
evaluation folder. The exporter checks their fingerprints and independently
recomputes accuracy, macro-F1, class metrics, and confusion counts from them.
It also verifies that all models saw identical ordered test images and labels.
Evaluation confirmed that no model weights or buffers changed.

See `docs/evaluation.md` in the repository for commands and `docs/training.md`
for training and checkpoint selection. Use the measured results and stated
limits when writing up this experiment.
"""
    (destination / "final-results.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/evaluation/final-01"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/evaluation"))
    args = parser.parse_args()
    summary = collect_evaluation(args.evaluation_dir)
    write_report(summary, args.output_dir)
    print(args.output_dir / "final-results.md")


if __name__ == "__main__":
    main()
