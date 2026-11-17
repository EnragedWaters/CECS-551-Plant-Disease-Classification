# Optional follow-up experiments

The submitted comparison used PlantVillage training images and **no random
augmentation**. That remains the default. The options below prepare separate
experiments; their availability is not evidence that they improve accuracy.

| Option | What changes |
| --- | --- |
| `--augmentation none` (default) | No random image changes. RGB conversion, resizing, padding, and normalization still happen. |
| `--augmentation basic-v1` | Training images receive random crops, flips, rotations, and color changes. Validation and test images stay unchanged. |
| `--include-plantdoc-train` | Allows eligible PlantDoc **training** images into a separately prepared training/validation pool. Never allows PlantDoc test images into training. |

The options are independent. Try one change at a time if the aim is to understand
what helped. Combining both cannot tell us which change caused a difference.

## Augmentation only

Use the original image lists and a new output folder. From the repository root,
with the project Python environment active:

```bash
python -m plant_disease.train --trial --augmentation basic-v1 --output-dir outputs/trials/augmentation-01
```

This is a short feasibility check, not a final accuracy experiment. For a full
run, omit `--trial` and use a fresh folder such as
`outputs/experiments/augmentation-01/runs/source-seed-551`.

`basic-v1` applies these operations in order, before the usual normalization:

- Random crop resized to 224 × 224: 70–100% of image area, aspect ratio 0.75–1.33,
  bilinear interpolation with antialiasing.
- Horizontal flip with a 50% chance.
- Rotation between −20° and +20°, nearest-neighbor interpolation, fixed canvas,
  and the usual mean-color fill.
- Brightness, contrast, and saturation factors between 0.75 and 1.25, with hue
  shifted between −0.05 and +0.05. Color adjustments have a random order.

The full recipe is saved in `run.json` and every checkpoint. It is built from
the same configuration used to construct the transforms. Cropping may remove
disease clues, and color changes may alter symptoms; usefulness must be measured.
Raw image files are never edited.

## Including PlantDoc training images

First prepare separate lists from the existing audit, then explicitly opt in
when training:

```bash
python scripts/build_splits.py --include-plantdoc-train --output-dir outputs/splits-plantdoc-01
python -m plant_disease.train --trial --include-plantdoc-train --splits-dir outputs/splits-plantdoc-01 --output-dir outputs/trials/plantdoc-01
```

Augmentation stays off unless `--augmentation basic-v1` is also supplied.
The split builder refuses a nonempty output directory and refuses to put mixed
lists in the original `outputs/splits/` folder. Training checks that its option
matches the policy recorded with the lists.

This splits the combined training pool, so **validation can also contain
PlantDoc images**, and some PlantVillage train/validation assignments may change.
It answers a different question from the original source-only transfer study.
Do not directly compare its validation accuracy with the original validation
accuracy as if both used the same photographs. Reports show actual counts by
dataset and explicitly identify the changed experiment. Test-image lists remain
reserved, with known duplicate/group overlap checks across the lists.

## Full runs and reports

For an augmentation-only experiment, the following is an example schedule.
These commands train real models; they were not run as part of the code repair.
Keep run names `source-seed-551`, `source-seed-552`, and `source-seed-553` inside
a separate experiment folder because the three-seed collector expects them.
The folder name is just a compatibility convention; the saved policy describes
the actual data used.

```bash
for seed in 551 552 553; do
  python -m plant_disease.train --seed "$seed" --augmentation basic-v1 --output-dir "outputs/experiments/augmentation-01/runs/source-seed-$seed"
  python -m plant_disease.verify_training --run-dir "outputs/experiments/augmentation-01/runs/source-seed-$seed"
  python -m plant_disease.summarize_training --run-dir "outputs/experiments/augmentation-01/runs/source-seed-$seed" --output-dir outputs/experiments/augmentation-01/reports
done
python -m plant_disease.summarize_repetitions --runs-root outputs/experiments/augmentation-01/runs --output-dir outputs/experiments/augmentation-01/reports
```

For mixed-data runs, use a different experiment folder, pass
`--include-plantdoc-train --splits-dir outputs/splits-plantdoc-01` to training,
and pass the same `--splits-dir` to verification. Keep the three seeds, budgets,
software, worker count, image lists, and options consistent within a comparison.
Repeatability checks cover both zero and multiple image-loading workers;
identical numbers across different hardware/software are not promised.

Before any test evaluation, freeze the settings and record the selection.
Evaluation accepts `--runs-root`, `--selection`, `--splits-dir`, and a fresh
`--output-dir`. Point all four at this experiment's artifacts. It reads the
training policy from the selection record and never augments test images.
The evaluation exporter also needs its own `--output-dir`.

Our original test results have already been viewed. Later experiments on those
same test images are **exploratory**, not fresh unseen confirmation. Do not tune
repeatedly against those scores and describe the winning result as an unbiased
final test.

## Preserve the original results

Existing runs and report files cannot be overwritten. Follow-up report exporters
also reject the original `docs/training-runs/` and `docs/evaluation/` locations.
Keep generated experiment records in `outputs/experiments/`, which Git ignores.
Review any evidence separately before deciding to publish it.

Original checkpoints retain fingerprints of the code that created them.
**Use the original code at commit `9619975` to verify or evaluate those existing
checkpoints**, with their matching saved image lists and local data paths.
Changed training code should fail that fingerprint check; do not remove the
check or rewrite old metadata to force a match. New runs receive new records.
Any earlier run that says `augmentation: none` despite actually augmenting needs
its original code and evidence reviewed; this repair does not retroactively
validate it or relabel it as a successful experiment.
