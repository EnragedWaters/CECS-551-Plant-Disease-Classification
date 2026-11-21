# Training the two model versions

The training code, GPU trial, and all three planned training pairs are complete.
The [combined results](training-runs/repeated-training.md) show every seed, mean
validation scores, and variation between runs. All six selected models passed
reload verification. Final testing is also complete: see the
[held-out test results](evaluation/final-results.md) and [evaluation guide](evaluation.md).
The validation scores below are separate from those final measurements.

The commands below describe the original source-only method. Augmentation and
PlantDoc training are off by default. See [optional experiments](optional-experiments.md)
for explicit options and separate output folders. Existing checkpoints must be
verified with their original code (`9619975`); new code fingerprints will differ.
See [reproduction instructions](reproduction.md) for the exact original-version
commands, including separate worktrees and fresh-run selection records.

## Run a small trial

Activate the environment from [setup.md](setup.md), and run from the repository
root. The datasets, local ImageNet weights, and [split manifests](experiment-data.md)
must already exist.

```bash
python -m plant_disease.train --trial --output-dir outputs/trials/my-first-trial
```

This runs both versions for two epochs (two passes through the selected training
images), using up to 32 training and eight validation images per category.
The first epoch trains only the new category layer in both versions. During the
second epoch, the fine-tuned version also updates the earlier image-processing
weights. The frozen version continues learning only its category layer.

Choose a new output folder for each run; existing runs cannot be overwritten.
Report exporters also refuse to replace existing files; choose a fresh report
folder when regenerating them. The examples keep regenerated reports in ignored outputs.
The default dataset location is `~/datasets/cecs551-plant-disease`. Use
`--data-root /path/to/datasets` if yours differs, or `--weights /path/to/file.pth`
to specify the previously downloaded weights separately. Nothing is downloaded.
CUDA is required by default; `--device cpu` explicitly chooses the slower CPU.
`--workers 0` disables background image-loading workers if they cause trouble.

## What happens to each image

The loader checks the file against its recorded checksum, corrects orientation,
converts it to RGB, and resizes it to fit inside 224 × 224 pixels without
stretching or cropping. It fills any remaining border with approximately the
ImageNet mean color, then applies ImageNet normalization (a fixed rescaling of
pixel values). Transparency is flattened onto that same color.

This deliberately uses padding instead of the reference weights' center crop,
so leaves near the edge remain visible. Six source training examples were
visually checked after resizing; tests also cover rectangular, grayscale, and
transparent images. This checks implementation, not which preprocessing gives
the best accuracy. There is no random augmentation in this initial comparison.
[Reference model preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).

## What learns, and how the comparison stays fair

Both models start from the same pinned MobileNetV3-Small `IMAGENET1K_V1` weights.
The entire original classifier is replaced with one linear layer taking 576
pooled features and producing scores for our 22 categories. This leaves 939,702
parameters in total; the new category layer contains 12,694.

The models use matching initial weights, image lists, per-epoch image order,
preprocessing, and epoch limits. Batch-normalization running statistics stay
fixed in both. During fine-tuning, its learned scale and offset can change
alongside the other feature weights. Checks stop training if the frozen weights
or running statistics change unexpectedly.

Initial settings are batch size 32, seed 551, float32 precision, and AdamW with
weight decay 0.0001. The category layer's learning rate is 0.001; the earlier
layers use 0.0001 when unfrozen. One epoch warms up the category layer. The
optimizer is reset after warmup in both versions to keep that boundary matched.
These are starting settings, not proven optimal settings.

The saved best model has the lowest **source validation loss after warmup**.
Loss measures prediction error; lower is better. Warmup checkpoints are excluded
from selection in both versions. Early stopping ends a run after three eligible
epochs without improvement. The runner never opens source-test or target-test
manifests, and it rejects target images in its training/validation inputs.
Source validation may guide later settings; PlantDoc test scores must not.

## What the November 13 trial showed

Both models ran on the RTX 4060 Laptop GPU using 704 training images and 176
validation images across all 22 categories. Each completed two epochs.

| Check | Frozen features | Fine-tuned features |
| --- | ---: | ---: |
| Trainable parameters after warmup | 12,694 | 939,702 |
| Peak PyTorch allocated GPU memory | 148 MiB | 579 MiB |
| Peak PyTorch reserved GPU memory | 200 MiB | 712 MiB |
| Training + validation loop time, both epochs | 5.0 seconds | 3.8 seconds |

These memory figures exclude CUDA context and other applications. The timings
exclude setup and checkpoint writing; the first model also encountered initial
GPU startup costs. They are **not a speed ranking** or a prediction of full-run
duration. The full-source run described below now checks sustained training on all
selected source images; these earlier trial timings remain a separate record.

The two models had identical warmup metrics. Both changed their category-layer
weights and reduced loss. Only the fine-tuned model changed its feature weights;
both preserved batch-normalization statistics. Both saved checkpoints reloaded
successfully and produced 22 finite scores on a source validation image.

[Recorded trial measurements](feasibility-trial.json) include the small-sample
metrics and code/data fingerprints. These metrics are debugging evidence, not
final accuracy results or proof that one model generalizes better. All 19 unit
tests passed, including checks for image preparation, frozen/fine-tuned updates,
changed image files, and training/validation overlap.

## Saved files and the next step

Each ignored run folder contains:

- `run.json`: settings, software versions, category order, and fingerprints.
- `selected_train.csv` and `selected_validation.csv`: the exact images used.
- `frozen/` and `finetune/`: epoch history, a result summary, and `best.pt`/`last.pt`.
- `comparison.json`: results for the requested modes.

Checkpoints contain model and optimizer state, but automatic resume is not yet
implemented. Keep these files locally; do not commit model weights or personal
dataset paths. The small public trial summary omits local absolute paths.

The first full experiment used all 18,023 training and 4,529 validation images.
Its command was:

```bash
python -m plant_disease.train --output-dir outputs/runs/source-seed-551
```

Without `--trial`, the default limit is ten epochs per model, including warmup.
`--epochs`, `--batch-size`, and `--seed` allow recorded changes. Deterministic
algorithms are requested, but identical results across hardware/software versions
are not guaranteed. The completed pair took about 11.1 minutes in training
and validation loops: 6.1 minutes frozen and 5.0 minutes fine-tuned. It selected
epochs nine and five respectively; fine-tuning stopped after epoch eight.
Both saved models were reloaded and their metrics reproduced on all 4,529
validation images. The selected validation accuracies were 97.57% and 99.36%.

The other two pairs have now completed with unchanged settings. Their loop
times and selected checkpoints are in the combined report. Across all three
pairs, mean validation accuracy is 97.54% for frozen features and 98.98% for
fine-tuning. Fine-tuning scored higher in every pair, but its accuracy varied
more across seeds (sample standard deviation 0.38 versus 0.18 percentage points).
This is a descriptive validation comparison, not a significance test or final
generalization result. Final test evaluation and the
[IEEE report](phase4-report/Phase-4-Plant-Disease-Classification.md) are complete.

## Read a completed full-source run

After both versions finish, export a table and learning curves with:

```bash
python -m plant_disease.summarize_training --run-dir outputs/runs/source-seed-551 --output-dir outputs/reports/reproduction-01
```

This example creates Markdown, JSON, and a PNG under ignored
`outputs/reports/reproduction-01/`. Review new evidence before publishing it. The exporter reads saved measurements and
fingerprints the checkpoints; it does not open images or test lists. It rejects
an incomplete comparison, a trial run, or a changed selected-image list.

The table shows the validation metrics from the selected checkpoint. The last
epoch may be worse, and the highest-accuracy epoch is not necessarily the one
with lowest validation loss. Keeping that distinction makes the report describe
the actual model selected for evaluation.

## Repetition budget

Use three matched pairs with training seeds **551, 552, and 553**, keeping the
existing split manifests fixed. A training seed changes the new category layer's
initial values and the order images are shown; it does not rebuild our splits.
All three pairs are complete. Frozen runs each finished ten epochs, selecting
epochs 9/10/10 for seeds 551/552/553. Fine-tuned runs finished 8/7/6 epochs,
selecting 5/4/3. Settings and stopping rules were retained throughout.

Keep the current ten-epoch limit, warmup, early-stopping rule, preprocessing,
batch size, and learning rates the same for every pair. Report all three seeds
instead of selecting whichever produces the highest score. This gives a modest
check on variation between runs; three repetitions do not establish statistical
certainty or compensate for the small target test set.

The [selected-checkpoint record](training-runs/selected-checkpoints.json) now
identifies every chosen model and its fingerprint before final evaluation.
All six have now been evaluated on the same reserved source and target test
manifests, with no further training or seed selection. See the final evaluation
guide for those measurements. Preserve the fixed settings for this experiment;
do not choose seeds or tune the reported models using the test scores.


## Reproduce the remaining repetitions and combined report

These commands keep the split manifests fixed. Run each training command once;
choose a fresh folder if rerunning an experiment rather than overwrite a record.

```bash
python -m plant_disease.train --seed 552 --output-dir outputs/runs/source-seed-552
python -m plant_disease.train --seed 553 --output-dir outputs/runs/source-seed-553
```

For each seed (551, 552, 553), verify the selected checkpoints and export its
individual report. For example:

```bash
python -m plant_disease.verify_training --run-dir outputs/runs/source-seed-552
python -m plant_disease.summarize_training --run-dir outputs/runs/source-seed-552 --output-dir outputs/reports/reproduction-01
```

The verifier reopens both saved models and checks all source validation images
using the recorded settings. It stops if checkpoint details, image lists, core
training code, expected frozen weights, or validation metrics do not match.
Use `--data-root` if your local datasets are elsewhere. The pinned pretrained
weights should be in that root's `weights/` folder. Verification uses the same
software/device settings as training; it is not a cross-hardware equivalence test.

After all three pairs pass verification:

```bash
python -m plant_disease.summarize_repetitions --output-dir outputs/reports/reproduction-01
```

This example writes the combined Markdown/JSON/PNG report under
`outputs/reports/reproduction-01/`.
It compares all three seeds, reports their mean and sample standard deviation,
and rejects changed training settings or image lists. It also writes
`selected-checkpoints.json`, recording all six selected models and their file
fingerprints before final testing. It does not pick a winning seed, average
weights, or combine predictions into an ensemble. No final test images are
opened by these reporting or verification commands.
