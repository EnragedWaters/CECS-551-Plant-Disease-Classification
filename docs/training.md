# Training the two model versions

The training code, small GPU trial, and first full-source training pair are
complete. [The first results](training-runs/source-seed-551.md) include validation
scores and learning curves. Repeated runs and final test evaluation remain;
validation scores do not establish performance on unfamiliar datasets.

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

This cost supports the three-pair budget below. Reserve roughly 30 more minutes
for two additional pairs, with extra time if both reach the epoch limit or the
laptop slows down. This is an estimate, not a guaranteed runtime. Final test
evaluation, repeated-run analysis, and the Phase 4 report remain to do.

## Read a completed full-source run

After both versions finish, export a table and learning curves with:

```bash
python -m plant_disease.summarize_training --run-dir outputs/runs/source-seed-551
```

This creates Markdown, JSON, and a PNG under `docs/training-runs/`. Those small
results files can be committed. The exporter reads saved measurements and
fingerprints the checkpoints; it does not open images or test lists. It rejects
an incomplete comparison, a trial run, or a changed selected-image list.

The table shows the validation metrics from the selected checkpoint. The last
epoch may be worse, and the highest-accuracy epoch is not necessarily the one
with lowest validation loss. Keeping that distinction makes the report describe
the actual model we will later evaluate.

## Repetition budget

Use three matched pairs with training seeds **551, 552, and 553**, keeping the
existing split manifests fixed. A training seed changes the new category layer's
initial values and the order images are shown; it does not rebuild our splits.
The first pair uses 551. The other two pairs are planned, not completed here.

Keep the current ten-epoch limit, warmup, early-stopping rule, preprocessing,
batch size, and learning rates the same for every pair. Report all three seeds
instead of selecting whichever produces the highest score. This gives a modest
check on variation between runs; three repetitions do not establish statistical
certainty or compensate for the small target test set.

Before final evaluation, record the chosen checkpoint for every model and seed.
Then evaluate each once on the same reserved source and target test manifests.
Do not adjust training settings or choose seeds using those test results.
