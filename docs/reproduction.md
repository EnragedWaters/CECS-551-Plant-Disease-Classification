# Reproduce the original paper

The submitted paper describes commit **9619975**, before the optional augmentation
and PlantDoc-training changes. Use that original version for the original saved
models or a reproduction of the paper. Keep later experiments in separate folders.

You do not need to reproduce training to read the results or give the presentation.
The [visual demo](presentation/demo-guide.md) uses the existing saved models and
loads their verified original code automatically.

## Create a separate checkout of the original code

Start in the current repository root, with a normal clone containing its history:

```bash
git fetch origin
project_repo="$(pwd)"
git worktree add --detach ../CECS551-original-reproduction 9619975a2f64954bc25d056708962eea1895d84b
cd ../CECS551-original-reproduction
source ~/.venvs/cecs551-plant-disease/bin/activate
python scripts/check_environment.py --require-cuda
```

This leaves your main checkout and branch alone. The sibling folder contains
the old code, with its own ignored outputs. Choose another unused worktree folder
name if that one already exists. If the environment does not exist yet, follow
[setup.md](setup.md) first. The full original workflow uses the documented NVIDIA
GPU environment; the lightweight visual demo defaults to CPU instead.

## Use the original saved experiment on the prepared laptop

These commands assume the original run folders, split lists and prediction files
still exist under the current repository's ignored `outputs/`. The `project_repo`
variable above points to that repository. Set it again if using a new terminal.

To regenerate tables and plots from the already saved predictions, without
running models or opening test images:

```bash
python -m plant_disease.summarize_evaluation \
  --evaluation-dir "$project_repo/outputs/evaluation/final-01" \
  --output-dir outputs/reports/original-predictions-01
```

To deliberately rerun inference on the original models and test lists:

```bash
python -m plant_disease.evaluate \
  --runs-root "$project_repo/outputs/runs" \
  --splits-dir "$project_repo/outputs/splits" \
  --selection "$project_repo/docs/training-runs/selected-checkpoints.json" \
  --output-dir outputs/evaluation/original-recheck-01
python -m plant_disease.summarize_evaluation \
  --evaluation-dir outputs/evaluation/original-recheck-01 \
  --output-dir outputs/reports/original-recheck-01
```

Use a **new output directory** every time. Historical exporters do not all have
current overwrite protections; do not point them at the published `docs/` results
or an earlier run. The evaluator itself refuses an existing destination.
The rerun does not create a fresh unseen test set. It is a reproducibility check.

Dataset and pinned ImageNet-weight files default to
`~/datasets/cecs551-plant-disease`. If needed, add `--data-root /your/data/folder`
to evaluation. It must contain both dataset subfolders and `weights/`.
Expected outputs are twelve prediction CSVs (six models × two test sets), result
JSONs, aggregate Markdown/JSON, a comparison plot, and confusion matrices.

The paper's mean accuracies were 97.59% / 21.68% frozen and 99.07% / 25.87%
fine-tuned (PlantVillage / PlantDoc). Different hardware, kernels or batch sizes
can change floating-point outputs. Checksums and metadata identify the actual
files; do not edit fingerprints to make a different checkpoint pass.

## Fresh reproduction when no saved models exist

Run the following **inside the separate original-code worktree**, starting with
empty output directories. This downloads data if missing and trains new models;
it is not part of presentation startup. Allow roughly 10–15 GiB for data,
archives and outputs, plus the environment. See [data access](data-downloads.md)
for versions, licensing and download sizes.

```bash
python scripts/prepare_data.py
python scripts/analyze_data.py
python scripts/build_splits.py
python -m unittest discover -s tests -v

for seed in 551 552 553; do
  python -m plant_disease.train --seed "$seed" --output-dir "outputs/runs/source-seed-$seed"
  python -m plant_disease.verify_training --run-dir "outputs/runs/source-seed-$seed"
done

python -m plant_disease.summarize_repetitions \
  --runs-root outputs/runs \
  --output-dir outputs/reports/reproduction-selection-01
```

Before testing, preserve the generated `selected-checkpoints.json` as the
selection record for **these new runs**. Do not reuse the published selection
file: newly saved checkpoints can have different file hashes. Verify that the
split counts are 18,023 train, 4,529 validation, 5,487 source test and 183 target
test, with the same 22 categories. Keep all three seeds; do not choose one using
its test performance.

```bash
python -m plant_disease.evaluate \
  --runs-root outputs/runs \
  --splits-dir outputs/splits \
  --selection outputs/reports/reproduction-selection-01/selected-checkpoints.json \
  --output-dir outputs/evaluation/reproduction-01
python -m plant_disease.summarize_evaluation \
  --evaluation-dir outputs/evaluation/reproduction-01 \
  --output-dir outputs/reports/reproduction-final-01
```

Record any differences as a separate reproduction. The original paper, original
checkpoints and published results remain unchanged. Repeated training does not
make the already examined test data new or suitable for tuning.

## Return to current work

```bash
cd "$project_repo"
```

The worktree remains available for later inspection. Do not delete it until its
local outputs are backed up or no longer needed. Current optional experiments
have their own [guide](optional-experiments.md); they are not the paper's method.
