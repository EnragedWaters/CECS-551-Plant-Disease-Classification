# Data for the initial Phase 4 experiment

We use 22 shared plant/condition categories: ten healthy categories and twelve
named conditions supported by the [category review](category-mapping.md).
The exact labels and their output order are in
[configs/categories.json](../configs/categories.json).

Five pairs remain outside this experiment: apple rust, corn rust, corn leaf
blight, bell-pepper leaf spot, and tomato yellow virus. Their labels are too
uncertain to assume a specific match. This decision was made before training
or seeing any model scores. The selected labels are dataset annotations, not
independently verified diagnoses.

## Which images are used?

The November 13 preparation run produced:

| Image list | Count | Purpose |
| --- | ---: | --- |
| `train.csv` | 18,023 | Fit the model using PlantVillage images |
| `validation.csv` | 4,529 | Choose settings and checkpoints using PlantVillage only |
| `source_test.csv` | 5,487 | Final test on reserved PlantVillage images |
| `target_test.csv` | 183 | Final test on eligible existing PlantDoc test images |

Every list contains all 22 categories. Both model versions must use these same
lists and category order. PlantDoc training images do not fit the model or
select settings. The external test measures this filtered subset, not the
whole original PlantDoc dataset or general field performance.
Only 3–12 PlantDoc test images remain per category, so individual-category
scores will be sensitive to a few mistakes. Report those counts with results.

## How the split is made

1. Read the full image audit, including unused categories and PlantDoc training
   images, so they can still expose conflicting labels on a test image.
2. Join files with identical bytes or the same available leaf identifier into
   connected groups. This keeps known related photographs together.
3. Exclude groups with conflicting labels; do not guess the correct diagnosis.
   Also exclude unreadable images, unsupported categories, and PlantDoc training
   images. Every excluded path receives a reason in `excluded.csv`.
4. Preserve existing test members of a group and exclude its training members.
   If a group spans both test datasets, retain the PlantDoc test members and
   exclude the source copies. Never move a training picture into a test set.
5. Retain one deterministic copy of byte-identical images within a selected
   group. Different photos of the same known leaf stay together but are not
   automatically discarded.
6. Reserve approximately 20% of each remaining source category for validation,
   using entire groups and seed **551**. Group sizes prevent exact percentages.

The script verifies category coverage, prevents known groups/hashes crossing
splits, and requires every input path to be selected or excluded exactly once.
It stops if any chosen category loses all examples in a required split.

Most of the 28,661 unused images are outside the chosen categories or belong
to PlantDoc training. The audit also identified 18 files in conflicting-label
groups, five training images related to reserved tests, and 16 redundant byte
copies. These are exclusive bookkeeping reasons, not a count of bad images.
Raw files are never changed, copied, or deleted by this preparation step.

## Reproduce the lists

Activate the environment described in [setup.md](setup.md), prepare the pinned
datasets using [data-downloads.md](data-downloads.md), then run:

```bash
python scripts/analyze_data.py
python scripts/build_splits.py
python -m unittest discover -s tests -v
```

If the output folder already contains files, choose a fresh `--output-dir`;
existing split files are not overwritten. Optional PlantDoc training requires
its own folder and explicit flags; see [optional experiments](optional-experiments.md).
The counts above remain the original submitted experiment's counts.

For a different data location, pass `--data-root` to the audit script. Manifest
paths remain relative to that root, so personal home-directory paths are not
embedded in them. `build_splits.py` accepts `--audit-dir`, `--categories`,
`--output-dir`, `--seed`, and `--validation-fraction`; changing these changes
the experiment and must be documented before model selection/testing.

Generated CSVs and `summary.json` are in ignored `outputs/splits/`. The summary
records per-class counts, source versions, and fingerprints of the category
configuration, builder, inventory, and CSV files. The small
[recorded split summary](split-summary.json) lets others compare their generated
files without committing datasets or large inventories.

## Remaining limits

Exact byte matching does not catch every re-encoded or cropped copy. Leaf IDs
are missing for some source images, and PlantDoc has no equivalent grouping
metadata in this audit. Known-group separation therefore does not prove that
every test image is independent of every training image. Any later confirmed
overlap requires a documented preparation change and regenerated lists before
experiments. Likewise, excluding known label conflicts does not verify all
remaining disease labels. We keep valid difficult photos and do not use test
predictions to decide which pictures or categories to remove.
