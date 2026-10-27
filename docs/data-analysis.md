# Phase 3: preliminary image analysis

This is Max's reproducible data-inspection workflow. It supports the rubric's
preliminary EDA and preprocessing-plan criteria. It is **not** model training or
a statement of results before the notebook has been run on the actual images.
The analysis preserves source category names without assuming that PlantVillage
and PlantDoc categories are interchangeable; Jacob's category matching remains
separate.

## Reproduce the audit

Activate the environment in [setup.md](setup.md), then run from the repository
root after the data preparation documented in [data-downloads.md](data-downloads.md):

```bash
python scripts/analyze_data.py
```

Defaults are `~/datasets/cecs551-plant-disease` for data, `outputs/eda/` for
results, **four uniformly selected files per dataset/category** for the visual
sample, and random seed `551`. Override without changing the raw files:

```bash
python scripts/analyze_data.py \
  --data-root /path/to/cecs551-plant-disease \
  --output-dir outputs/eda \
  --samples-per-category 4 --seed 551
```

Then open [`notebooks/data_analysis.ipynb`](../notebooks/data_analysis.ipynb)
using the configured Python kernel. Its first executable cell repeats the
analysis, so setting `DATA_ROOT` in that cell is necessary on other computers.
It displays the counts, visualizes separate source-category distributions,
illustrates sample images, and lists cases for manual review. The notebook
writes PNG charts under `outputs/eda/` when executed. The input images are
never resized, filtered, labeled, or moved on disk by this workflow.

### Generated files (intentionally ignored by Git)

| File | Meaning |
| --- | --- |
| `outputs/eda/image_inventory.csv` | Each discovered supported image, original category/split, relative file path, resolved leaf ID if available, size, SHA-256, readability, and measurements |
| `outputs/eda/class_counts.csv` | Measured source category/split file counts, including separately identified unreadable files |
| `outputs/eda/image_dimensions.csv` | Original dimensions and mode for successfully decoded files |
| `outputs/eda/sample_manifest.csv` | Exact filenames of reproducible class-stratified samples; edge-variance blur *proxy* on the decodable sample only |
| `outputs/eda/quality_flags.csv` | Unreadable images, split-list mismatches, exact repeated file bytes, and related leaves across source splits |
| `outputs/eda/summary.json` | Scan timestamp, source revisions from the local download manifest (or `unknown` if missing), scan totals, sample method/seed, and issue counts |
| `outputs/eda/*.png` | Notebook-generated class-balance figures and example grid |

Dataset counts in this repository's [download manifest](download-manifest.json)
are from October 23, 2026 and are **not** a substitute for the analyzer's
measured, category-level inventory. The analyzer cannot run without the actual
extracted image directories and PlantVillage's author-provided split/leaf-map
metadata. This repository does not track or embed the raw datasets.

## Interpretation and suggested preprocessing

The analysis is deliberately observational. Record counts **and** the number
and sampling method of visually inspected photos. Before selecting a
preprocessing pipeline, review the notebook evidence and document:

| Possible finding | Appropriate follow-up (only if supported) |
| --- | --- |
| Unreadable image files | List paths, confirm failures, and exclude only those files; preserve the raw source and audit logs |
| Different sizes or aspect ratios | Standardize model inputs and normalization consistently; check whether any crop removes disease symptoms or leaf context |
| Exact duplicates or same-leaf IDs spanning a partition | Investigate potential leakage, then keep related photos together when defining the final source training/validation/test split |
| Visually confirmed blur, uneven lighting, occlusion, or clutter | Describe their prevalence in the inspected sample; keep valid challenging examples for external evaluation and consider justified *training-only* augmentations |
| Class imbalance | Report class counts and plan macro-averaged metrics alongside accuracy, rather than claiming balanced data |

### Scope and important limitations

- **Source partitions are not final experiment partitions.** The analyzer
  records PlantVillage's supplied `color_train.txt`/`color_test.txt` assignments
  and PlantDoc's original train/test folders; it never reshuffles these.
- PlantVillage's `leaf-map.json` is interpreted using the author dataset
  loader's lookup/disambiguation logic. Unknown or ambiguous IDs remain blank,
  not guessed from filenames. A leaf group spanning both source splits is a
  *warning*, not an automatic relabeling of either dataset.
- SHA-256 proves identical **file bytes**, not the same photograph after
  recompression, cropping, or recoloring. A hash shared by different partitions
  is flagged for manual review. Images of the same leaf that are visually
  different may evade that check.
- Edge variance of a grayscale thumbnail is a **relative sharpness proxy**,
  recorded on the selected sample only. It cannot classify blur reliably, and
  background complexity can increase the score. Background clutter, damaged
  crops, wrong labels, and lighting must be checked visually.
- Missing disease labels and confirmed category equivalences cannot be inferred
  from image pixels alone. Images that are challenging but valid should not be
  automatically discarded. PlantDoc test images remain for external evaluation
  and must not guide model hyperparameters or the final training pipeline.
- Review figures saved under `outputs/eda/` before copying selected ones to
  `docs/figures/` for the combined report. Include source attribution if
  reproducing dataset images, following their source licenses.

## Validation without the full datasets

The test suite creates a handful of synthetic images, a corrupt file, known
identical images, and mocked PlantVillage metadata. It verifies counts, split
membership, same-leaf flags, duplicate detection, missing paths, and stable
sampling without downloading any dataset:

```bash
python -m unittest discover -s tests -v
```

**Report status:** the implementation and synthetic tests can be validated in
this repository; actual class-balance charts, inspection observations, and
preprocessing recommendations must be completed by executing the notebook on
Troy's prepared local image files. Do not report synthetic test results as
measurements of PlantVillage or PlantDoc.
