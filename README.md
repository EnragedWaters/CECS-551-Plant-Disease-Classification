# CECS-551-Plant-Disease-Classification

A semester project exploring plant disease classification across PlantVillage and PlantDoc images.

## Current focus

Phase 4: prepare reproducible image splits, then compare frozen-feature and
fine-tuned MobileNetV3-Small models. Training has not started.

After following the setup guide and downloading the data, run from this folder:

```bash
python scripts/analyze_data.py
python scripts/build_splits.py
python -m unittest discover -s tests -v
```

The [experiment data guide](docs/experiment-data.md) explains the 22 selected
categories, exclusions, and generated training/validation/test lists. Raw images
stay untouched; generated lists are stored locally in `outputs/splits/`.

The proposed comparison and computing checks are in [the model plan](docs/model-plan.md).
The [setup guide](docs/setup.md) covers the Python environment, GPU check, and
when dataset downloads are needed.
