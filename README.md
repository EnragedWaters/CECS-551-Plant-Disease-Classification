# CECS-551-Plant-Disease-Classification

A semester project exploring plant disease classification across PlantVillage and PlantDoc images.

## Current focus

Phase 4: compare frozen-feature and fine-tuned MobileNetV3-Small models.
The first full training comparison is complete. See the
[results and learning curves](docs/training-runs/source-seed-551.md). These are
PlantVillage validation scores; repeated runs and final test evaluation are next.

After following the setup guide and downloading the data, run from this folder:

```bash
python scripts/analyze_data.py
python scripts/build_splits.py
python -m unittest discover -s tests -v
python -m plant_disease.train --trial --output-dir outputs/trials/my-first-trial
```

The [experiment data guide](docs/experiment-data.md) explains the 22 selected
categories, exclusions, and generated training/validation/test lists. Raw images
stay untouched; generated lists are stored locally in `outputs/splits/`.

The [training guide](docs/training.md) explains the two versions, run settings,
saved outputs, and trial findings. The comparison is described in
[the model plan](docs/model-plan.md).
The [setup guide](docs/setup.md) covers the Python environment, GPU check, and
when dataset downloads are needed.
