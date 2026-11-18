# CECS-551-Plant-Disease-Classification

A semester project exploring plant disease classification across PlantVillage and PlantDoc images.

Before starting team or AI-assisted work, read [AGENTS.md](AGENTS.md) for the
existing report, project locations, and guidance on follow-up changes.

## Current focus

Phase 4: compare frozen-feature and fine-tuned MobileNetV3-Small models.
Training and final testing are complete. See the [final test results](docs/evaluation/final-results.md):
accuracy was high on PlantVillage and much lower on PlantDoc. The submitted
Phase 4 IEEE report is available as a [PDF](docs/phase4-report/Phase-4-Plant-Disease-Classification.pdf)
and [readable Markdown](docs/phase4-report/Phase-4-Plant-Disease-Classification.md).
It describes the original experiment at commit `9619975`; later optional
experiments are not part of this submission. The [evaluation guide](docs/evaluation.md)
explains how to reproduce the tests and regenerate tables and plots.

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

Optional augmentation and PlantDoc-training experiments are described in the
[follow-up guide](docs/optional-experiments.md). They use separate records and
do not replace the original results.
