# CECS-551-Plant-Disease-Classification

A semester project exploring plant disease classification across PlantVillage and PlantDoc images.

## Phase 3 focus

Troy's model/report portion:

- Explain the choice of MobileNetV3-Small and the two planned training approaches.
- Outline the computing resources and what still needs checking.
- Combine the team's sections into the Phase 3 report when available.

Model implementation and training come later.

The proposed comparison and computing checks are in [the model plan](docs/model-plan.md).
The [setup guide](docs/setup.md) covers the Python environment, GPU check, and
when dataset downloads are needed.

## Phase 3 data analysis

Max's preliminary EDA uses the prepared, unchanged PlantVillage color and
cropped PlantDoc images. Once the data have been prepared, run:

```bash
python scripts/analyze_data.py
```

The script inventories original categories and splits, checks readability and
image sizes, flags exact duplicates and possible related-leaf leakage, and
records seeded image samples in `outputs/eda/` (ignored by Git). Open
[`notebooks/data_analysis.ipynb`](notebooks/data_analysis.ipynb) for class-balance
charts, captioned image grids, manual review, and report-ready notes. See the
[data-analysis guide](docs/data-analysis.md) for details and limitations.
No training is performed.
