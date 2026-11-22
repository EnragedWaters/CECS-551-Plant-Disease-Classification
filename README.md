# Plant Disease Classification Across Image Datasets

A semester project comparing frozen and fine-tuned models on PlantVillage and
PlantDoc. The experiment, IEEE paper draft, presentation, and visual demo are prepared.

## Run the demo

Open **[ONE-CLICK-LAUNCHERS](ONE-CLICK-LAUNCHERS/README.md)** and choose the file
for Windows, macOS, or Linux/WSL. Keep the terminal open while using the demo.

Choose a photograph and click **Run & show all stages** for a scrollable comparison
of both models. The second tab runs a 22-image sample from each dataset; those
counts are separate from the published full evaluation. The summary explains
what each stage changes, with optional filter, averaging and score lessons.

A new computer needs the Python environment, datasets, and saved models first.
See [setup and demo instructions](docs/presentation/demo-guide.md), including the
recorded offline backup. WSL is optional on Windows.

## Open the slides and notes

[Presentation guide](docs/presentation/README.md) — editable PowerPoint, PDF,
speaker notes, real dataset photographs, visual comparisons and suggested
four-person speaking portions.

## Read the paper and results

- [IEEE paper — readable Markdown](docs/phase4-report/Phase-4-Plant-Disease-Classification.md)
- [IEEE paper — PDF](docs/phase4-report/Phase-4-Plant-Disease-Classification.pdf)
- [Full evaluation results](docs/evaluation/final-results.md)

The original experiment is complete. Accuracy was high on PlantVillage and much
lower on PlantDoc. The paper describes code version `9619975`; optional later
experiments do not replace those findings.

## Working on the project

Use a branch and open a draft pull request early so work is visible.
Changes reach `main` through a pull request after Windows and Linux checks pass.
Normal commits to your branch are unchanged.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, local checks and fixes
for rejected pushes or failed checks.

Read [AGENTS.md](AGENTS.md) before team or AI-assisted changes. It maps the
existing files and explains what must be preserved.
[Reproduction instructions](docs/reproduction.md) cover the original experiment;
[optional experiments](docs/optional-experiments.md) are separate follow-up work.
The [repository audit](docs/repository-audit.md) records maintenance fixes and
what was deliberately preserved.

## Submission files

[Phase-4-Submission](Phase-4-Submission/README.md) contains the repository-link TXT, IEEE paper, slides and speaker notes.
