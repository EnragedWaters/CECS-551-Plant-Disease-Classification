# Team and coding-assistant guide

## Start here: the report and findings already exist

As of November 17, 2026, the original Phase 4 experiment is complete: data
preparation, three training seeds, checkpoint verification, and final testing.
**A first draft of the IEEE paper has already been submitted.** It includes the
introduction, literature review, methods, measured results, discussion, and
conclusion. The team can review it and propose improvements; there is no need
to start a second paper or invent findings to fill it.

The Phase 4 presentation and visual demo are now prepared; see
[the presentation guide](docs/presentation/README.md) before creating another
deck or demo. Confirm the class schedule and rehearse the existing materials.

Read these before planning work:

1. [Existing IEEE paper — Markdown](docs/phase4-report/Phase-4-Plant-Disease-Classification.md)
2. [Submitted IEEE paper — PDF](docs/phase4-report/Phase-4-Plant-Disease-Classification.pdf)
3. [Measured test results and limitations](docs/evaluation/final-results.md)
4. [Project overview](README.md)

For a broad request such as “complete the project,” first compare the existing
work with the specific requested deliverable or supplied rubric. Identify what
is actually missing before changing anything. Do not assume the paper, training,
or evaluation still needs to be created. If the request has no concrete gap,
explain the current status and establish which improvement is wanted.

## Where things are

| Location | What it contains |
| --- | --- |
| [docs/presentation/](docs/presentation/) | Phase 4 PowerPoint/PDF, speaker notes, demo instructions and rubric coverage. Start here for presentation work. |
| [demo/](demo/) and [plant_disease/demo.py](plant_disease/demo.py) | Local visual demonstration of verified original saved models. Image-bearing replay/backup files remain in ignored outputs. |
| [docs/reproduction.md](docs/reproduction.md) | Exact commands for the paper’s original code version and a separate fresh reproduction. |
| [docs/phase4-report/](docs/phase4-report/) | The existing IEEE paper in Markdown and PDF. The PDF preserves the first submission. |
| [docs/evaluation/](docs/evaluation/) | Final results, JSON evidence, per-category CSV scores, and plots. Use these to check numerical claims. |
| [docs/training-runs/](docs/training-runs/) | Individual seed reports, the combined training report, and selected-checkpoint records. These validation scores are separate from final test scores. |
| [configs/categories.json](configs/categories.json) | The 22 shared categories and their dataset names. |
| [docs/category-mapping.md](docs/category-mapping.md) | Mapping decisions and uncertain categories excluded from the experiment. |
| [docs/experiment-data.md](docs/experiment-data.md) | Training, validation, and test lists; grouping and exclusion rules. |
| [docs/image-analysis.md](docs/image-analysis.md) | Image inspection findings and examples. |
| [docs/setup.md](docs/setup.md) and [docs/data-downloads.md](docs/data-downloads.md) | Environment setup, dataset sources, and local data preparation. |
| [docs/training.md](docs/training.md) and [docs/evaluation.md](docs/evaluation.md) | Commands, saved outputs, checkpoint checks, and evaluation procedure. |
| [docs/optional-experiments.md](docs/optional-experiments.md) | Separate augmentation and PlantDoc-training experiments; these are optional follow-ups. |
| [plant_disease/](plant_disease/) | Model, image loading, training, verification, evaluation, and report-generation code. |
| [scripts/](scripts/) | Setup/data preparation and inspection scripts, plus the earlier dataset-analysis notebooks. |
| [tests/](tests/) | Automated checks, including small synthetic training/evaluation tests. |

`outputs/` is Git-ignored local storage for generated image lists, checkpoints,
predictions, and new experiment reports. Raw datasets and downloaded weights
are not included in the repository. Their absence in a fresh clone does not
mean the experiment was never completed. Use the setup/data guides and each
person's local paths; do not hardcode another teammate's home directory.

Some documents and notebooks describe earlier plans. For what was actually
measured, use the paper's identified code version and saved result records.
Private course material, team evaluations, and any local `AGENTS.local.md` or
`WALKTHROUGH.md` are not team deliverables and must remain out of Git. A clone
does not need these private files to understand or work on the project.

## Keep the original experiment distinct from follow-up work

The paper describes code version `9619975`: MobileNetV3-Small, frozen features
versus fine-tuning, PlantVillage-only training, no random augmentation, and
seeds 551, 552, and 553. It tests on PlantVillage and cropped PlantDoc. Strong
PlantVillage accuracy did not carry over to PlantDoc; this is a measured finding,
not a missing result that needs to be replaced with a better-looking score.

Later code supports optional augmentation and optional PlantDoc training.
**Having those options implemented does not mean those experiments were run or
that they improved accuracy.** Synthetic test scores are software checks, not
plant-disease research results.

- Keep the submitted PDF and original experiment evidence intact. Propose paper
  revisions separately; an updated Markdown draft does not update the submitted
  PDF or Canvas submission automatically.
- Support numerical claims with saved run records. Distinguish proposed changes,
  implementation checks, measured findings, and another paper's reported results.
- Use separate output folders for new experiments and record the actual settings,
  datasets, splits, and seeds. Do not overwrite original results or mix unlike runs.
- Keep test images out of training. The original test scores have already been
  viewed; tuning against them does not produce a fresh unseen final test.
- PlantDoc training changes the source-only research question and can change the
  validation population. Explain that difference before comparing results.
- Existing checkpoints require their original code to pass fingerprint checks.
  Do not disable those checks or rewrite old metadata to make changed code pass.
- Cross-dataset results do not isolate weather or photography skill as causes.
  Keep the scope as a classification study; no app or diagnostic service is required.

## Making a useful contribution

Start with the existing files. Useful tasks include checking the introduction
against the research question, verifying citations and results, clarifying an
explanation, checking setup instructions, or proposing a clearly scoped follow-up.
A new finding needs reproducible evidence; a report edit can improve clarity
without claiming a new experiment.

Check `git status` and the diff before editing. Preserve other people's work,
keep changes focused, and use branches/pull requests for team review. Follow the
user's instructions about staging, committing, publishing, and merging; a broad
request to finish the project does not authorize replacing the submission.
Use plain-English commit titles that explain the change.

For code changes, use the project environment and run relevant checks. The full
suite runs from the repository root with:

```bash
python -m unittest discover -s tests -v
```

The tests use invented data and small models; they do not require downloading
the real image datasets. Do not launch full training or large downloads merely
to review the report. Explain code changes with useful comments and report what
was checked, what changed, and any remaining limitations.
