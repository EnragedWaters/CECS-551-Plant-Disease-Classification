# Team and coding-assistant guide

## Start here: the report and findings already exist

As of November 17, 2026, the original Phase 4 experiment is complete: data
preparation, three training seeds, checkpoint verification, and final testing.
**A first draft of the IEEE paper has already been submitted.** It includes the
introduction, literature review, methods, measured results, discussion, and
conclusion. The team can review it and propose improvements; there is no need
to start a second paper or invent findings to fill it.

For presenting, start in [ONE-CLICK-LAUNCHERS](ONE-CLICK-LAUNCHERS/README.md) and
choose the OS-labeled launcher. Windows supports native Python with optional WSL;
macOS and Linux/WSL have separate launchers. The demo defaults to all seven
processing stages together and also offers a running two-dataset sample.
Keep the default view visual: one short caption per stage, photos/maps/diagrams
first, and explanations/calculations behind optional details. Do not repeat
paragraphs beneath both models or expand every lesson by default.
`demo/lessons.js` adds stage purposes, preprocessing geometry, a separate toy
filter, actual pooling arithmetic and classifier contributions. The toy must
remain clearly separate from the real model. `demo_trace.py` verifies the real
numerical reconstructions without changing predictions. Keep these controls
working in the self-contained offline export too.
Use `--no-open-browser` for launcher/server checks and headless browsers for UI
verification; do not take window focus during automated work.
Keep launcher helpers in `scripts/`; do not recreate launchers at the repo root.
Personal `demo-settings.local.json` stays at the root and is Git-ignored.

The Phase 4 presentation and visual demo are now prepared; see
[the presentation guide](docs/presentation/README.md) before creating another
deck or demo. Confirm the class schedule and rehearse the existing materials.
The slides now use real photos, diagrams and editable result bars. Keep visible
text brief and explanations in the notes. `docs/presentation/slides.json` and
`scripts/build_presentation.py` are the sources; `assets/manifest.json` and
`PHOTO_CREDITS.md` record the small licensed presentation illustrations. Do not
replace them with generated photos or treat selected examples as full-test scores.
Rebuild with `--office libreoffice`, inspect all rendered slides, and refresh
`Phase-4-Submission/presentation/` after changes. The evidence checker is
`scripts/check_presentation.py`; it requires the original local data and outputs.

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
| [Phase-4-Submission/](Phase-4-Submission/README.md) | Ready-to-use repository-link TXT, IEEE paper and presentation files. Keep copies synchronized with their sources in docs/. |
| [ONE-CLICK-LAUNCHERS/](ONE-CLICK-LAUNCHERS/README.md) | Presenter entry point: choose your OS and start the existing demo. |
| [docs/presentation/](docs/presentation/) | Phase 4 PowerPoint/PDF, speaker notes, demo instructions and rubric coverage. Start here for presentation work. |
| [demo/](demo/) and [plant_disease/demo.py](plant_disease/demo.py) | Local demonstration of both datasets, actual intermediate features, and a fixed teaching sample using verified original models. Image-bearing replay/backup files remain in ignored outputs. |
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

## Required checks and visible contributions

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing the shared branch.
Use a feature branch, push progress, and open a draft pull request early. Shared
repository branches are preferred; forks are allowed and face the same merge
checks. Do not assume a commit or pushed branch means the work is on main.

Run `python scripts/check_project.py` with the documented test environment.
GitHub requires the Windows and Linux jobs to pass against the latest main.
Ordinary local commits and feature-branch pushes remain available; main rejects
direct pushes, force pushes and deletion. No local commit-blocking hook is needed.
If someone asks “why won't it commit?”, distinguish a local Git/editor error,
a protected-main push rejection, and a failed pull-request check. Read the actual
error and failed job log before recommending a fix.

Fix the underlying problem. Do not disable tests, swallow failures, remove
required jobs, weaken rules or bypass checks to merge. For a necessary change
to automation, explain it in the pull request, verify the guard still detects
failures, and ask the owner to review. Keep required check names stable. This
review convention does not impose a reviewer gate on every owner-authored PR.
The owner retains settings access for deliberate recovery, not routine bypass.

Use headless browsers and invented CPU fixtures for CI. Do not require private
datasets, research weights, GPU hardware or research training. Never treat fixture
scores as findings. Checkers catch common accidental local-file additions, not
all secrets or scientific errors. Preserve the original evidence and make any
new experiment or paper revision explicitly separate.
