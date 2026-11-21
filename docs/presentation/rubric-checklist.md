# Updated Phase 4 rubric coverage

Reviewed November 20, 2026 against the supplied
`updated_CECS551_Phase4_Rubric.pdf` (two pages, 50 points).
The PDF stays with local course material; it is not needed to run the repository.
This is a coverage review, not a grade prediction or a claim that the live
presentation has already happened.

| Rubric item | Points | Evidence and status |
| --- | ---: | --- |
| Demonstration | 5 | Local visual demo runs both original saved models, displays both datasets, actual feature maps and predictions. Browser checks cover seven stages for six examples and the 22-pair running sample; recorded HTML/PDF backups are prepared. Still needs delivery in class. |
| Organization and clarity | 4 | Ten main slides cover problem → data → method → demo → findings → limits. Two backup slides cover reproduction and selected references. |
| Technical explanation | 3 | Speaker notes explain pretrained/frozen/fine-tuned models, splits, category mapping, metrics, repeated seeds and limitations. Team still needs to practice answering questions. |
| Delivery and team participation | 3 | Suggested speaking portions for four people are provided. Agreement, rehearsal and balanced live participation remain team actions. |
| Complete and correct code | 5 | Existing pipeline plus verified-checkpoint demo; 72 automated tests passed. Original experiment code is identified as 9619975. No new training was needed. |
| README and reproduction steps | 4 | README links setup, data access, demo and exact original-code reproduction commands with expected outputs. Local saved files and fresh-clone prerequisites are distinguished. |
| Organization and documentation | 3 | Shared AGENTS.md and presentation index identify the existing paper, results, code, demo and notes. New code has purpose/usage comments. |
| Results and project history | 3 | Original tables, plots, seed records and useful commit history are preserved. Models, data, image-bearing backups, private notes and evaluations stay out of Git. |
| Problem and related work | 4 | Existing paper's Introduction/Literature Review explains casual-user motivation, prior work and the limited empirical contribution. |
| Methods and reproducibility | 5 | Existing Methods covers pinned data, preprocessing, split rules, model design, training settings and evaluation. New reproduction guide makes the commands easier to follow. |
| Results and discussion | 4 | Existing paper reports both approaches, all seeds, both test sets, plots, error patterns and limitations. Frozen features provide the baseline. |
| IEEE format and citations | 5 | Existing six-page PDF uses IEEEtran conference layout, authors, abstract, index terms, numbered sections, tables/figures and eleven numbered references. Confirm any separately assigned template/author-detail rule. |
| Writing and visual quality | 2 | Original PDF and presentation layouts reviewed; slide notes define the main terms. Final Google Slides import/projector check and team proofreading remain. |

## Important changes from the older PDFs

The older PDFs describe repository and report components with separate course
weights. This updated PDF explicitly assigns **15 points to demo/presentation,
15 to the repository and 20 to the paper**, including balanced team participation.
The paper guidance overlaps much of what was already required. The PDF comparison
alone does not establish when the presentation was first announced verbally.
The new rubric requires demonstrating the project; it does not require an app,
a new architecture, a higher target accuracy, or additional training experiments.

## Dates and details that need confirmation

- The updated PDF prints **December 3, 2026, at 23:59** as the deadline.
- A later class update reported by the team says **December 15** for submission.
- The team discussion mentions **December 3 or 8** for presentations, with
  disagreement about which presentation these dates refer to.
- The updated PDF refers to the **assigned time**, but gives no presentation length.
- Confirm the scheduled slot, presentation length, final upload deadline, and any
  extra slide/demo upload requirement from Canvas or written instructor/TA guidance.
  These were not independently verified against Canvas during this work.

Keep the earlier link-only TXT, IEEE paper and team evaluation requirements
unless the current Canvas instructions explicitly replace them. Team evaluation
ratings are a private human decision; no ratings were changed for this work.
The old ten-page limit is not repeated in the updated PDF, but the existing
six-page paper remains within it anyway.

## What was verified locally

- 72 automated tests, including original checks and demo checks for historical
  code/image integrity, safe routes, replay export, real feature hooks, exact
  logits, unchanged model state, hook cleanup, shared color scales, and stable
  sample selection without prediction-based filtering. New checks cover normalized
  pixel examples, all 576 means, all logits, reconstruction mismatch rejection,
  and browser suppression regardless of argument order.
- Original seed-551 checkpoint and code hashes checked at demo startup.
- Twelve curated predictions and 88 running-sample predictions match the original
  evaluation's predicted categories on CPU. Scores can vary with device/batch size.
- Both datasets, all seven stages together in the default summary, individual-stage
  playback controls, sample counters,
  pause/resume/reset, learning controls and narrow windows checked in a headless
  browser. The filter toy gives 0 / +3 / −3 for uniform / opposite-edge examples;
  all pooling-channel and classifier-category selectors were exercised. Offline replay
  works with HTTP blocked. Image-bearing screenshots and PDF backups remain local.
- OS-labeled launchers are grouped in ONE-CLICK-LAUNCHERS; the main README
  points presenters to the demo, slides and paper. Relocated launchers were
  checked from unrelated working directories and paths containing spaces.
- Native Windows and Windows-to-WSL launcher paths exercised with real saved-model
  inference using `--no-open-browser`. The new flag was forwarded through Windows
  native, Windows-to-WSL, and Linux launchers; shell tests also cover the macOS
  wrapper. No visible browser was launched. Linux/macOS shell dispatch checked under Linux; macOS Finder launch
  remains unverified on an actual Mac. Six launcher tests cover local settings,
  prerequisite errors, argument forwarding and failure handling.
- All 100 demo predicted score vectors, prepared images and feature maps match
  the prior replay exactly. Largest explanatory reconstruction differences on the
  six real images: 4.42e-7 for pooled averages and 4.77e-6 for raw category scores.
  These checks preserve the original findings; no additional training occurred.
- The updated local backup is `outputs/phase4-demo-learning/replay-final/offline-demo.html`;
  screenshots, numerical checks and PDF backups are beside it under the ignored
  `outputs/phase4-demo-learning/` folder.
- PPTX contains twelve slides and speaker notes for all twelve. Text and shapes
  are editable; the chart is an embedded copy of the published result plot.
- Twelve-page PDF companion visually reviewed. Google Slides import and actual
  classroom display have not been observed and still need a quick human check.
- The reproduction exporter was checked against the original saved prediction
  files in an isolated checkout. Original report/results were not overwritten.

## Remaining human steps

Confirm logistics, agree on speaking portions, rehearse the actual demo/handoffs,
check the imported slides, and review Canvas's final upload list. The package is
prepared for those steps; no submissions or messages were sent by the assistant.
