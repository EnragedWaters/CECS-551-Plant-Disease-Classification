# Phase 4 presentation package

The experiment and first IEEE submission are complete. These materials explain
that existing work; they do not add new training results or replace the paper.

| Open this | What it contains |
| --- | --- |
| [Phase-4-Presentation.pptx](Phase-4-Presentation.pptx) | Editable PowerPoint, including speaker notes. Real photos, editable diagrams and editable result bars for Google Slides import. |
| [Phase-4-Presentation.pdf](Phase-4-Presentation.pdf) | Matching PDF companion for presenting or reviewing without PowerPoint. |
| [PHOTO_CREDITS.md](PHOTO_CREDITS.md) | Photo IDs, source links, licenses and a manifest of the selected illustrations. |
| [speaker-notes.md](speaker-notes.md) | Plain-English explanations, demo sequence, suggested four-person handoffs. |
| [demo-guide.md](demo-guide.md) | Start the live visual demo, use the offline backup, and troubleshoot. |
| [rubric-checklist.md](rubric-checklist.md) | Coverage of the updated rubric and remaining instructor-dependent details. |
| [../reproduction.md](../reproduction.md) | Original-code setup and complete reproduction sequence. |
| [../phase4-report/Phase-4-Plant-Disease-Classification.md](../phase4-report/Phase-4-Plant-Disease-Classification.md) | Existing IEEE paper; unchanged. |

## Presentation shape

There are **ten main slides and two backup slides**. The suggested speaking
sequence is about 8 minutes, with extra time for handoffs; allow roughly
9–10 minutes in rehearsal. This is a provisional structure, not an instructor
assigned time limit. Notes also suggest how to shorten it to around five minutes.

| Suggested speaker | Portion |
| --- | --- |
| Jacob | Slides 1–3: problem, motivation and datasets |
| Troy | Slides 4–5: separate image lists and the two model approaches |
| Max | Slides 6–7: visual demo and measured results |
| Shriyansh | Slides 8–10: error pattern, limitations and conclusion |

These are proposed handoffs, not claims that anyone has agreed or rehearsed.
Everyone should understand the research question, the distinction between
training/validation/testing, and why PlantDoc performance is much lower.
The rubric scores preparation, delivery and balanced participation; files alone
cannot fulfill that live requirement.

## Open in Google Slides

Upload `Phase-4-Presentation.pptx` to your Drive and open it with Google Slides.
Check the fonts, chart readability and speaker notes before presenting. The deck
uses standard editable shapes/text with Arial and no macros or unusual effects.
Photographs and recorded feature maps are embedded images; labels, diagrams
and chart bars remain editable. The PDF and Markdown notes are backups if an
import changes layout or omits notes. An actual Google Slides import has **not**
been performed as part of the local checks.

The supplied PDF was exported from the actual PPTX using headless LibreOffice.
All twelve rendered slides were visually reviewed, and all slide text survived
the export. Photo hashes, labels and example predictions were checked against
the original evidence. Check the imported deck on the classroom computer/projector.

## Before presenting

1. Confirm the official date and time allowance; see the checklist's open questions.
2. Agree on speaking portions and rehearse the handoffs once with a timer.
3. Open [ONE-CLICK-LAUNCHERS](../../ONE-CLICK-LAUNCHERS/README.md), choose your OS,
   and start the demo before class. Try both dataset panels, the processing stages,
   and the running sample; see the demo guide for a short sequence.
4. Keep `outputs/demo-visual-cleanup/replay/offline-demo.html` and
   `outputs/demo-visual-cleanup/Demo-Visual-Backup.pdf` available locally/on a flash drive.
   They contain dataset photographs and are deliberately ignored by Git.
5. If a backup is used, identify it as recorded. Do not claim training occurred
   during the demo or that either the six walkthrough examples or the 44-image teaching sample
   represents the full test results.

No additional model training is required for this presentation.
The first paper PDF and original evidence remain unchanged. Any later paper
revision should be reviewed separately rather than silently replacing them.

## Edit and rebuild

For hand edits, use PowerPoint/Google Slides. For reproducible source edits,
change `slides.json` (wording and notes) and, if necessary,
`scripts/build_presentation.py` (layout). Then, from the repository root:

```bash
python -m pip install -r requirements-presentation.txt
python scripts/build_presentation.py --office libreoffice
```

Install LibreOffice separately for this export command (`soffice` is also a
valid executable). Without `--office`, the builder creates a fallback PDF from
the same layout instead of exporting the PPTX. The supplied final PDF uses the
office export. All conversion runs headlessly.

The builder overwrites only the presentation PPTX/PDF and generated speaker
notes; it does not touch the IEEE paper. Export files should be reviewed again
after any source or layout change. Keep manual edits separately if you intend
to rebuild: the builder cannot merge edits made in PowerPoint/Google Slides.
PDF generation uses the DejaVu Sans fonts at the standard Ubuntu font path;
install `fonts-dejavu-core` or adjust the font path on another platform.

The committed `assets/` folder contains six small, credited photo derivatives
and three actual model-display captures. Rebuilding slides needs no raw datasets
or model weights. To recreate these illustrations, see PHOTO_CREDITS.md.

With the original local images, prediction CSVs and verified replay available:

```bash
python scripts/check_presentation.py --data-root /path/to/datasets \
  --predictions /path/to/final-01 --replay /path/to/replay.json
```

The checker also needs Poppler's `pdftotext`. It checks image/output hashes,
labels, prediction categories, full-result means, PDF text and slide boundaries.
It writes a local report under ignored `outputs/presentation-redesign/`.
Render every page with `pdftoppm` and inspect it after layout changes. Refresh
`Phase-4-Submission/presentation/` with the final PPTX, PDF and speaker notes.

Authoring tools: [python-pptx speaker notes](https://python-pptx.readthedocs.io/en/latest/user/notes.html).
Browser checks use [Playwright screenshots](https://playwright.dev/python/docs/screenshots).
Neither tool uploads these files or communicates with the team.
