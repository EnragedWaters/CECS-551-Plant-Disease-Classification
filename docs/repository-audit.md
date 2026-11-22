# Repository maintenance audit — November 21, 2026

Reviewed from `20caec5`: repository files, data preparation, split/training/
evaluation safeguards, demo and launchers, setup/reproduction instructions,
checks and branch rules, and presentation/submission consistency.

## Findings and fixes

| Priority | Finding | Improvement |
| --- | --- | --- |
| High | ZIP extraction opened an existing image for writing before verifying the archive checksum. A damaged member could destroy that copy. Existing directory links could also redirect extraction outside the destination. | Extract each file to a temporary sibling, verify it completely, then replace its destination. Reject escaping links and unsafe Windows/Linux path spellings. Regression tests use tiny damaged archives, not the datasets. |
| Medium | Data preparation had a fixed home-directory destination and no useful command-line help. | Add `--data-root` and normal `--help`; retain default location, source revisions, color selection and bytes. Document custom paths on native Windows too. |
| Medium | Presentation authoring loaded Ubuntu fonts at import and relied on the OS text encoding. | Use Linux's existing fonts when available, otherwise Matplotlib's bundled DejaVu fonts; explicitly read/write UTF-8. Test rebuilding the complete deck in a separate folder without Linux fonts. |
| Medium | Matching submission copies could both be stale after slide-source changes. | Compare slide count, titles, subtitles, references, speaking portions and notes against the PPTX and Markdown notes. Test that stale source edits fail. A consistent slide count replaces the arbitrary twelve-slide limit. |
| Medium | Launcher preflight accepted Python 3.10, but evidence verification uses `hashlib.file_digest`, available from 3.11. | Reject older Python before model imports, with a clear message recommending the documented Python 3.12 environment. |
| Low | One common assessment filename escaped the existing file guard; setup prose mixed historical laptop details with current instructions. | Extend filename detection and its regression test. Remove old free-space claims and clarify completed research versus initial setup checks. |

The CPU check environment now includes the same pinned PowerPoint/ReportLab
packages already used for authoring. No research dependency versions changed.

## Intentionally retained

- The submitted paper, all original numerical results, category mappings,
  preprocessing and model weights. No retraining or new model selection.
- Original experiment snapshot `9619975`, historical notebooks and earlier plans.
  They document prior work; current commands and original reproduction commands
  serve different purposes. The [reproduction guide](reproduction.md) identifies
  the appropriate snapshot and separate output directories.
- The four byte-identical pairs are intentional paper/deck/notes submission
  copies. Their synchronization is checked; replacing them with links would
  make the submission folder less useful. No verified redundant script needed
  deletion, and no private/local files were removed.
- Existing one-click launcher locations, demo layout and predictions. The
  launcher fix changes its prerequisite error, not the presentation experience.
- Required Windows/Linux jobs, strict up-to-date-main protection, pull requests,
  and the absence of bypass actors. No new commit hooks or reviewer bottlenecks.

## Verification

- The baseline passed before changes. The updated suite covers corrupted ZIPs,
  escaping paths, custom data roots, unsupported Python, portable fonts, a full
  isolated deck build and stale exports, alongside existing scientific safeguards.
- `python scripts/check_project.py` runs syntax/import/configuration checks,
  unit tests, applicable OS launchers, and the live/offline headless browser demo.
  Required GitHub jobs run the same command on Windows and Linux before merging.
- Checked local Markdown file-link targets: none missing. Checked original
  reproduction flags against `9619975`, rather than assuming current flags match.
- Recomputed published aggregate metrics from all twelve saved prediction tables;
  they match. All six saved checkpoint fingerprints and the submitted paper hash
  match their original records. No original experiment files were rewritten.
- The existing presentation evidence check passed: original photos/captured maps,
  saved predictions, PDF text, result bars and page boundaries. Submission copies
  match. A separate bundled-font rebuild was exported through headless Office
  and checked without replacing the published presentation.

## Limits and worthwhile follow-up

Checks cannot establish scientific validity, catch every private filename, or
replace visual review. Source/export checks compare text, not every layout edit;
continue rendering and inspecting exports after intentional slide changes.
Native macOS and every possible Windows/WSL configuration are not covered by the
two GitHub runners. Rehearse on the actual presenting computer.

Archive replacement is atomic per file, not a transaction for the whole dataset.
The existing downloader records pinned source revisions and local hashes; only
PlantVillage's archive has a predeclared full download hash. A separately reviewed
source-lock manifest could tighten future download verification. This audit did
not redefine the provenance of already completed experiments.

No scientific or submission decision is needed for these maintenance changes.
Presentation timing and instructor-specific requirements still need confirmation
as described in the existing presentation guide. Published notes' software-test
counts describe an earlier snapshot; use current check logs for current totals.
