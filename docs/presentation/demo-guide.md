# Run the visual demonstration

The demo has two views: **Inside the models** shows real processing outputs;
**Compare 22 images per dataset** runs a fixed teaching sample. PlantVillage and
PlantDoc have separate, visible thumbnail panels. Both use the same original
seed-551 frozen and fine-tuned models. No training or downloads occur.

## Open from your operating system

Open **[ONE-CLICK-LAUNCHERS](../../ONE-CLICK-LAUNCHERS/README.md)** in the repository.

| System | Launcher in that folder |
| --- | --- |
| Windows (with or without WSL) | Double-click **Windows - Run Demo.cmd**. |
| macOS | Double-click **macOS - Run Demo.command**. |
| Linux / WSL | Run **Linux or WSL - Run Demo.sh**; from that folder use `bash "Linux or WSL - Run Demo.sh"`. A desktop file manager may ask to execute it or open it as text. |

The launchers locate the repository from their own location, even when the path
contains spaces. They find a ready Python environment, start the demo, and open
**http://localhost:8766**. Keep the terminal open; **Ctrl+C** stops the server.
CPU is the default; a GPU is not required. No packages or data are downloaded.

On Windows, a ready **native Windows** environment is preferred. WSL is checked
only if no native environment is ready. On Linux/macOS/WSL, Python runs natively.
The search checks `.venv` or `venv` inside the repository, an activated environment,
and Python on the system path; Linux/macOS/WSL also checks the original
`~/.venvs/cecs551-plant-disease` location. Windows can also use `py -3`.
A ready environment needs the packages, original files, and Git history below.
A real demo failure is reported; it does not silently start another model run.

Set **PLANT_DISEASE_PYTHON** to a specific Python executable if needed. An explicit
choice is respected: a bad path is reported instead of silently falling back.
Use Windows paths in Windows and POSIX paths in Linux/macOS/WSL. Environments
cannot be shared between Windows and WSL; create one for the OS where it runs.

## First-time setup on another computer

The launchers work after setup; a fresh clone does not include the environment,
weights or dataset photographs. Use a normal Git clone with history and install
Git and Python (3.12 was used for the original experiment).

Create a local environment from the repository folder:

```powershell
# Native Windows, with Python installed:
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
```

```bash
# Linux or macOS:
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Install PyTorch and torchvision using the [official OS/CPU installation selector](https://pytorch.org/get-started/locally/)
into that environment, then install NumPy and Pillow. Use `.venv\Scripts\python.exe -m pip`
on Windows or `.venv/bin/python -m pip` on Linux/macOS in place of the selector's
`pip` command. Existing project environments may already have these packages.
The original verified versions are recorded in [setup](../setup.md); package or
device differences can affect numerical scores. No new training is needed to
use the existing trusted models. [Python's environment guide](https://docs.python.org/3/library/venv.html)
explains why each OS needs its own environment.

Copy the exact saved models, test manifests, and pinned datasets listed below
privately onto the presenting computer. If their paths differ from the defaults,
create **demo-settings.local.json** in the repository folder, for example:

```json
{
  "data_root": "D:/datasets/cecs551-plant-disease",
  "runs_root": "outputs/runs",
  "splits_dir": "outputs/splits",
  "port": 8766,
  "device": "cpu"
}
```

Replace `data_root` with the actual local location. Windows forward slashes avoid
JSON backslash escaping. Paths can also be relative to the repository; `~` means
the current OS user's home. Omit settings to use defaults. The file is Git-ignored
so teammates do not inherit your personal paths. Launcher command-line arguments
such as `--port 8767` override these settings. Direct `python -m plant_disease.demo`
uses its own defaults/arguments; local JSON is a launcher convenience.

Windows native checkouts preserve the fingerprinted JSON and code line endings
through `.gitattributes`. If an older checkout reports changed evidence, inspect
local changes or use a fresh clone; do not edit hashes to bypass verification.
The launcher reports missing packages, data or history before starting. The
recorded HTML backup below needs none of those prerequisites and can be copied
with its attribution file to another computer.

Use browser full-screen for class; scroll or adjust zoom for the projector.

## Start without opening or focusing a browser

Every launcher accepts `--no-open-browser`. Normal double-click behavior still
opens the browser. For a quiet start from the repository root:

```powershell
& ".\ONE-CLICK-LAUNCHERS\Windows - Run Demo.cmd" --no-open-browser
```

```bash
bash "ONE-CLICK-LAUNCHERS/Linux or WSL - Run Demo.sh" --no-open-browser
# macOS uses the same flag with "macOS - Run Demo.command".
```

The server prints its local address; open it yourself when ready. The flag wins
even if `--open-browser` is also present. Automated browser checks run headlessly,
so they do not create visible windows or take focus.

## Inside the models

Choose a thumbnail in either dataset panel, then click **Run & show all stages**.
The default **All stages summary** displays all seven stages together in one
scrollable page, with frozen and fine-tuned outputs side by side at every stage.
There is no need to click each stage separately. Selecting another photograph
clears the old results until you run it.

**Play** switches to slowed explanatory playback. **Pause**, **Back**, **Next**,
the numbered stage buttons, and the speed control still work for individual
steps. **All stages summary** stops playback and returns to the complete view
without running the models again. Playback reveals captured outputs; it does
not mean inference takes that long.

| Stage | What it means |
| --- | --- |
| Original | The dataset photograph and its annotation (the expected category). |
| Prepared input | The actual image preparation used by the experiment: fit and pad to 224 × 224, then normalize numbers for the model. The visible preview shows the image before numeric normalization. |
| Early, middle, late features | Actual responses captured after MobileNetV3 feature blocks 0, 4, and 12. Each tile displays one channel: a grid of numbers produced by the model. |
| Pooling | The model averages each of its 576 final channels into one number. All 576 numbers are arranged as a square for display; their positions are not locations in the photograph. |
| Category scores | The final layer produces 22 scores, converted with softmax. The highest score is the prediction. The top five are visible; expand the rest to see all categories. A large score does not guarantee correctness. |

Feature pictures use channels **0–3**, fixed in advance. Blue shows negative
responses, orange positive, and pale colors values near zero. At each stage,
both models use **one shared scale** based on the displayed channels’ largest absolute response;
the displayed shape and scale describe the captured numbers. Earlier and later
stages can have different scales. These are feature responses, **not attention
maps, highlighted disease diagnoses, or proof of why a category was selected**.
There are no invented intermediate category guesses.

### Learn what changes and why

The default view uses one short caption per stage. Photos, feature maps and
score bars stay visible together. **Why this step?** opens the longer explanation;
**Model setup & dimensions** contains the training comparison and dimension strip.
Pooling shows a real 7 × 7 map → its average → the 576-value display immediately.
The dimension strip reads **channels × height × width**: for example, 576 × 7 × 7
means 576 separate grids, each with 49 positions. These are selected observation
points, not every internal layer.

- **Prepared:** the outline marks the resized photograph; space outside it is
  padding. Square photographs need no padding. Expand the color-number lesson
  under **Pixel values & resizing** to follow a real center pixel through the original normalization settings.
  The ordinary-color preview is not a picture of the normalized numbers.
- **Early:** expand **Try a toy filter (separate example)** and switch from a uniform patch to
  opposite edges. Multiplication and addition explain how a pattern produces a
  response. This hand-written toy is separate from MobileNet and never changes
  the photograph or model. It does not identify what a real channel detects.
- **Pooling:** expand **Inspect the numbers**. Choose channels 0–3 and follow
  their actual 7 × 7 grid, sum, division by 49, and observed average. The outlined
  cell in the 576-value display follows that channel. Averaging keeps a summary
  while losing exact spatial layout. The numeric grid uses the shared late-stage
  color scale; the 576-value display uses the pooled stage's own shared scale.
- **Scores:** expand **Inspect score contributions** Choose the prediction,
  runner-up, or dataset label. Repeated roles share one option. Five largest
  absolute contributions are shown as feature × weight, with the other 571
  added together and the learned bias included. Positive and negative bars raise
  or lower that category's raw score; they do not establish a biological cause.
  Bar lengths scale within the selected category; compare printed numbers
  across models or categories.
  Softmax uses all 22 raw scores to produce the final shares, not just these
  expanded categories.
- **Frozen versus fine-tuned:** expand the training comparison above the stages.
  Frozen feature weights stayed unchanged during plant training; fine-tuned
  weights could adapt after warmup. Both models are fixed during this demo.
  A new image changes responses, not weights.

Lesson controls read captured outputs without rerunning inference. Switching
photographs clears old calculations. Expanded numbers are rounded for reading;
backend checks use unrounded values. All stages remain together in the summary.

The six thumbnails include correct and incorrect predictions selected after the
original evaluation. They are illustrations, not a representative accuracy test.
Seed 551 is the first declared seed, not a best-performing seed selection.

## Running comparison across both datasets

Open **Compare 22 images per dataset**. **Start comparison** processes one image
from each dataset at a time, through both models. Progress, the current images,
correct/incorrect counts and running accuracy update as each pair finishes.
**Pause** finishes the current pair and then stops; **Next pair** advances once.
**Resume** continues; **Reset** clears only the demonstration counts. Adjust the
pause between pairs, including zero pause for a quick classroom run.

Selection is reproducible and independent of predictions: within each of the
22 shared categories, select the lowest image SHA-256 checksum, then relative
path to break ties, from each original test manifest. This yields **44 images
and 88 predictions**. A pair has the same annotated category, not the same leaf.
Source paths, image hashes, labels, and outputs are stored in the local replay's
`replay.json`. The UI also shows the current source path and a result log.

On the prepared CPU run, both models matched **22/22 PlantVillage** annotations
and **5/22 PlantDoc** annotations. All 88 predicted categories matched their
original saved evaluation records. This small, category-balanced sample differs
from the full test distribution. It does not estimate general accuracy or replace
the published evaluation of 5,487 PlantVillage and 183 PlantDoc images over three
seeds. In particular, its equal counts do not contradict the full evaluation's
average improvement after fine-tuning. Do not choose different samples simply
to make one approach look better.

## Suggested classroom sequence (about two minutes)

1. Identify both datasets and the frozen/fine-tuned difference. Run one
   PlantVillage image and follow the visuals down the page.
2. Point out the prepared-photo outline, then the smaller response maps. Use
   the pooling diagram to show a map becoming one average. Explain the score
   bars; expand arithmetic only if someone asks.
3. Show a PlantDoc mistake (example 6), or example 5 where only fine-tuning
   matches the annotation. Start a few sample pairs; call partial counts partial.
4. Return to the full-results slide. These lessons illustrate computation;
   published evaluation supports the research conclusions.

The toy filter and deeper arithmetic are useful for questions or a longer
learning session. Do not try to explain every control within two minutes.
Confirm the actual presentation time allowance before rehearsing.

## What must exist locally

A fresh GitHub clone does not contain datasets or saved models. Required:

- Git history containing original commit `9619975`; a ZIP or shallow clone may
  lack it.
- The pinned datasets and ImageNet weights under
  `~/datasets/cecs551-plant-disease/`.
- `outputs/splits/source_test.csv` and `target_test.csv`.
- Original `outputs/runs/source-seed-551/frozen/best.pt` and
  `outputs/runs/source-seed-551/finetune/best.pt`.
- The project Python environment and packages; no web framework is needed.

Use `--data-root`, `--runs-root`, and `--splits-dir` for other locations, keeping
the directory structure beneath each root. See [setup](../setup.md),
[data access](../data-downloads.md), and [reproduction](../reproduction.md).
The demo accepts the exact original checkpoints, not arbitrary retrained files.
Use the prepared laptop or transfer trusted original files privately for another
presenting machine. Keep weights and images out of Git.

## How the real outputs are checked

The demo loads original model and preprocessing code from Git history into an
isolated temporary package. It verifies code fingerprints, checkpoint hashes,
category order, test manifests, and image bytes. It does not switch branches or
relax the original checks.

Observation hooks copy outputs from actual feature blocks and average pooling;
they do not replace outputs. Each captured run is compared with an ordinary
forward pass: logits must be exactly equal on that device. Weights and buffers
are checked before and after inference, gradients are disabled, and hooks are
removed even if processing fails. Unit tests cover these properties. The learning capture checks all 576 means
against observed pooling, then all 22 logits against `sum(feature × weight) + bias`.
Double-precision teaching arithmetic is compared with float32 inference using
small numerical tolerances; it does not replace the actual prediction. CPU scores
can differ slightly from original GPU/batched scores; the original measured
results are never overwritten. See [PyTorch's hook documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

## Recorded and static backups

The prepared portable replay is:

```text
outputs/demo-visual-cleanup/replay/offline-demo.html
```

Double-click it in File Explorer. It contains both views, all six processing
captures and all 22 sample pairs. It needs no Python or network. Its badge says
**Recorded replay · offline**; click **Show recorded stages** for the full summary. Explicitly identify it as a recorded backup in
class. Copy the HTML and neighboring `ATTRIBUTION.md` to your flash drive.

Static PDFs and screenshots are under:

```text
outputs/demo-visual-cleanup/browser-check/
outputs/demo-visual-cleanup/Demo-Visual-Backup.pdf
```

These contain dataset images and stay Git-ignored. The combined PDF includes
processing stages and completed sample counts. Earlier backups remain available
but show the older interface; use the paths above for this version.

To generate another replay, choose a NEW folder (existing exports are protected):

```bash
python -m plant_disease.demo --no-open-browser --export outputs/phase4-demo-learning/replay-new
```

For browser verification, install the optional tools once, start the live demo,
and use a new output directory for each run:

```bash
python -m pip install -r requirements-presentation.txt
python -m playwright install chromium --only-shell
python scripts/check_demo_browser.py --output-dir outputs/phase4-demo-learning/browser-check-new
```

The default replay path is the prepared `replay` above. Override `--url`,
`--replay`, or `--browser` for other locations. Checks exercise all seven stages
of all six examples, both datasets, playback, sample counts, pause/resume/reset,
learning controls, real arithmetic, and narrow windows. Offline checks block HTTP. Screenshots and printable PDFs
are saved in the specified ignored folder. Nothing is uploaded.

## If something fails

- **Missing history:** fetch the original history; do not substitute current code.
- **Missing models/data:** use the prepared laptop or recorded replay.
- **Fingerprint mismatch:** locate the original files; do not edit expected hashes.
- **Address occupied:** choose another port; leave unrelated processes alone.
- **Browser cannot connect:** wait for startup and keep the terminal open. Try
  `http://127.0.0.1:8766` inside WSL if Windows localhost forwarding fails.
