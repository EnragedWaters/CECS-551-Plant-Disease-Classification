# Run the visual demonstration

The demo has two views: **Inside the models** shows real processing outputs;
**Compare 22 images per dataset** runs a fixed teaching sample. PlantVillage and
PlantDoc have separate, visible thumbnail panels. Both use the same original
seed-551 frozen and fine-tuned models. No training or downloads occur.

## Double-click from Windows

In the repository folder in Windows File Explorer, double-click **Run Demo.cmd**.
It starts the prepared environment in your default WSL distribution (Ubuntu on
this laptop) and opens your Windows browser. Keep its terminal window open;
press **Ctrl+C** there to stop. An error leaves the window open to read.
The launcher works from its own folder, including paths with spaces.

The default WSL distribution must contain the environment and saved files below.
A different Python path can be set through `PLANT_DISEASE_PYTHON` inside WSL.
Nothing is installed automatically. Alternatively, from a WSL terminal:

```bash
source ~/.venvs/cecs551-plant-disease/bin/activate
python -m plant_disease.demo
```

Open **http://localhost:8766** after the startup message. CPU is the default;
a GPU is not needed. Use `--port 8767` if another program occupies the port.
Use browser full-screen for class; scroll or adjust zoom for the projector.

## Inside the models

Choose a thumbnail in either dataset panel, then click **Run & watch**. This
runs both saved models and captures real outputs. It then plays those outputs
slowly so there is time to explain them. **Play**, **Pause**, **Back**, **Next**,
the numbered stages, and the speed control let you explore at your own pace.
This is explanatory playback, not a claim that inference takes that long.

| Stage | What it means |
| --- | --- |
| Original | The dataset photograph and its annotation (the expected category). |
| Prepared input | The actual image preparation used by the experiment: fit and pad to 224 × 224, then normalize numbers for the model. The visible preview shows the image before numeric normalization. |
| Early, middle, late features | Actual responses captured after MobileNetV3 feature blocks 0, 4, and 12. Each tile displays one channel: a grid of numbers produced by the model. |
| Summary | The model averages each of its 576 final channels into one number. All 576 numbers are arranged as a square for display; their positions are not locations in the photograph. |
| Category scores | The final layer produces 22 scores, converted with softmax. The highest score is the prediction. The top five are visible; expand the rest to see all categories. A large score does not guarantee correctness. |

Feature pictures use channels **0–3**, fixed in advance. Blue shows negative
responses, orange positive, and pale colors values near zero. At each stage,
both models use **one shared scale** based on the displayed channels’ largest absolute response;
the displayed shape and scale describe the captured numbers. Earlier and later
stages can have different scales. These are feature responses, **not attention
maps, highlighted disease diagnoses, or proof of why a category was selected**.
There are no invented intermediate category guesses.

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

1. Point out both dataset panels. Select a PlantVillage photograph and run the
   walkthrough. Pause at Early features and explain one channel and the colors.
2. Step to Summary and Scores. Select PlantDoc example 6 to show a mistake;
   example 5 illustrates a case where only fine-tuning matches the annotation.
3. Start the running comparison. Use zero pause or show only a few pairs if time
   is short. Identify partial counts as partial. Explain the selection rule.
4. Return to the full-results slide. The sample and feature pictures illustrate
   behavior; the published evaluation supports the research conclusions.

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
removed even if processing fails. Unit tests cover these properties. CPU scores
can differ slightly from original GPU/batched scores; the original measured
results are never overwritten. See [PyTorch's hook documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook).

## Recorded and static backups

The prepared portable replay is:

```text
outputs/phase4-demo-v2/replay-final/offline-demo.html
```

Double-click it in File Explorer. It contains both views, all six processing
captures and all 22 sample pairs. It needs no Python or network. Its badge says
**Recorded replay · offline**. Explicitly identify it as a recorded backup in
class. Copy the HTML and neighboring `ATTRIBUTION.md` to your flash drive.

Static PDFs and screenshots are under:

```text
outputs/phase4-demo-v2/browser-check-final/
outputs/phase4-demo-v2/Demo-Backup.pdf
```

These contain dataset images and stay Git-ignored. The combined PDF includes
processing stages and completed sample counts. Earlier backups remain available
but show the older interface; use the paths above for this version.

To generate another replay, choose a NEW folder (existing exports are protected):

```bash
python -m plant_disease.demo --export outputs/phase4-demo-v2/replay-new
```

For browser verification, install the optional tools once, start the live demo,
and use a new output directory for each run:

```bash
python -m pip install -r requirements-presentation.txt
python -m playwright install chromium --only-shell
python scripts/check_demo_browser.py --output-dir outputs/phase4-demo-v2/browser-check-new
```

The default replay path is the prepared `replay-final` above. Override `--url`,
`--replay`, or `--browser` for other locations. Checks exercise all seven stages
of all six examples, both datasets, playback, sample counts, pause/resume/reset,
and narrow windows. Offline checks block HTTP. Screenshots and printable PDFs
are saved in the specified ignored folder. Nothing is uploaded.

## If something fails

- **Missing history:** fetch the original history; do not substitute current code.
- **Missing models/data:** use the prepared laptop or recorded replay.
- **Fingerprint mismatch:** locate the original files; do not edit expected hashes.
- **Address occupied:** choose another port; leave unrelated processes alone.
- **Browser cannot connect:** wait for startup and keep the terminal open. Try
  `http://127.0.0.1:8766` inside WSL if Windows localhost forwarding fails.
