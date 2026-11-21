# Run the visual demonstration

The demo shows an original leaf photograph, its prepared input, the dataset's
label, and predictions from both saved models. It runs locally in a browser.
It does not train, download anything, or change the submitted results.

## On the prepared laptop

In a WSL terminal, from the repository folder:

```bash
source ~/.venvs/cecs551-plant-disease/bin/activate
python -m plant_disease.demo
```

Wait for **Open http://localhost:8766**, then open that address in your browser.
Keep the terminal running. **Ctrl+C** stops it. CPU is the default, so a working
GPU is not needed. If the port is occupied, use `--port 8767` and that address.
For a small projector, use browser full-screen and reduce zoom enough to show
both photographs and the prediction cards together.

Choose an example, then click **Run both models**. This computes predictions
at that moment; it does not just display the old prediction table. Previous,
Next, the dropdown, and the left/right arrow keys move between examples.

A simple 60–90 second demonstration:

1. Example 1: both models match the PlantVillage annotation. Point out the
   original photo, the padded input, and the two predictions.
2. Example 5: only the fine-tuned model matches this PlantDoc annotation.
3. Example 6: neither model matches. A high model score does not guarantee a
   correct answer. Return to the slide showing the full test results.

These examples were deliberately selected **after** evaluation to illustrate
successes and mistakes. They are not representative or a new estimate of accuracy.
Both models use **seed 551**, the first declared seed, not the best-performing one.
The full paper uses all three seeds and all 5,487 / 183 selected test images.

## What must exist locally

A fresh GitHub clone does not include datasets or saved models. This demo uses:

- The original repository history containing commit `9619975` (use a normal
  Git clone; a downloaded ZIP or shallow clone may lack this history).
- The pinned raw images and ImageNet weights under
  `~/datasets/cecs551-plant-disease/`.
- Original `outputs/splits/source_test.csv` and `target_test.csv`.
- Original `outputs/runs/source-seed-551/frozen/best.pt` and
  `outputs/runs/source-seed-551/finetune/best.pt`.
- The existing project Python environment and packages. No extra web framework
  or presentation-authoring package is needed for the live demo.

Use `--data-root`, `--runs-root`, and `--splits-dir` for other local locations.
Keep the same directory structure beneath each supplied root.
For missing datasets/environment, follow [setup](../setup.md) and
[data access](../data-downloads.md). [Reproduction](../reproduction.md) explains
fresh training separately. The demo intentionally accepts the **original saved
checkpoints**, not arbitrary retrained checkpoints whose hashes differ.
Use the prepared laptop or transfer the exact trusted files privately for a
second presenting machine; do not put weights or raw images in Git.

## Why old code is loaded automatically

The paper's saved models were trained before optional experiment changes.
The demo reads the original model and image-processing files from Git history
into a temporary directory, verifies their hashes against the saved run, and
loads them under an isolated name. It does **not** switch your branch, alter
current training code, disable fingerprint checks, or edit checkpoint metadata.

It checks the checkpoint files, category order, test-list hashes, and each
example's image bytes. Inference uses evaluation mode and no gradients; a state
check confirms that model weights and buffers stay unchanged. CPU/single-image
scores can differ slightly from the original GPU/batch evaluation. The live
prediction is displayed honestly, including a notice if its predicted category
changes. These demo outputs never replace the original measured results.

## Offline backup

On the prepared laptop, the portable backup is:

```text
outputs/phase4-demo/replay/offline-demo.html
```

Double-click it from Windows File Explorer. It contains the images and recorded
outputs, works without Python, and needs no network. Its badge says **Recorded
replay · offline**, and its button says **Show recorded predictions**. If used
in class, explicitly say this is a recorded backup, not a live model run.
Copy the HTML and `ATTRIBUTION.md` together to your flash drive.

To create a new backup from the verified saved models:

```bash
python -m plant_disease.demo --export outputs/phase4-demo/replay-new
```

Use a new destination; existing backups are not overwritten. Export computes
both models' outputs for all six examples and embeds them in one HTML file.
The image-bearing backups are kept out of Git. Source paths, selection policy,
and attribution are in `demo/examples.json` and `demo/ATTRIBUTION.md`.

## Optional browser verification and static backups

This step is for maintainers; it is not needed to run the demo. Install the
optional tools and Chromium once:

```bash
python -m pip install -r requirements-presentation.txt
python -m playwright install chromium --only-shell
```

With the demo running in another terminal and the replay exported:

```bash
python scripts/check_demo_browser.py --output-dir outputs/phase4-demo/browser-check-new
```

The check clicks through all six examples live and offline, checks that images
load and predictions appear, checks navigation and narrow windows, and saves
screenshots plus six printable PDF backups. Use `--url`, `--replay`, or
`--browser` if the server, backup, or Chromium executable is elsewhere.
These tools do not upload the project or contact teammates.

On the prepared laptop, the reviewed screenshots and PDFs are under
`outputs/phase4-demo/browser-check-03/`; `outputs/phase4-demo/Demo-Backup.pdf`
combines the six printable examples. They are recorded illustrations.

## If something fails

- **Missing history:** fetch the original commit/history from origin; do not
  replace old code with current code to silence a check.
- **Missing models/data:** use the prepared machine or the offline backup.
- **Changed file / fingerprint mismatch:** stop and locate the original files;
  do not edit expected hashes to make a different experiment pass.
- **Address already in use:** choose another port; leave the unrelated process alone.
- **Browser cannot connect:** wait for the startup URL, keep the terminal open,
  and try `http://127.0.0.1:8766` inside WSL if Windows localhost forwarding fails.
