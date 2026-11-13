# Final evaluation

All six preselected models have been evaluated on the reserved test images.
The main result is a large drop between datasets: mean accuracy was 97.59%
frozen / 99.07% fine-tuned on PlantVillage, and 21.68% / 25.87% on PlantDoc.
Fine-tuning helped on this subset, but neither approach demonstrated reliable
performance across both datasets. The experiment investigated robustness;
it did not confirm that photography or weather has no effect.

Read the [results, plots, and limitations](evaluation/final-results.md).
The [per-class CSV](evaluation/per-class-results.csv) contains category counts
and detailed scores. The [JSON record](evaluation/final-results.json) preserves
every seed's results and the information needed to check them.

## What was fixed before testing

The three training seeds were 551, 552, and 553. Both models were retained for
every seed. The [selection record](training-runs/selected-checkpoints.json)
was committed at `4935bf8` before final evaluation, recording the six model
fingerprints, category configuration, and split fingerprints. The frozen models
use epochs 9/10/10; fine-tuned models use epochs 5/4/3. Selection used only
PlantVillage validation loss after warmup. No test score selected a model.

Evaluation uses the same RGB conversion, aspect-preserving resize, padding to
224 × 224, and ImageNet normalization used during training. There is no random
augmentation, prediction averaging, adaptation, or test-time training. Every
model sees all 5,487 PlantVillage test images and all 183 eligible PlantDoc test
images in the same order, with the same 22 output categories. Source validation
and either test set are different image lists.

## Run the evaluation

Follow [setup](setup.md), [data preparation](experiment-data.md), and
[training](training.md) first. The selected model files must exist locally.
From the repository root with the project environment active:

```bash
python -m plant_disease.evaluate --output-dir outputs/evaluation/final-01
python -m plant_disease.summarize_evaluation --evaluation-dir outputs/evaluation/final-01
```

The first command was run once for this experiment. It refuses to overwrite an
existing folder. The second command can regenerate reports from saved
predictions without opening test images or running the models again.

Use `--data-root` if the datasets and pinned weights are stored elsewhere.
`--runs-root`, `--splits-dir`, `--categories`, and `--selection` can point to
copies of the same recorded artifacts. Their fingerprints must still match.
Evaluation defaults to CUDA, batch size 32 from the training record, and two
loading workers. `--device cpu` explicitly chooses CPU execution; numerical
identity across devices/software is not guaranteed. `--workers 0` disables
background image loaders.

Fresh retraining can produce different checkpoint file fingerprints, even if
the method is the same. For a new reproduction, complete all three pairs,
verify them, and generate a new selection record **before looking at any of
its test predictions**. Preserve the original experiment's records separately.
Our current test results must not be used to tune these reported models; a
follow-up experiment would need a clearly separated evaluation plan.

## What is saved and checked

The ignored output folder contains settings, a copy of the selection record,
twelve per-image prediction CSVs, twelve detailed model/dataset result JSONs,
and one combined `results.json`. Predictions include the annotated category,
model guess, and softmax score. That score is not a calibrated probability of
correctness or a diagnosis.

The evaluator checks model and manifest fingerprints, category mappings,
counts, and known training/validation/test overlaps before inference. Image
bytes are checked against the audit. Evaluation uses no gradients, and a state
fingerprint check confirms that neither weights nor running statistics changed.

The exporter checks prediction file fingerprints and independently recomputes
accuracy, macro-F1, confusion counts, and per-class metrics from saved labels.
It requires exactly one result for each selected model on each dataset and
identical ordered images/labels across models. Nothing is silently skipped.

Shareable tables, plots, and the evidence JSON go under `docs/evaluation/`.
Raw image data, model weights, personal paths, and individual prediction CSVs
remain outside the committed results. The confusion plots show mean percentages
within each true category, across the three separate models of each type;
they do not represent ensemble predictions.

## What these results can support

They support a descriptive comparison of this frozen-feature baseline and this
fine-tuned model on these two filtered datasets. They show a large cross-dataset
performance gap, and better mean target scores for fine-tuning in this experiment.
They do not establish a causal effect of weather or camera skill, general field
accuracy, or statistical significance. Sample standard deviations describe
variation between training seeds on the same images, not uncertainty from new
test samples. PlantDoc has only 3–12 images per selected category. Label quality,
missing leaf metadata, and undetected altered duplicates remain limitations.

The next step is the IEEE-style Phase 4 report: describe the fixed method,
present all seeds and both datasets, interpret the gap, and discuss limitations
and future work. Final test results should be reported as measured rather than
retraining until the test score looks better.
