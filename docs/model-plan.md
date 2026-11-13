# Model and computing plan

Status: implementation, GPU trial, and all three full-source training pairs are
complete. All six selected models passed reload checks. Final test evaluation
and the final report remain. See the
[training guide](training.md) for commands, measured costs, and current settings.

## Model choice

The starting model is ImageNet-pretrained MobileNetV3-Small. Its compact size
keeps the planned comparison manageable, while pretraining gives it learned
image features to start from. The original ImageNet classifier is replaced
with a new linear category layer after the pooled image features. Its output
size comes from the shared category configuration. The initial Phase 4
[data preparation](experiment-data.md) selects 22 plant–condition categories.

The 2025 benchmark by Richter and Kim includes MobileNetV3-Small and compares
frozen features with subsequent fine-tuning across plant-disease datasets. This
supports the comparison as a reasonable starting point, not a guarantee of
performance on a different dataset. Their models were trained and evaluated
within each dataset separately, with a different implementation and classifier.
This project is not a reproduction of their exact settings or reported scores.
[Paper](https://doi.org/10.1038/s41598-025-03235-w).

## Two model versions

| | Frozen-feature baseline | Fine-tuned version |
| --- | --- | --- |
| Starting point | ImageNet-pretrained MobileNetV3-Small | Same pretrained starting weights |
| New classifier | One linear category layer | Same layer design and matched initial values |
| What learns | Only the new category layer | The new layer, then the pretrained image-processing layers as well |
| Purpose | Test how useful the existing features are | Test whether adapting those features helps |

The baseline is deliberately simple: a linear classifier over fixed pretrained
features. Fine-tuning tests one main change while keeping the underlying
architecture the same. Both approaches are illustrated in the
[PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

Both versions use the same PlantVillage training/validation split, shared
categories, and preparation rules. Model settings and checkpoints are selected
with source validation only. Final evaluation uses the same reserved
PlantVillage images and eligible existing cropped PlantDoc test images.
PlantDoc training images remain unused in the main experiment.

The implementation uses explicit `IMAGENET1K_V1` weights and records the
installed PyTorch/torchvision versions. The frozen feature extractor must remain
unchanged, including its batch-normalization running statistics. The existing
proposal also keeps those statistics fixed during fine-tuning; trainable weights
can still adapt. The trial verified 12,694 trainable parameters in the frozen
version and 939,702 during fine-tuning.

Image preparation converts to RGB, preserves the entire image while resizing
and padding to 224 × 224, and applies ImageNet normalization. Source examples
were checked visually. Padding deliberately replaces the reference center crop
to avoid removing image edges; it has not been proven the most accurate choice.
[Official model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).

## Computing feasibility

An October 23, 2026 check found an NVIDIA GeForce RTX 4060 Laptop GPU with
8,188 MiB of total video memory, approximately 8 GiB. A dedicated WSL2
environment now has Python 3.12.3, PyTorch 2.13.0, and torchvision 0.28.0
with CUDA 12.6 support. MobileNetV3-Small successfully processed one synthetic
image on the GPU using random weights. Package compatibility also passed
`pip check`. This confirms basic execution, not real training memory fit,
speed, or accuracy. See [the setup guide](setup.md) for the environment and check.

The laptop is the first option; Google Colab is a fallback. Colab accelerator
availability and runtime limits vary, so the plan does not assume a particular
GPU or uninterrupted session. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

On November 13, both versions completed a two-epoch trial using 704 training
and 176 validation images with batch size 32. Peak PyTorch allocated memory was
148 MiB frozen and 579 MiB fine-tuned; reserved memory peaked at 200 and 712 MiB.
These exclude CUDA context and other applications. Saved models reloaded
successfully, and checks confirmed that only the intended weights changed.
The [training guide](training.md) records timings and limits. The short trial
supports GPU feasibility, not final accuracy or a full-experiment time budget.
The first full-data pair subsequently completed: ten frozen epochs and eight
fine-tuned epochs took about 11.1 minutes of combined training/validation loops.
Its saved models reproduced their selected validation metrics after reload.
See the [results table and curves](training-runs/source-seed-551.md). The declared
budget was three matched pairs (training seeds 551, 552, 553) on the same fixed
splits and settings. All are complete and reported in the
[combined comparison](training-runs/repeated-training.md). Mean validation
accuracy is 97.54% frozen and 98.98% fine-tuned. These are checkpoint-selection
scores on one validation set; final tests remain reserved. All six selected
checkpoint fingerprints are recorded before testing.

| Possible problem | Planned response |
| --- | --- |
| Fine-tuning exceeds memory | Reduce batch size and apply the revised size consistently to both versions before the comparison |
| Training takes too long | Use validation-based early stopping and a declared run budget; document any reduction in repetitions |
| Colab disconnects or is unavailable | Prefer the laptop; save resumable checkpoints if Colab is used |
| Fine-tuning improves PlantVillage but harms PlantDoc | Report both results; avoid tuning against PlantDoc test scores |
| Category matches or image counts are unsuitable | Finalize the output categories after the data audit and narrow scope with documented reasons if needed |

## Still to confirm

The initial category selection and image lists are recorded in the experiment
data guide. Known byte duplicates and resolved leaf groups are kept from crossing
splits; incomplete metadata and remaining label uncertainty are limitations.
Image preparation and both training modes are implemented and checked in the
trial and three full-source pairs. Final evaluation remains. Use all six
recorded checkpoints and the fixed settings; do not select a seed or retune
using final test performance. The evaluation
plan will report accuracy, macro-F1,
per-category results, counts, and confusion matrices, alongside measured costs.
Cross-dataset differences cannot identify weather or photography skill as causes.
