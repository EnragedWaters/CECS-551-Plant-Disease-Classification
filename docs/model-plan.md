# Model and computing plan

Status: proposed experiment. The local environment and basic GPU execution have
been verified; the plant-disease training pipeline and experiments have not started.

## Model choice

The starting model is ImageNet-pretrained MobileNetV3-Small. Its compact size
keeps the planned comparison manageable, while pretraining gives it learned
image features to start from. The original ImageNet classifier will be replaced
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

The implementation will use explicit `IMAGENET1K_V1` weights and record the
installed PyTorch/torchvision versions. The frozen feature extractor must remain
unchanged, including its batch-normalization running statistics. The existing
proposal also keeps those statistics fixed during fine-tuning; trainable weights
can still adapt. Trainable parameter counts will be checked before experiments.

Image preparation will follow the selected weights' input requirements, with
the resize/crop choice checked against actual leaf examples before it is fixed.
The documentation provides the reference preprocessing; automatic cropping
should not be assumed suitable before inspection.
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

When implementation begins, a short source-only trial will check:

- Whether both planned training versions run correctly on the GPU.
- Peak training memory and time per epoch on a recorded image count and batch size.
- Whether the proposed run budget fits the available time.

No runtime, training-memory requirement, or model score has been measured yet.
Batch size, learning rates, epoch limits, and repetition count remain provisional
until this check. A small saved model file does not establish training memory use.

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
The image transformations and training settings still need implementation and
source-only checks. The evaluation plan will report accuracy, macro-F1,
per-category results, counts, and confusion matrices, alongside measured costs.
Cross-dataset differences cannot identify weather or photography skill as causes.
