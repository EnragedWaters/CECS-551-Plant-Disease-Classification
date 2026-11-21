# Demonstration images and recorded outputs

The live demo loads images from the local datasets. Its generated replay/backups
stay in ignored `outputs/`. The example manifest records relative source names
and checksums. A small set of credited presentation derivatives is included in
[the slide assets](../docs/presentation/PHOTO_CREDITS.md); raw datasets remain local.

- **PlantVillage:** S. P. Mohanty, author-hosted distribution, revision
  `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
  https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7
  Declared license: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/
- **Cropped PlantDoc:** D. Singh et al., PlantDoc (2020), cropped repository
  maintained by P. Kayal et al., revision `5467f6012d78d1c446145d5f582da6096f852ae8`.
  https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8
  Declared license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The source path for each photograph is visible in the demo and stored in
`examples.json` for the six walkthrough examples. The 44-image running sample
is recorded with relative source paths, checksums and labels in the exported
`replay.json`. Display copies correct orientation and resize; model previews
also pad to 224 × 224. Preserve these credits and the source image licenses
with exported image copies. Dataset-provided licensing is not an independent
investigation of the provenance of every internet-sourced PlantDoc photograph.

Predictions come from the original seed-551 models, selected before final testing.
The examples deliberately include correct and incorrect predictions after the
full evaluation. They are not random, representative, or a new test set.

The running comparison selects one image per category per dataset from the original
test manifests: lowest SHA-256 checksum, then relative path to break ties. This
selection does not use predictions. Its 22 pairs cover all shared categories but
are not representative of dataset frequencies or an independent new evaluation.
Feature-map pictures display real model activations; they are not attention maps
or a definitive explanation of a prediction.
