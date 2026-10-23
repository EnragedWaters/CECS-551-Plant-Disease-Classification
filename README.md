# CECS-551-Plant-Disease-Classification
In class semester project

This project investigates how a classifier trained on controlled PlantVillage leaf photographs performs on less-controlled cropped PlantDoc photographs.

The planned experiment compares an ImageNet-pretrained MobileNetV3-Small with a frozen feature extractor against a fine-tuned version of the same model. Both will be evaluated on the same held-out images using shared plant–condition categories.

## Status

Planning stage. Dataset auditing, implementation, and training have not yet been completed in this repository folder. Setup instructions and runnable experiments will be added as the implementation develops.

## Planned evaluation

Report accuracy, macro-F1, category-level results, and computational cost. Keep training, validation, and test data separate, including related or duplicate images. Cross-dataset results will measure performance under the chosen protocol; they will not isolate weather or photography skill as causes.

## Data and generated files

Raw datasets and generated model weights are excluded from version control. Dataset access instructions and reproducible preparation steps will accompany the implementation.
