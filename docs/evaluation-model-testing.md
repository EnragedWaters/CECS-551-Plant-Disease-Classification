## Testing the model and presenting the results

This section records the testing plan before training and scoring begin. The model will be evaluated on images that were not used to fit its weights or choose settings.

Same final categories and test images will be used for both model versions. The supplied PlantVillage lists contain
43,596 training paths and 10,709 test paths. All paths exist locally, with no filename overlap. These are source lists,
not our final experiment counts; different filenames can still show the same leaf.

### Which pictures go where?

| Pictures | Purpose |
| --- | --- |
| One PlantVillage training portion | Teaching the model the categories |
| Separate PlantVillage validation portion | Configuring model settings, compare frozen and fine-tuned versions, and apply early stopping. |
| Reserved PlantVillage test portion | Measuring performance on unseen pictures from the original dataset, to be used for final evaluation |
| Matching categories from the existing PlantDoc test folders | Measuring performance on a different dataset and test domain transfer. PlantDoc training images remain outside the main experiment. |

Both model versions must use the same PlantVillage split, image preparation, category order, and evaluation sets. Related pictures or pictures of the same leaf will remain in one split so the test score does not benefit from near-duplicates.

PlantDoc is an external evaluation set, not a tuning set. Its test counts are small, and some category matches still need confirmation. Only mappings supported by the category review should be included in the primary comparison. If a mapping cannot be supported, we will exclude it consistently and document the reason before training.

### How will the results be judged?

The main measures are:

- **Accuracy:** the overall percentage of correct predictions.
- **Balanced accuracy:** the average recall across categories, so large categories do not dominate the result.
- **Per-category precision, recall, and F1:** shows which diseases or healthy categories work well or poorly.
- **Macro-F1:** the average F1 across categories, giving each category equal weight.
- **Weighted-F1:** summarizes performance while accounting for the number of examples in each category.
- **Confusion matrix and category counts:** a table to show which categories are mistaken for others, together
with how many examples were tested.

The evaluation cell below displays a classification report and two confusion matrices. Rows are the actual labels and columns are the predicted labels. Diagonal cells are correct predictions; off-diagonal cells show which categories were confused.

- The **count matrix** shows the number of pictures in each result.
- The **row-normalized matrix** shows the fraction of each actual category assigned to each prediction, which makes categories with different test sizes easier to compare.

The PlantVillage test result is the in-domain result. The PlantDoc test result is the external-domain result. We will compare frozen-feature and fine-tuned MobileNetV3-Small using the same measures rather than selecting a model based on PlantDoc test performance.

### Possible problems and responses

| Possible problem | Response |
| --- | --- |
| Categories do not match between datasets | Use only confirmed matches. Do not select mappings because they produce better scores. |
| Too few pictures of a disease | Report the support count and explain that the per-category result may be unstable. Do not hide the category. |
| Possible duplicate or related pictures cross a split | Audit filenames, hashes, and related-leaf records; keep related pictures together and document exclusions in a separate manifest. |
| PlantDoc labels are broader or uncertain | Keep the mapping out of the primary comparison until source evidence supports it; report it as a limitation if it remains unresolved. |
| Training needs too much time or memory | Run a short timing and memory trial, reduce batch size if needed, and apply the same comparison rules to both model versions. |
Colab sessions or access may be interrupted | Prefer the laptop and save resumable checkpoints if Colab is used
| Fine-tuning improves PlantVillage but harms PlantDoc | Report both results. This may indicate domain shift rather than a reason to tune on the PlantDoc test set. |

### Computing feasibility

The available RTX 4060 Laptop GPU has approximately 8 GB of video memory. PyTorch and torchvision are
installed, a synthetic-image GPU check passed, and the downloaded pretrained weights passed their hash check and
loaded successfully. The data, weights, and retained archives occupy about 4.9 GiB. These checks establish access
and basic execution, not training speed, memory fit, or accuracy.
A short PlantVillage-only training trial will measure memory use and time per epoch for both versions. We will use
it to set batch size and a practical run budget. The laptop is the first option; Google Colab is a fallback with variable
availability. [7] Dataset differences will be reported without attributing them specifically to weather or photography
skill.