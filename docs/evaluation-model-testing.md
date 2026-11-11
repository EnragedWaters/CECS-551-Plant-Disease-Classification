## 5. Testing the model and presenting the results

This section records the testing plan before training and scoring begin. The model will be evaluated on images that were not used to fit its weights or choose settings.

### Which pictures go where?

| Pictures | Purpose |
| --- | --- |
| One PlantVillage training portion | Teach the model the categories. |
| Separate PlantVillage validation portion | Choose model settings, compare frozen and fine-tuned versions, and apply early stopping. |
| Reserved PlantVillage test portion | Measure performance on unseen pictures from the original dataset. Use this only for the final evaluation. |
| Matching categories from the existing PlantDoc test folders | Measure performance on a different dataset and test domain transfer. PlantDoc training images remain outside the main experiment. |

Both model versions must use the same PlantVillage split, image preparation, category order, and evaluation sets. Related pictures or pictures of the same leaf should remain in one split so the test score does not benefit from near-duplicates.

PlantDoc is an external evaluation set, not a tuning set. Its test counts are small, and some category matches still need confirmation. Only mappings supported by the category review should be included in the primary comparison. If a mapping cannot be supported, exclude it consistently and document the reason before training.

### How will the results be judged?

The main measures are:

- **Accuracy:** the overall percentage of correct predictions.
- **Balanced accuracy:** the average recall across categories, so large categories do not dominate the result.
- **Per-category precision, recall, and F1:** shows which diseases or healthy categories work well or poorly.
- **Macro-F1:** the average F1 across categories, giving each category equal weight.
- **Weighted-F1:** summarizes performance while accounting for the number of examples in each category.
- **Support:** the number of test pictures behind each category's score.

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
| Fine-tuning improves PlantVillage but harms PlantDoc | Report both results. This may indicate domain shift rather than a reason to tune on the PlantDoc test set. |

No model scores are required until training and final category decisions are complete. The notebook should preserve the data counts, split definitions, model settings, random seeds, and software versions used for the final results.