# Phase 4 speaker notes

Provisional ~9-minute version including demo; duration and dates must be confirmed. Slides 11–12 are backup.

These are suggested handoffs, not confirmed commitments. Agree on speakers and rehearse together.

For ~5 minutes: shorten slides 2–5, demonstrate examples 1 and 6 only, and summarize slides 8–9. For more time: discuss the full demo and questions. Confirm the actual time limit first.

## 1. Plant disease classification across image datasets

Suggested speaker: **Jacob**. Suggested time: 30 seconds.

Our project asks whether a plant disease classifier can handle images beyond the collection it learned from. We compared two ways of adapting the same lightweight model. The experiment is complete; this presentation explains our measured results, including where the models fail. The main finding is a large drop on the second dataset, not a claim that we solved plant diagnosis.

## 2. The practical problem

Suggested speaker: **Jacob**. Suggested time: 35 seconds.

A gardener will not necessarily control lighting, framing, or background. This motivates evaluating a classifier on a different image collection. We are investigating robustness, not promising it. Comparing two datasets does not isolate weather or photography skill as causes. This is a research comparison; we did not build a diagnostic service.

## 3. Two collections, one shared label set

Suggested speaker: **Jacob**. Suggested time: 40 seconds.

PlantVillage is the source collection: the one we trained on. Cropped PlantDoc is the target collection: the different one used for external testing. We matched 22 plant-condition categories and excluded five uncertain pairs rather than assume similar names meant identical conditions. These labels are dataset annotations, not diagnoses independently verified by us. Cropped PlantDoc is also not a controlled sample of future users.

## 4. Keep learning separate from testing

Suggested speaker: **Troy**. Suggested time: 45 seconds.

Training fits the model weights. Validation selects the saved checkpoint with the lowest prediction loss, without using the final tests. The two test sets answer different questions: familiar collection versus different collection. We grouped known photos of the same leaf and checked identical files to reduce overlap. Some related or altered duplicates may still escape these checks. PlantDoc test results did not select models or training settings. The first two boxes describe development; the last two are separate final evaluations, not additional training stages.

## 5. Same starting model. Different learning.

Suggested speaker: **Troy**. Suggested time: 50 seconds.

Pretrained means the model already learned image patterns on ImageNet before our plant experiment. The frozen baseline reuses these patterns unchanged and only learns the final category layer. Fine-tuning also adjusts the earlier feature weights after one warmup epoch. We replaced the entire original classifier with one linear 576-to-22 layer. Batch-normalization running statistics stayed fixed in both versions. Three seeds repeat training with different initial category weights and image order, using the same split. Maximum ten epochs, AdamW, and source-validation early stopping were fixed. No random augmentation was used for the published experiment; later optional code does not change that. During inference both models are fixed. Different photographs change the responses, not the weights. The demo’s training comparison explains this distinction visually.

## 6. Follow the numbers. Compare the datasets.

Suggested speaker: **Max**. Suggested time: 120 seconds.

Before class, open ONE-CLICK-LAUNCHERS and choose your OS. Both dataset panels are visible. Choose a PlantVillage thumbnail and Run & show all stages. The full summary stays visible; there is no need to click through tabs. Follow the visuals and short captions. Dimensions and the training comparison are under Model setup & dimensions if needed. Prepared input outlines the resized photo and added padding. Early features respond to local patterns. Middle and late blocks combine them into smaller maps with more channels. These are real responses, not disease masks. Scroll to Pooling: each 7-by-7 late map becomes one average, leaving 576 values. Point to the map → average → 576-value diagram. Use Inspect the numbers only if someone asks for the calculation. Scores multiply those features by learned category weights and add a bias. Use Inspect score contributions if time permits: the five largest contributions plus all remaining contributions reconstruct its raw score. Softmax compares all 22 scores; a high score can still be wrong. Select PlantDoc example 6 for a mistake, or example 5 where only fine-tuning matches the annotation. Start Compare 22 images per dataset; show a few pairs or use zero pause. Identify partial counts as partial. The full balanced sample gives both models 22/22 on PlantVillage and 5/22 on PlantDoc, not the full evaluation shown next. Seed 551 is the first planned seed. The six thumbnails were selected after evaluation; the running sample uses a prediction-independent checksum rule, one photo per category per dataset. Neither is a representative new test. Playback reveals captured values; no weights change while demonstrating. The optional toy filter is explicitly a separate, hand-written teaching example, not a MobileNet filter. Leave it for questions if time is short. If using offline replay, say it is recorded. Aim for two minutes on the essentials; allow a longer walkthrough only if the instructor time limit permits.

## 7. Strong source scores did not transfer

Suggested speaker: **Max**. Suggested time: 45 seconds.

Frozen-feature mean accuracy was 97.59 percent on PlantVillage and 21.68 percent on PlantDoc. Fine-tuned accuracy was 99.07 and 25.87 percent respectively. The PlantDoc improvement was 4.19 percentage points, not 4.19 percent relative improvement. Macro-F1, which gives each category equal weight, shows the same broad pattern. Error bars describe variation between three training seeds on the same test images; they are not confidence intervals for new photographs. We report every seed, not just the best run.

## 8. A recurring mistake worth investigating

Suggested speaker: **Shriyansh**. Suggested time: 35 seconds.

This is one error pattern from the full evaluation, not just our demo examples. Ten of the 183 target images were annotated as tomato late blight. The frozen models predicted that label 61, 70, and 75 times; the fine-tuned models predicted it 59, 54, and 42 times. This identifies a behavior to investigate. It does not tell us the visual or biological cause, and it does not justify changing the dataset labels simply because the model disagrees.

## 9. What these results cannot establish

Suggested speaker: **Shriyansh**. Suggested time: 40 seconds.

We evaluated one architecture, one split and one fixed training configuration. The target test set is small and internet-sourced. We cannot claim general field accuracy or statistical significance from three training seeds. Our uncertainty is not solved by counting each repeated prediction as a new test photograph. The domain shift includes many simultaneous differences, so we cannot attribute the accuracy drop to weather, camera skills or any single cause.

## 10. High familiar-image accuracy is not enough.

Suggested speaker: **Shriyansh**. Suggested time: 35 seconds.

Our contribution is an empirical baseline and reproducibility exercise, not a new architecture or the first study of this dataset shift. We completed the agreed comparison and found a clear gap. Future augmentation or adaptation experiments need their own records and evaluation plan. We should not tune against these already observed test scores and call that a new unseen test. The code, paper, plots and detailed measurements are available in the repository. Invite questions.

## 11. Backup: how someone can repeat the work

Backup slide; use for questions.

Use this slide only for questions. The original saved checkpoints are fingerprinted and require the original code; the demo loads verified historical modules automatically. Full reproduction uses a separate worktree at 9619975 so optional later training changes do not get mixed into the paper. Raw images, models and private notes are not on GitHub. Reproduction commands generate separate records; exact numerical equality across different devices is not guaranteed. The processing display uses temporary observation hooks that copy actual layer outputs. Checks confirm identical category scores with and without those hooks, unchanged weights and buffers, and no retained hooks. Fixed channels and shared stage color scales make the two models easier to compare; these displays do not establish a causal explanation. The running sample is chosen from the original test manifests and recorded with source paths and image hashes in the local replay JSON. The learning display verifies all 576 spatial averages and reconstructs all 22 logits from the actual linear weights and bias. Double-precision explanatory sums can differ slightly from float32 execution. A top-five contribution display also includes the sum of the other 571 features, so nothing is silently omitted. The optional filter lesson is separate from the real model. Headless tests and --no-open-browser avoid launching or focusing a browser window.

## 12. Selected references

Backup slide; use for questions.

These are selected sources, not a replacement for the full literature review. The 2019 reference identifies the architecture; the plant-disease literature review emphasizes papers from 2020 onward. Salman et al. also study cross-dataset performance using a different model and protocol. Do not compare headline scores as if the image lists, categories and training access were identical.
