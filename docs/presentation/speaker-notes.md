# Phase 4 speaker notes

Provisional ~9-minute version including demo; duration and dates must be confirmed. Slides 11–12 are backup.

Suggested portions: Jacob 1–3; Troy 4–5; Max 6–7; Shriyansh 8–10. Slides 11–12 are backup.
Confirm the assigned duration and rehearse. For five minutes, shorten background and use one demo example.

Photo credits and original source paths are included below and in PHOTO_CREDITS.md.

## 1. Plant disease classification

Suggested speaker: **Jacob** · 30 seconds (provisional).

Our project asks whether a plant disease classifier can handle images beyond the collection it learned from. We compared two ways of adapting the same lightweight model. The experiment is complete; this presentation explains our measured results, including where the models fail. The main finding is a large drop on the second dataset, not a claim that we solved plant diagnosis.

Image credits (full frames retained unless identified as model processing):
pd-apple-01: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 2. Will it work on a different collection?

Suggested speaker: **Jacob** · 35 seconds (provisional).

Casual plant owners take photographs with different framing, backgrounds and lighting. These are real dataset photographs, not pictures of study participants. Our question is whether a model trained on PlantVillage transfers to cropped PlantDoc. The two photos share the apple-scab annotation but are different leaves. They illustrate the motivation, not a controlled test of weather or photography skill. A confident wrong category could mislead a user; we investigate that limitation rather than promise a diagnostic service.

Image credits (full frames retained unless identified as model processing):
pv-apple-01: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab 3055.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pd-apple-02: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 3. Two collections. Same 22 labels.

Suggested speaker: **Jacob** · 40 seconds (provisional).

PlantVillage is the source collection: the one we trained on. Cropped PlantDoc is the target collection: the different one used for external testing. We matched 22 plant-condition categories and excluded five uncertain pairs rather than assume similar names meant identical conditions. These labels are dataset annotations, not diagnoses independently verified by us. Cropped PlantDoc is also not a controlled sample of future users. The photographs show two shared annotated conditions: apple scab and grape black rot. Each column is a different collection, not matched views of the same plants. Images are selected illustrations and retain their original framing. The 22 shared categories include ten healthy and twelve named conditions; five uncertain pairs were excluded. PlantDoc is internet-sourced and is not a controlled sample of casual users.

Image credits (full frames retained unless identified as model processing):
pv-apple-01: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab 3055.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pv-grape-01: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Grape___Black_rot/994db94a-4a38-45c0-a36b-98daa4fa18d2___FAM_B.Rot 3383.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pd-apple-01: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pd-grape-01: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/grape leaf black rot/brleaf2_zoom.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 4. Learn first. Test separately.

Suggested speaker: **Troy** · 45 seconds (provisional).

Training fits the model weights. Validation selects the saved checkpoint with the lowest prediction loss, without using the final tests. The two test sets answer different questions: familiar collection versus different collection. We grouped known photos of the same leaf and checked identical files to reduce overlap. Some related or altered duplicates may still escape these checks. PlantDoc test results did not select models or training settings. The first two boxes describe development; the last two are separate final evaluations, not additional training stages. Read the diagram left to right. Training fits the candidate models. Validation selects the saved checkpoints. Only those selected checkpoints reach the two separate test collections. There is no path from PlantDoc test results back to training or selection.

## 5. Same model. Different learning.

Suggested speaker: **Troy** · 50 seconds (provisional).

Pretrained means the model already learned image patterns on ImageNet before our plant experiment. The frozen baseline reuses these patterns unchanged and only learns the final category layer. Fine-tuning also adjusts the earlier feature weights after one warmup epoch. We replaced the entire original classifier with one linear 576-to-22 layer. Batch-normalization running statistics stayed fixed in both versions. Three seeds repeat training with different initial category weights and image order, using the same split. Maximum ten epochs, AdamW, and source-validation early stopping were fixed. No random augmentation was used for the published experiment; later optional code does not change that. During inference both models are fixed. Different photographs change the responses, not the weights. The demo’s training comparison explains this distinction visually. The colored blocks are an editable schematic, not an exact layer-by-layer drawing of MobileNet. Blue means the feature weights remain fixed during plant training; green means weights are allowed to learn. Both versions produce 576 pooled features and 22 category scores. Both saved models stay fixed during demonstration.

Image credits (full frames retained unless identified as model processing):
pv-apple-01: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab 3055.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 6. Watch the image become a prediction

Suggested speaker: **Max** · 120 seconds (provisional).

Open ONE-CLICK-LAUNCHERS and choose your OS before class. This slide is a recorded visual preview of the live demo. The full-frame PlantDoc photo is prepared with the original resize/pad settings; the two small tiles are actual final-block channel 0 responses from seed 551, using one shared color scale. These are not disease masks or attention maps. Both models classify this apple-scab-labeled image incorrectly: frozen predicts potato late blight, fine-tuned predicts tomato early blight. Start the live demo with both dataset panels visible, choose a PlantVillage example and Run & show all stages. Follow the photos and maps; show the map-to-average diagram and score bars. Longer explanations are optional. Then show a PlantDoc mistake. If time permits, run a few pairs in Compare 22 images per dataset and identify the partial sample counts. The balanced 22-pair sample is separate from the published full evaluation. If using the portable replay, explicitly say it is recorded. Do not describe explanatory playback as model thinking time. Allow about two minutes for the main sequence; use the detailed lessons only if time permits.

Image credits (full frames retained unless identified as model processing):
pd-apple-02: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
prepared-apple: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: Original preprocessing display
late-frozen: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: Actual seed-551 final-block channel 0; shared color scale; not an attention map.
late-finetune: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: Actual seed-551 final-block channel 0; shared color scale; not an attention map.

## 7. Familiar-image accuracy did not transfer

Suggested speaker: **Max** · 45 seconds (provisional).

Frozen-feature mean accuracy was 97.59 percent on PlantVillage and 21.68 percent on PlantDoc. Fine-tuned accuracy was 99.07 and 25.87 percent respectively. The PlantDoc improvement was 4.19 percentage points, not 4.19 percent relative improvement. Macro-F1, which gives each category equal weight, shows the same broad pattern. Error bars describe variation between three training seeds on the same test images; they are not confidence intervals for new photographs. We report every seed, not just the best run.

## 8. Three examples. Three different outcomes.

Suggested speaker: **Shriyansh** · 35 seconds (provisional).

These three selected seed-551 examples all carry the dataset annotation apple scab. In the first PlantVillage photo, both models predict apple scab. In the second PlantVillage photo, frozen predicts tomato leaf mold while fine-tuned predicts apple scab. In the PlantDoc photo, frozen predicts potato late blight and fine-tuned predicts tomato early blight. These examples were chosen after evaluation to explain different outcomes; they are not random or representative, and their proportions are not an accuracy estimate. Predictions come from the saved original evaluation and agree in category with the CPU replay. A dataset label is a reference annotation, not our independently verified diagnosis. In the full PlantDoc evaluation, overprediction of tomato late blight was another recurring error: 61–75 times for frozen and 42–59 times for fine-tuned, versus 10 annotated images. That identifies a behavior, not its cause.

Image credits (full frames retained unless identified as model processing):
pv-apple-01: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Apple___Apple_scab/0261a6e4-21f8-481a-8827-b674e6955644___FREC_Scab 3055.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pv-apple-02: S. P. Mohanty / PlantVillage; CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/). Source: https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7. File: plantvillage/raw/color/Apple___Apple_scab/154842bb-6ee3-4176-9007-cf7a43cfe140___FREC_Scab 3434.JPG. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.
pd-apple-02: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/1b321015-6e33-4f18-aade-888f4383fe92.jpeg.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 9. Useful evidence. Limited claims.

Suggested speaker: **Shriyansh** · 40 seconds (provisional).

We evaluated one architecture, one split and one fixed training configuration. The target test set is small and internet-sourced. We cannot claim general field accuracy or statistical significance from three training seeds. Our uncertainty is not solved by counting each repeated prediction as a new test photograph. The domain shift includes many simultaneous differences, so we cannot attribute the accuracy drop to weather, camera skills or any single cause. The three visuals summarize sample size, changes occurring together and uncertain reference labels. The small dot grid is a symbol for a sample, not a plot of individual outcomes. The arrows around the question mark are a schematic of unresolved causes, not measured causal effects.

Image credits (full frames retained unless identified as model processing):
pd-apple-01: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 10. Evaluate beyond the training collection.

Suggested speaker: **Shriyansh** · 35 seconds (provisional).

Our contribution is an empirical baseline and reproducibility exercise, not a new architecture or the first study of this dataset shift. We completed the agreed comparison and found a clear gap. Future augmentation or adaptation experiments need their own records and evaluation plan. We should not tune against these already observed test scores and call that a new unseen test. The code, paper, plots and detailed measurements are available in the repository. Invite questions. The closing photograph is an illustrative PlantDoc image, not an example of the average accuracy. End with the distinction between strong same-dataset testing and demonstrated practical robustness.

Image credits (full frames retained unless identified as model processing):
pd-apple-01: D. Singh et al. / cropped repository by P. Kayal et al.; CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Source: https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8. File: plantdoc/test/Apple Scab Leaf/052609%20Hartman%20Crabapple%20scab%20single%20leaf.JPG.jpg. Changes: EXIF orientation, RGB, fit without cropping, JPEG encoding.

## 11. Backup: follow the evidence

Suggested speaker: **Any speaker** · 0 seconds (provisional).

Use this slide only for questions. The original saved checkpoints are fingerprinted and require the original code; the demo loads verified historical modules automatically. Full reproduction uses a separate worktree at 9619975 so optional later training changes do not get mixed into the paper. Raw images, models and private notes are not on GitHub. Reproduction commands generate separate records; exact numerical equality across different devices is not guaranteed. The processing display uses temporary observation hooks that copy actual layer outputs. Checks confirm identical category scores with and without those hooks, unchanged weights and buffers, and no retained hooks. Fixed channels and shared stage color scales make the two models easier to compare; these displays do not establish a causal explanation. The running sample is chosen from the original test manifests and recorded with source paths and image hashes in the local replay JSON. The learning display verifies all 576 spatial averages and reconstructs all 22 logits from the actual linear weights and bias. Double-precision explanatory sums can differ slightly from float32 execution. A top-five contribution display also includes the sum of the other 571 features, so nothing is silently omitted. The optional filter lesson is separate from the real model. Headless tests and --no-open-browser avoid launching or focusing a browser window. The diagram summarizes the reproduction workflow. See docs/reproduction.md for commands and dependencies. The 72 current software tests include later demo checks; the original paper accurately records 39 checks at its experiment snapshot. No new training was needed for these visuals.

## 12. References & image credits

Suggested speaker: **Any speaker** · 0 seconds (provisional).

These are selected sources, not a replacement for the full literature review. The 2019 reference identifies the architecture; the plant-disease literature review emphasizes papers from 2020 onward. Salman et al. also study cross-dataset performance using a different model and protocol. Do not compare headline scores as if the image lists, categories and training access were identical. Real photograph derivatives are credited in PHOTO_CREDITS.md and assets/manifest.json. PlantVillage photos are attributed to S. P. Mohanty under CC BY-SA 3.0. PlantDoc photos are attributed to D. Singh et al. with cropped repository by P. Kayal et al., under the repository-declared CC BY 4.0. These declarations are not an independent audit of internet-image provenance. Whole frames are retained with orientation correction and resizing; no AI-generated photos, symptom enhancement or background replacement was used. Prepared inputs and feature maps are actual recorded displays and are identified as such.
