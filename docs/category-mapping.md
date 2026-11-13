# Proposed dataset category matches

This review replaces the inclusion decisions on pages 6–10 of `CECS 551 Phase 3_datasets.pdf`. That PDF is the original export and has not been regenerated. Use the decisions below when assembling the report.

All 27 pairs remain in the notebook for exploratory counts and pictures.
For the initial Phase 4 experiment, the ten supported healthy matches and
twelve named-condition matches are selected; the five unresolved pairs are
excluded. The [experiment data guide](experiment-data.md) and
[machine-readable category list](../configs/categories.json) record this
conservative decision and the image-level exclusions. Raw images have not
been deleted or relabeled, and training has not started.

## What the source supports

The PlantDoc paper, Section 3.1, says its collection used scientific and common names from the 38 PlantVillage categories. Figure 2 explicitly names ten healthy categories. Together with the corresponding repository folders, this supports the proposed healthy matches below; they no longer rely solely on guessing that “leaf” means “healthy.” This is a category interpretation, not verification of every image label. [PlantDoc paper, page 2, Section 3.1 and Figure 2](https://arxiv.org/pdf/1911.10317v1).

Twelve other pairs use corresponding condition names in the folder labels. Five broader disease labels still need confirmation. In particular, the paper also uses broad names for rust, corn blight, and tomato yellow virus. Its bacterial bell-pepper example in Figure 1 does not by itself establish the meaning of every image in the leaf-spot folder.

## Proposed mapping

“Named condition” means the category names correspond, not that the images or
their annotations have been independently validated. The table preserves the
Phase 3 evidence review; Phase 4 inclusion follows the category list linked above.

| PlantVillage folder | PlantDoc folder | Basis / current decision |
| --- | --- | --- |
| `Apple___Apple_scab` | `Apple Scab Leaf` | Named condition; proposed match. |
| `Apple___Cedar_apple_rust` | `Apple rust leaf` | Needs confirmation: PlantDoc says rust, but does not specify cedar apple rust. |
| `Apple___healthy` | `Apple leaf` | Healthy category named in Figure 2; proposed match. |
| `Blueberry___healthy` | `Blueberry leaf` | Healthy category named in Figure 2; proposed match. |
| `Cherry_(including_sour)___healthy` | `Cherry leaf` | Healthy category named in Figure 2; proposed match. |
| `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | `Corn Gray leaf spot` | Named condition; proposed match. |
| `Corn_(maize)___Common_rust_` | `Corn rust leaf` | Needs confirmation: PlantDoc says rust, but does not specify common rust. |
| `Corn_(maize)___Northern_Leaf_Blight` | `Corn leaf blight` | Intended match is supported by the collection method, but folder contents need review: filenames refer to northern blight and other diseases. See the evidence below. |
| `Grape___Black_rot` | `grape leaf black rot` | Named condition; proposed match. |
| `Grape___healthy` | `grape leaf` | Healthy category named in Figure 2; proposed match. |
| `Peach___healthy` | `Peach leaf` | Healthy category named in Figure 2; proposed match. |
| `Pepper,_bell___Bacterial_spot` | `Bell_pepper leaf spot` | Needs confirmation: PlantDoc says leaf spot. Figure 1 shows a bacterial example, but does not establish that the entire folder is bacterial spot. |
| `Pepper,_bell___healthy` | `Bell_pepper leaf` | Healthy category named in Figure 2; proposed match. |
| `Potato___Early_blight` | `Potato leaf early blight` | Named condition; proposed match. |
| `Potato___Late_blight` | `Potato leaf late blight` | Named condition; proposed match. |
| `Raspberry___healthy` | `Raspberry leaf` | Healthy category named in Figure 2; proposed match. |
| `Soybean___healthy` | `Soyabean leaf` | Healthy category named in Figure 2; proposed match. |
| `Squash___Powdery_mildew` | `Squash Powdery mildew leaf` | Named condition; proposed match. |
| `Strawberry___healthy` | `Strawberry leaf` | Healthy category named in Figure 2; proposed match. |
| `Tomato___Bacterial_spot` | `Tomato leaf bacterial spot` | Named condition; proposed match. |
| `Tomato___Early_blight` | `Tomato Early blight leaf` | Named condition; proposed match. |
| `Tomato___Late_blight` | `Tomato leaf late blight` | Named condition; proposed match. |
| `Tomato___Leaf_Mold` | `Tomato mold leaf` | Named condition; proposed match. |
| `Tomato___Septoria_leaf_spot` | `Tomato Septoria leaf spot` | Named condition; proposed match. |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | `Tomato leaf yellow virus` | Needs confirmation: PlantDoc says yellow virus, but does not explicitly identify yellow leaf curl virus. |
| `Tomato___Tomato_mosaic_virus` | `Tomato leaf mosaic virus` | Named condition; proposed match. |
| `Tomato___healthy` | `Tomato leaf` | Healthy category named in Figure 2; proposed match. |

## Corn leaf blight: intended meaning and image-label questions

The PlantDoc authors collected images using PlantVillage's scientific and common
category names (Section 3.1). That supports Northern Leaf Blight as the intended
counterpart of `Corn leaf blight`; different spelling alone is not evidence of
a wrong match. This is our interpretation of the collection method, not an
explicit author-provided folder-to-folder mapping.

A filename review on October 30 found a more concrete question in the pinned
local PlantDoc download. The folder contains 180 train and 12 test images.
Examples include:

| Split | Filename | What it suggests checking |
| --- | --- | --- |
| train | `Northern-Corn-Leaf-Blight.jpg` | Supports the intended Northern Leaf Blight interpretation. |
| train | `Southern-Corn-Leaf-Blight-USDA.ashx?w=770.jpg` | Names southern rather than northern blight. |
| train | `Stewarts-Leaf-Blight-Phase-24.jpg` | Names another leaf-blight condition. |
| test | `2013Corn_GrayLeafSpot_0815_0003.JPG.jpg` | Names gray leaf spot. |
| test | `corn-disease-update-fig-3-gray-leaf-spot.jpg` | Names gray leaf spot. |
| test | `corn-gray-leaf-spot-f4.jpg` | Names gray leaf spot. |

[The file review record](corn-blight-review.csv) includes these and two additional
examples, their source paths, SHA-256 fingerprints, and links to the pinned
author repository. It is a targeted list of filename clues, not a count of
confirmed labeling errors or a complete review of the dataset.

Filenames are not diagnoses: they can be misleading or describe an article
rather than the specific pictured leaf. We have not verified these images'
original source captions or disease labels. The conflicting names therefore
justify review; they do not justify automatically deleting or relabeling files.
The test-file examples also mean we cannot clear this concern merely by noting
that PlantDoc training images are unused in our main experiment.

For Phase 3, record this as a specific label-quality risk. Keep the pair visible
in exploratory counts, but leave final experiment inclusion pending a review of
original sources/annotations. If that review cannot support inclusion, exclude
the pair consistently from the shared experiment categories and document the
reason before training/model selection. Keep raw files unchanged, and record any
future exclusions in a separate manifest. Do not use model predictions or test
scores to decide which annotations to change.

## Before finalizing the experiment

- Find author documentation or annotation information supporting each broader label’s specific meaning; record the source for any approved match.
- If a match remains unresolved, leave it out of the main comparison and document why. Do not choose mappings based on which ones produce better test scores.
- Check related leaves, duplicate content, image eligibility, and test coverage separately. Matching category names does not resolve those issues.
- Keep the spider-mite pair out of the current proposed external comparison: this cropped PlantDoc version has two training images and zero test images in that folder. The lack of target test examples is the relevant reason; PlantDoc training counts do not determine whether we can train on PlantVillage.

## Test coverage for these proposed pairs

The notebook now shows PlantDoc train and test counts separately. Across all
27 proposed pairs, there are 2,340 train images and 236 test images, with only
4–12 test images per category. The two remaining downloaded PlantDoc images
belong to the excluded spider-mite training folder. The chart compares total
dataset sizes; its PlantDoc bars include both train and test images.

Our main experiment trains on PlantVillage, so PlantDoc training counts do not
determine whether a category can be trained. The small PlantDoc test counts do
limit how confidently we can interpret results for individual categories:
a single mistake can noticeably change a category's score. These are observed
folder counts, before final category decisions and duplicate checks, not a
final approved test set or measured model results.

## Replacement paragraph for the report

We identified 27 proposed category pairs for exploratory comparison. The PlantDoc paper supports the interpretation of ten healthy categories, and twelve other pairs use corresponding condition names. Five pairs involving apple rust, corn rust, corn leaf blight, bell-pepper leaf spot, and tomato yellow virus still require review. The collection method supports Northern Leaf Blight as the intended corn-blight match, but some files in that folder have names referring to other diseases, including in the test partition. These are label-quality clues, not confirmed mislabeling. Counts and example pictures remain visible for inspection, while final inclusion is pending the documented review. Pairs that cannot be supported will be excluded from the main comparison. Category matching does not by itself establish image-label accuracy or rule out duplicates.

## Dataset sources

- [PlantVillage color folders](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color). Exact labels were checked against the pinned local download recorded in `download-manifest.json`.
- [Cropped PlantDoc, downloaded revision](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8).
- Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020; the linked author preprint was posted in 2019. [Publication DOI](https://doi.org/10.1145/3371158.3371196).
