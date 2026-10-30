# Preliminary image analysis

On October 30, 2026, we checked image sizes and color formats, then looked at
24 example pictures. This adds to the counts and charts in Jacob's notebook
and helps us decide how to prepare the images for the model in Phase 3.
The dataset versions are listed in [the download record](download-manifest.json).
We have not trained the model or finished checking for duplicate images and
incorrect disease labels.

## Image sizes and formats

The script successfully read basic file information, such as width, height, and
color format, for all 56,883 images. This information is called **metadata**.
It does not require loading the whole picture. An earlier check during setup
did fully load every image successfully.

| Image collection / folder | Images | Width range (pixels) | Height range (pixels) | Non-square images |
| --- | ---: | --- | --- | ---: |
| PlantVillage color | 54,305 | 256 only | 256 only | 0 |
| PlantDoc train | 2,342 | 115–6,000 | 85–6,000 | 2,252 |
| PlantDoc test | 236 | 115–6,000 | 69–5,312 | 229 |

PlantVillage pictures are all the same size. PlantDoc pictures vary widely,
and most are wider than they are tall or taller than they are wide.
The width and height ranges are measured separately, so their smallest values
may come from different pictures. These are the sizes stored in the files,
before rotating or resizing them for use.

Most pictures store their colors as RGB: red, green, and blue. A few use other
formats. PlantVillage has one image with an extra transparency layer (RGBA).
PlantDoc train has five with transparency, five using a format commonly used
for printing (CMYK), and one grayscale picture. PlantDoc test is all RGB.
We should convert every picture to RGB so the model receives a consistent
format. This does not restore missing colors to a grayscale picture.

The full PlantVillage collection has 38 categories, ranging from 152 to 5,507
images each. PlantDoc train has 28 categories with 2–180 images each; test has
27 with 4–12 each. These totals include all downloaded categories. We have not
yet finalized which matching categories will be used in the experiment.
The [category review](category-mapping.md) explains the 27 proposed pairs and
why the two training-only spider-mite images are outside that comparison.

## What the sample pictures show

We chose six categories to illustrate both healthy and disease-labeled leaves:
apple scab, apple healthy, grape black rot, potato early blight, tomato early
blight, and tomato healthy. The script randomly selected two different pictures
from each category in each dataset, giving 24 pictures in total. It uses sorted
file lists and a fixed random starting value (seed 7), so running it again on
the same data selects the same pictures.

PlantDoc examples came only from its training folder. We kept every selected
picture, regardless of how it looked. These examples are separate from the
54-picture page in Jacob's notebook. The script selected and displayed them;
the observations below came from visually inspecting them, not automatic
detection of hands, watermarks, or disease symptoms.

**How to find an example:** `S1-1-1` means picture sheet 1, row 1, column 1.
Both sheets appear below. The left two columns show PlantVillage; the right
two show PlantDoc. The labels describe the dataset categories, not diagnoses
we have independently confirmed.

| What we noticed | Examples and why it matters |
| --- | --- |
| The 12 PlantVillage examples show detached leaves on fairly plain backgrounds. Shadows and leaf positions vary. | S1-1-1, S1-2-2, S2-1-2: pictures can be the same size without having the same lighting or leaf position. |
| Some PlantDoc pictures show several leaves or much of a plant. | S1-1-4, S1-2-4, S2-3-4: the category name does not tell us which leaf in the picture to focus on. |
| Some PlantDoc pictures include other objects or overlapping leaves. | A hand appears in S1-1-3; leaves overlap in S1-3-4 and S2-2-4. The model will encounter more than an isolated leaf. |
| Some pictures have visible watermarks or text. | S1-2-3, S1-3-3, S2-1-4: the model will also see markings that are unrelated to the leaf's condition. |
| Pictures have different shapes and show different amounts of the plant. | S2-2-3 is tall and narrow; S1-1-4 shows a wider scene. Cutting every picture into a square could remove useful leaf details. |

These pictures show examples of differences, not how common each difference is
across the datasets. We did not measure blur or lighting, confirm disease labels,
or test how these features affect the model. PlantDoc also has isolated leaves
on plain backgrounds, such as S1-2-3 and S1-3-3; not all its pictures are outdoor
scenes with busy backgrounds.

## What this means for preparing the images

- **Use the same color format:** convert images to RGB. Also adjust their pixel
  values in the way the pretrained model expects; this is called normalization.
- **Use a consistent size without accidentally cutting off the leaf:** compare
  shrinking an image and adding borders with cutting it to fit. Check examples
  from PlantVillage before choosing the rule. Apply the same final rule to both
  datasets, and prepare validation/test images the same way every time.
- **Keep difficult but valid pictures:** record watermarks and busy backgrounds
  as possible complications. Do not remove pictures just because they hurt scores.
- **Check for repeated pictures and photos of the same leaf:** keep related
  pictures together when separating training, validation, and testing data.
  This script lists files and sizes; it does not check whether two pictures
  show the same content.
- **Keep uncertain disease labels under review:** being able to open a picture
  does not mean its category is correct.

## Run the check again

Activate the project environment and run from the repository:

```bash
python scripts/inspect_images.py
```

For another local data location, supply `--data-root /path/to/dataset-root` or
set `PLANT_DISEASE_DATA_DIR`. `--output-dir` changes where the results are saved.
The script reads source files and writes generated outputs; it does not train,
download, delete, or relabel images. Re-running replaces its generated files.

Results are saved in `outputs/image-audit/`, which is ignored by Git. It contains:

- A file list with image sizes and color formats (`image-inventory.csv`).
- Counts for each category (`category-counts.csv`) and a size summary
  (`size-summary.json`).
- Any errors reading the file information (`header-errors.csv`).
- The two picture sheets and a list connecting each picture code to its original
  file, category, and train/test folder (`sample-manifest.csv`).

File locations in these lists are relative to the dataset folder, so they do
not include a personal home-folder path. Someone downloading the GitHub project
will need the datasets and a script run to generate their own picture sheets.

![Twelve examples: apple scab, apple healthy, grape black rot](../outputs/image-audit/sample-sheet-1.png)

![Twelve examples: potato early blight, tomato early blight, tomato healthy](../outputs/image-audit/sample-sheet-2.png)

Picture sources: PlantVillage, Mohanty et al., and PlantDoc, Singh et al.
Dataset versions and stated licenses are recorded in [data-downloads.md](data-downloads.md).
Pictures are resized for display, with watermarks left in place. The picture
list preserves the original folder labels; the sheets use our proposed category
names. We have not independently verified the original source of each picture.
