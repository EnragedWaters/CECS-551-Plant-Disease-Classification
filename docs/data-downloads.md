# Local datasets and pretrained weights

Downloaded and verified October 23, 2026:

| Data | Image files | Readability check |
| --- | ---: | --- |
| PlantVillage color | 54,305 | All decoded successfully |
| PlantDoc train | 2,342 | All decoded successfully |
| PlantDoc test | 236 | All decoded successfully |

The pretrained weight file passed its published hash-prefix check and loaded
into MobileNetV3-Small successfully. No training or prediction scoring was done.
Total local storage is about **4.9 GiB**, including retained ZIP archives;
about **111 GiB** remained free on C: after setup. The repository keeps a small
[copy of the download record](download-manifest.json), containing no images or
model weights. Counts describe this downloaded version, not paper headline
totals or final eligible experiment counts.

The local data folder is `~/datasets/cecs551-plant-disease` in WSL. It is outside
Git and OneDrive. For Ubuntu in WSL, the Windows File Explorer address looks
like this; replace `<your-linux-username>` with your WSL username:

```text
\\wsl.localhost\Ubuntu\home\<your-linux-username>\datasets\cecs551-plant-disease
```

| Folder or file | Contents |
| --- | --- |
| `plantvillage/raw/color/` | Original color images, organized by category |
| `plantvillage/leaf_grouping/leaf-map.json` | Source metadata identifying related leaf images |
| `plantvillage/splits/` | The author's color train/test lists, preserved for review |
| `plantdoc/train/` and `plantdoc/test/` | Cropped classification images in the original partitions |
| `weights/` | MobileNetV3-Small ImageNet starting weights |
| `archives/` | Original dataset ZIP downloads |
| `download-manifest.json` | Download URLs, source revisions, file fingerprints, and check results |

To prepare the same data on another WSL/Linux machine, activate the project
Python environment and run from the repository:

```bash
python scripts/prepare_data.py
```

The script downloads roughly 3 GB and extracts only PlantVillage's color
variant. It checks readability of the images and loads the pretrained weights
without training. Running it again reuses completed downloads, re-extracts the
original files, and repeats the checks. Keep manual edits out of these raw folders.

`scripts/CECS551DatasetsResearch.ipynb` reads these existing folders without
downloading another copy. Use the project Python environment as its notebook
kernel. For data stored elsewhere, edit `DATA_ROOT` in the first cell or set
`PLANT_DISEASE_DATA_DIR` before starting Jupyter. The selected folder should
contain `plantvillage/raw/color`, `plantdoc/train`, and `plantdoc/test`.
Missing folders produce a setup message rather than an automatic download.
The first notebook cell writes the full paths to `dataset-paths.local.txt` in
the notebook's working folder (usually `scripts/` or the repository root).
That file is ignored by Git. It is a local location report, not a configuration
file; change `DATA_ROOT` or the environment variable to choose another location.
The saved notebook output only reports whether the folders were found.

Sources are pinned to these revisions:

- [PlantVillage author-hosted distribution](https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7).
  Its README records CC BY-SA 3.0; retain attribution and consult those terms
  before redistributing images or adaptations.
- [Cropped PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8).
  Its included license is CC BY 4.0.
- [MobileNetV3-Small `IMAGENET1K_V1`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
  The official weight URL comes from the installed torchvision package.

The saved source split lists are not yet our final experiment splits. We still
need to review shared categories, related images, duplicates, and eligibility.
The PlantVillage lists contain 43,596 train and 10,709 test paths: all exist
locally, with no filename overlap. This does not establish absence of related
leaves or duplicate content across the lists.
PlantDoc training images remain unused in the main source-only experiment;
PlantDoc test images must not guide training settings. A successful file check
does not establish correct labels or replace the team's data analysis.
